from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, 240, 1280)
OUTPUT_CONTRACTS = {
    "rust_class_map": ((1, 240, 1280), np.dtype(np.uint8)),
    "rust_class_counts": ((4,), np.dtype(np.int32)),
    "rust_poor_severe": ((1,), np.dtype(np.uint8)),
    "crack_candidate_map": ((1, 128, 1280), np.dtype(np.uint8)),
    "crack_candidate_pixels": ((1,), np.dtype(np.int32)),
    "crack_probability_threshold": ((1,), np.dtype(np.float32)),
    "multitask_outputs_finite": ((1,), np.dtype(np.uint8)),
}


def cuda_call(cudart: Any, result: tuple[Any, ...], operation: str) -> Any:
    if not result:
        raise RuntimeError(f"{operation} returned no CUDA status")
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with CUDA error {result[0]}")
    return None if len(result) == 1 else result[1]


def host_array(pointer: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    byte_count = int(np.prod(shape)) * dtype.itemsize
    byte_pointer = ctypes.cast(
        ctypes.c_void_p(int(pointer)), ctypes.POINTER(ctypes.c_ubyte)
    )
    return np.ctypeslib.as_array(byte_pointer, shape=(byte_count,)).view(dtype).reshape(shape)


def validate_outputs(
    outputs: dict[str, np.ndarray], expected_probability_threshold: float
) -> dict[str, Any]:
    rust_map = outputs["rust_class_map"][0]
    rust_counts = outputs["rust_class_counts"].astype(np.int64, copy=False)
    measured_rust = np.bincount(rust_map.ravel(), minlength=4)
    if not np.isin(rust_map, (0, 1, 2, 3)).all():
        raise RuntimeError("rust_class_map contains an invalid class ID")
    if not np.array_equal(rust_counts, measured_rust):
        raise RuntimeError("rust_class_counts does not match rust_class_map")
    if int(rust_counts.sum()) != 240 * 1280:
        raise RuntimeError("rust_class_counts does not cover 307200 pixels")
    expected_blocked = int(rust_counts[2] + rust_counts[3] > 0)
    if int(outputs["rust_poor_severe"][0]) != expected_blocked:
        raise RuntimeError("rust_poor_severe does not match rust counts")

    crack_map = outputs["crack_candidate_map"][0]
    crack_pixels = int(outputs["crack_candidate_pixels"][0])
    if not np.isin(crack_map, (0, 1)).all():
        raise RuntimeError("crack_candidate_map contains a value other than zero or one")
    if crack_pixels != int(np.count_nonzero(crack_map)):
        raise RuntimeError("crack_candidate_pixels does not match crack_candidate_map")
    threshold = float(outputs["crack_probability_threshold"][0])
    if not np.isclose(
        threshold, expected_probability_threshold, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError(
            "embedded crack threshold is "
            f"{threshold}, expected {expected_probability_threshold}"
        )
    if int(outputs["multitask_outputs_finite"][0]) != 1:
        raise RuntimeError("finite input produced a non-finite output flag")
    return {
        "rust_counts": rust_counts.tolist(),
        "rust_poor_severe": expected_blocked,
        "crack_candidate_pixels": crack_pixels,
        "crack_probability_threshold": threshold,
        "multitask_outputs_finite": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--expected-crack-probability-threshold", type=float, default=0.99
    )
    parser.add_argument("--expected-crack-min-component-pixels", type=int, default=1024)
    args = parser.parse_args()

    if not 0.0 < args.expected_crack_probability_threshold < 1.0:
        parser.error(
            "--expected-crack-probability-threshold must be between zero and one"
        )
    if args.expected_crack_min_component_pixels <= 0:
        parser.error("--expected-crack-min-component-pixels must be positive")

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("Set NVIDIA_TF32_OVERRIDE=0 before starting this process")

    import tensorrt as trt
    from cuda import cudart

    if not str(trt.__version__).startswith("10.3."):
        raise RuntimeError(f"TensorRT 10.3.x is required, got {trt.__version__}")
    serialized = args.plan.expanduser().resolve().read_bytes()
    actual_sha = hashlib.sha256(serialized).hexdigest()
    if actual_sha != args.sha256.lower():
        raise RuntimeError(f"plan SHA-256 mismatch: {actual_sha}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the plan")

    expected_names = {INPUT_NAME, *OUTPUT_CONTRACTS}
    names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    if engine.num_io_tensors != 8 or names != expected_names:
        raise RuntimeError(f"unexpected I/O tensor set: {sorted(names)}")

    contracts = {INPUT_NAME: (INPUT_SHAPE, np.dtype(np.float32)), **OUTPUT_CONTRACTS}
    io_summary: dict[str, Any] = {}
    for name, (shape, dtype) in contracts.items():
        expected_mode = (
            trt.TensorIOMode.INPUT if name == INPUT_NAME else trt.TensorIOMode.OUTPUT
        )
        actual_shape = tuple(engine.get_tensor_shape(name))
        actual_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        if engine.get_tensor_mode(name) != expected_mode:
            raise RuntimeError(f"{name}: wrong input/output mode")
        if actual_shape != shape or actual_dtype != dtype:
            raise RuntimeError(
                f"{name}: expected {dtype} {shape}, got {actual_dtype} {actual_shape}"
            )
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise RuntimeError(f"{name}: TensorRT format is not LINEAR")
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise RuntimeError(f"{name}: TensorRT location is not DEVICE")
        io_summary[name] = {
            "mode": "input" if name == INPUT_NAME else "output",
            "shape": list(shape),
            "dtype": str(dtype),
            "format": "LINEAR",
            "location": "DEVICE",
        }

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an execution context")
    stream = cuda_call(cudart, cudart.cudaStreamCreate(), "cudaStreamCreate")
    host_pointers: dict[str, Any] = {}
    host_tensors: dict[str, np.ndarray] = {}
    device_pointers: dict[str, Any] = {}
    try:
        for name, (shape, dtype) in contracts.items():
            nbytes = int(np.prod(shape)) * dtype.itemsize
            host_pointer = cuda_call(
                cudart, cudart.cudaHostAlloc(nbytes, 0), f"cudaHostAlloc({name})"
            )
            device_pointer = cuda_call(
                cudart, cudart.cudaMalloc(nbytes), f"cudaMalloc({name})"
            )
            host_pointers[name] = host_pointer
            host_tensors[name] = host_array(host_pointer, shape, dtype)
            device_pointers[name] = device_pointer
            if not context.set_tensor_address(name, int(device_pointer)):
                raise RuntimeError(f"TensorRT rejected tensor address {name}")

        def infer(value: float) -> dict[str, np.ndarray]:
            host_tensors[INPUT_NAME].fill(value)
            input_tensor = host_tensors[INPUT_NAME]
            cuda_call(
                cudart,
                cudart.cudaMemcpyAsync(
                    device_pointers[INPUT_NAME],
                    input_tensor.ctypes.data,
                    input_tensor.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync(input)",
            )
            if not context.execute_async_v3(stream_handle=int(stream)):
                raise RuntimeError("execute_async_v3 returned false")
            for output_name in OUTPUT_CONTRACTS:
                output = host_tensors[output_name]
                cuda_call(
                    cudart,
                    cudart.cudaMemcpyAsync(
                        output.ctypes.data,
                        device_pointers[output_name],
                        output.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    f"cudaMemcpyAsync({output_name})",
                )
            cuda_call(cudart, cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            return {name: host_tensors[name].copy() for name in OUTPUT_CONTRACTS}

        normal_summary = validate_outputs(
            infer(0.0), args.expected_crack_probability_threshold
        )
        nonfinite_summary: dict[str, int] = {}
        for label, value in (
            ("nan", np.nan),
            ("positive_infinity", np.inf),
            ("negative_infinity", -np.inf),
        ):
            outputs = infer(float(value))
            flag = int(outputs["multitask_outputs_finite"][0])
            if flag != 0:
                raise RuntimeError(f"{label} input was not rejected; finite flag={flag}")
            nonfinite_summary[label] = flag

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "tensorrt_version": str(trt.__version__),
                    "plan": str(args.plan.expanduser().resolve()),
                    "plan_sha256": actual_sha,
                    "io_tensor_count": engine.num_io_tensors,
                    "io": io_summary,
                    "finite_zero_input": normal_summary,
                    "nonfinite_inputs": nonfinite_summary,
                    "external_crack_min_component_pixels": (
                        args.expected_crack_min_component_pixels
                    ),
                },
                indent=2,
            )
        )
        return 0
    finally:
        for pointer in device_pointers.values():
            cuda_call(cudart, cudart.cudaFree(pointer), "cudaFree")
        cuda_call(cudart, cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
        for pointer in host_pointers.values():
            cuda_call(cudart, cudart.cudaFreeHost(pointer), "cudaFreeHost")


if __name__ == "__main__":
    raise SystemExit(main())
