from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


INPUT_NAME = "images"
RAW_OUTPUT_NAME = "logits"
RUST_MAP_NAME = "rust_class_map"
LOGITS_NOT_NAN_NAME = "rust_logits_not_nan"
LOGITS_FINITE_ABS_NAME = "rust_logits_finite_abs"
INPUT_SHAPE = [1, 3, 240, 1280]
RAW_OUTPUT_SHAPE = [1, 4, 240, 1280]
FLOAT32_MAX = np.float32(np.finfo(np.float32).max)
EXPECTED_SOURCE_SHA256 = (
    "fd3bcedb29d7fad94d072b6fb4bf2cf7c15b8c9e699091b83f759b89529dae8f"
)


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def _initializer_digests(model: onnx.ModelProto) -> dict[str, str]:
    return {
        value.name: hashlib.sha256(value.SerializeToString()).hexdigest()
        for value in model.graph.initializer
    }


def add_rust_postprocessor(model: onnx.ModelProto, source_sha256: str) -> None:
    graph = model.graph
    if len(graph.input) != 1 or graph.input[0].name != INPUT_NAME:
        raise ValueError("Source ONNX input must be exactly 'images'.")
    if _shape(graph.input[0]) != INPUT_SHAPE:
        raise ValueError(f"Source input shape must be {INPUT_SHAPE}.")
    if graph.input[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Source images input must be FLOAT.")
    if len(graph.output) != 1 or graph.output[0].name != RAW_OUTPUT_NAME:
        raise ValueError("Source ONNX output must be exactly 'logits'.")
    if _shape(graph.output[0]) != RAW_OUTPUT_SHAPE:
        raise ValueError(f"Source logits shape must be {RAW_OUTPUT_SHAPE}.")
    if graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Source logits output must be FLOAT.")

    graph.initializer.extend(
        [
            _initializer("deploy/all_axes", np.array([0, 1, 2, 3], dtype=np.int64)),
            _initializer("deploy/vector_axis", np.array([0], dtype=np.int64)),
            _initializer("deploy/float32_max", np.array(FLOAT32_MAX, dtype=np.float32)),
        ]
    )

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "ArgMax",
            [RAW_OUTPUT_NAME],
            ["deploy/rust_class_i64"],
            name="deploy/rust_argmax",
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast",
            ["deploy/rust_class_i64"],
            ["deploy/rust_class_i32"],
            name="deploy/rust_class_i32",
            to=TensorProto.INT32,
        ),
        # TensorRT 10.3 reliably imports the same INT32 -> FLOAT -> UINT8 path
        # already used by the deployed multitask wrapper.
        helper.make_node(
            "Cast",
            ["deploy/rust_class_i32"],
            ["deploy/rust_class_float"],
            name="deploy/rust_class_float",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Cast",
            ["deploy/rust_class_float"],
            [RUST_MAP_NAME],
            name="deploy/rust_class_u8",
            to=TensorProto.UINT8,
        ),
    ]

    nodes.extend(
        [
            helper.make_node(
                "Abs",
                [RAW_OUTPUT_NAME],
                ["deploy/absolute_logits"],
                name="deploy/absolute_logits",
            ),
            helper.make_node(
                "LessOrEqual",
                ["deploy/absolute_logits", "deploy/float32_max"],
                ["deploy/finite_abs_bool"],
                name="deploy/finite_abs_compare",
            ),
            helper.make_node(
                "Cast",
                ["deploy/finite_abs_bool"],
                ["deploy/finite_abs_float"],
                name="deploy/finite_abs_float",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "ReduceMin",
                ["deploy/finite_abs_float", "deploy/all_axes"],
                ["deploy/finite_abs_scalar"],
                name="deploy/finite_abs_reduce",
                keepdims=0,
            ),
            helper.make_node(
                "Unsqueeze",
                ["deploy/finite_abs_scalar", "deploy/vector_axis"],
                ["deploy/finite_abs_vector"],
                name="deploy/finite_abs_unsqueeze",
            ),
            helper.make_node(
                "Cast",
                ["deploy/finite_abs_vector"],
                [LOGITS_FINITE_ABS_NAME],
                name="deploy/finite_abs_u8",
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "Equal",
                [RAW_OUTPUT_NAME, RAW_OUTPUT_NAME],
                ["deploy/not_nan_bool"],
                name="deploy/not_nan_self_equal",
            ),
            helper.make_node(
                "Cast",
                ["deploy/not_nan_bool"],
                ["deploy/not_nan_float"],
                name="deploy/not_nan_float",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "ReduceMin",
                ["deploy/not_nan_float", "deploy/all_axes"],
                ["deploy/not_nan_scalar"],
                name="deploy/not_nan_reduce",
                keepdims=0,
            ),
            helper.make_node(
                "Unsqueeze",
                ["deploy/not_nan_scalar", "deploy/vector_axis"],
                ["deploy/not_nan_vector"],
                name="deploy/not_nan_unsqueeze",
            ),
            helper.make_node(
                "Cast",
                ["deploy/not_nan_vector"],
                [LOGITS_NOT_NAN_NAME],
                name="deploy/not_nan_u8",
                to=TensorProto.UINT8,
            ),
        ]
    )
    graph.node.extend(nodes)

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                RUST_MAP_NAME, TensorProto.UINT8, [1, 240, 1280]
            ),
            helper.make_tensor_value_info(LOGITS_NOT_NAN_NAME, TensorProto.UINT8, [1]),
            helper.make_tensor_value_info(
                LOGITS_FINITE_ABS_NAME, TensorProto.UINT8, [1]
            ),
        ]
    )

    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            "rail_robot.variant": "rust-only-optimized-compact-v2",
            "rail_robot.source_sha256": source_sha256,
            "rail_robot.input_preprocessing": (
                "external RGB float32 0..1, ImageNet mean/std"
            ),
            "rail_robot.rust_class_order": "Good,Fair,Poor,Severe",
            "rail_robot.rust_postprocess": "raw argmax channels 0:4",
            "rail_robot.finite_guard": (
                "separate all-logits not-NaN and abs<=FLOAT32_MAX uint8 flags"
            ),
        }
    )
    helper.set_model_props(model, properties)


def build_optimized_model(source: Path, output: Path) -> tuple[str, str]:
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "Refusing to wrap an unapproved realtime rust ONNX: "
            f"expected SHA-256 {EXPECTED_SOURCE_SHA256}, got {source_sha256}."
        )
    model = onnx.load_from_string(source_bytes)
    onnx.checker.check_model(model)
    original_initializers = _initializer_digests(model)

    add_rust_postprocessor(model, source_sha256)
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)
    wrapped_initializers = _initializer_digests(model)
    for name, digest in original_initializers.items():
        if wrapped_initializers.get(name) != digest:
            raise RuntimeError(f"Source initializer changed while wrapping: {name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output))
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    return source_sha256, output_sha256


def verify_with_onnxruntime(source: Path, optimized: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for --verify-onnxruntime.") from exc

    rng = np.random.default_rng(20260815)
    tensor = rng.standard_normal(INPUT_SHAPE, dtype=np.float32)
    source_session = ort.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    )
    optimized_session = ort.InferenceSession(
        str(optimized), providers=["CPUExecutionProvider"]
    )
    logits = source_session.run([RAW_OUTPUT_NAME], {INPUT_NAME: tensor})[0]
    output_names = [item.name for item in optimized_session.get_outputs()]
    outputs = dict(
        zip(
            output_names,
            optimized_session.run(output_names, {INPUT_NAME: tensor}),
            strict=True,
        )
    )

    reference_map = np.argmax(logits, axis=1).astype(np.uint8)
    if not np.array_equal(outputs[RUST_MAP_NAME], reference_map):
        mismatch = int(np.count_nonzero(outputs[RUST_MAP_NAME] != reference_map))
        raise RuntimeError(
            f"Optimized rust class map differs at {mismatch} pixels."
        )
    expected_not_nan = np.array([int(not np.isnan(logits).any())], dtype=np.uint8)
    if not np.array_equal(outputs[LOGITS_NOT_NAN_NAME], expected_not_nan):
        raise RuntimeError("Optimized raw-logit not-NaN flag is incorrect.")
    expected_finite_abs = np.array(
        [int(np.less_equal(np.abs(logits), FLOAT32_MAX).all())], dtype=np.uint8
    )
    if not np.array_equal(outputs[LOGITS_FINITE_ABS_NAME], expected_finite_abs):
        raise RuntimeError("Optimized raw-logit finite-absolute flag is incorrect.")
    print(f"onnxruntime_argmax_match_pixels={reference_map.size}/{reference_map.size}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Append a GPU rust argmax map and separate finite-safety flags to "
            "the exact four-logit realtime rust ONNX."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-onnxruntime", action="store_true")
    args = parser.parse_args()

    source_sha256, output_sha256 = build_optimized_model(args.source, args.output)
    if args.verify_onnxruntime:
        verify_with_onnxruntime(args.source, args.output)
    print(f"source_sha256={source_sha256}")
    print(f"output={args.output.resolve()}")
    print(f"output_sha256={output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
