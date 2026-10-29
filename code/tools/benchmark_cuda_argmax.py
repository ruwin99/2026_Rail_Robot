from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from rust_detector import (  # noqa: E402
    STUDENT_OUTPUT_SHAPE,
    STUDENT_PROFILE,
    RustDetector,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the existing raw-logits D2H path with the same TensorRT "
            "plan plus an external CUDA argmax kernel. No camera or UART is opened."
        )
    )
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        required=True,
        help="image file or directory; repeat for multiple corpus roots",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def collect_images(roots: list[Path]) -> list[Path]:
    images: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
            images.add(root)
        elif root.is_dir():
            images.update(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    if not images:
        raise ValueError("No readable image paths were found in the corpus roots.")
    return sorted(images)


def load_roi(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read corpus image: {path}")
    if image.shape[0] < 240 or image.shape[1] != 1280:
        raise ValueError(
            f"Corpus image must be at least 1280x240, got "
            f"{image.shape[1]}x{image.shape[0]}: {path}"
        )
    return np.ascontiguousarray(image[:240, :1280])


def benchmark(detector: RustDetector, roi: np.ndarray, warmup: int, iterations: int):
    for _ in range(warmup):
        detector.detect(roi)
    samples_ms = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        detector.detect(roi)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    samples = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "fps": float(1000.0 / samples.mean()),
    }


def copy_logits_to_device(detector: RustDetector, logits: np.ndarray) -> None:
    detector._cuda_call(
        detector._cudart.cudaMemcpy(
            detector._device_output,
            logits.ctypes.data,
            logits.nbytes,
            detector._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        ),
        "cudaMemcpy(synthetic logits)",
    )


def verify_kernel_contract(detector: RustDetector) -> None:
    logits = np.zeros(STUDENT_OUTPUT_SHAPE, dtype=np.float32)
    logits[:, 1, :, :] = 2.0
    logits[:, 2, 0, 0] = 2.0  # first tie must remain class 1.
    logits[:, 3, 0, 1] = 3.0
    copy_logits_to_device(detector, logits)
    class_map = detector._gpu_postprocessor.process(
        detector._device_output,
        detector._stream,
    )
    reference = np.argmax(logits[0], axis=0).astype(np.uint8)
    if not np.array_equal(class_map, reference):
        mismatch = int(np.count_nonzero(class_map != reference))
        raise RuntimeError(f"Synthetic first-tie argmax mismatch: {mismatch} pixels")

    for label, value in (
        ("NaN", np.nan),
        ("positive infinity", np.inf),
        ("negative infinity", -np.inf),
    ):
        logits[0, 0, 0, 0] = value
        copy_logits_to_device(detector, logits)
        try:
            detector._gpu_postprocessor.process(
                detector._device_output,
                detector._stream,
            )
        except ValueError as exc:
            if "NaN or infinity" not in str(exc):
                raise
        else:
            raise RuntimeError(f"CUDA argmax did not reject {label}.")
        logits[0, 0, 0, 0] = 0.0


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be nonnegative and --iterations positive.")
    image_paths = collect_images(args.corpus)
    rois = [(path, load_roi(path)) for path in image_paths]

    raw_maps: list[np.ndarray] = []
    raw_controls: list[bool] = []
    raw_detector = RustDetector(
        args.engine,
        STUDENT_PROFILE,
        args.engine_sha256,
        gpu_argmax=False,
    )
    try:
        for _path, roi in rois:
            result = raw_detector.detect(roi)
            raw_maps.append(result.class_map.copy())
            raw_controls.append(bool(np.isin(result.class_map, (2, 3)).any()))
        raw_timing = benchmark(
            raw_detector,
            rois[0][1],
            args.warmup,
            args.iterations,
        )
        raw_method = raw_detector.method
    finally:
        raw_detector.close()

    mismatch_pixels = 0
    mismatch_images = 0
    control_mismatches = 0
    gpu_detector = RustDetector(
        args.engine,
        STUDENT_PROFILE,
        args.engine_sha256,
        gpu_argmax=True,
    )
    try:
        verify_kernel_contract(gpu_detector)
        for index, (_path, roi) in enumerate(rois):
            result = gpu_detector.detect(roi)
            mismatch = int(np.count_nonzero(result.class_map != raw_maps[index]))
            mismatch_pixels += mismatch
            mismatch_images += int(mismatch > 0)
            control = bool(np.isin(result.class_map, (2, 3)).any())
            control_mismatches += int(control != raw_controls[index])
        gpu_timing = benchmark(
            gpu_detector,
            rois[0][1],
            args.warmup,
            args.iterations,
        )
        gpu_method = gpu_detector.method
        gpu_d2h_nbytes = gpu_detector._gpu_postprocessor.d2h_nbytes
    finally:
        gpu_detector.close()

    raw_d2h_nbytes = int(np.prod(STUDENT_OUTPUT_SHAPE)) * np.dtype(np.float32).itemsize
    print(f"corpus_images={len(rois)}")
    print(f"compared_pixels={len(rois) * raw_maps[0].size}")
    print(f"class_map_mismatch_pixels={mismatch_pixels}")
    print(f"class_map_mismatch_images={mismatch_images}")
    print(f"control_mismatches={control_mismatches}")
    print("synthetic_finite_gate=PASS(NaN,+Inf,-Inf rejected)")
    print(f"raw_method={raw_method}")
    print(f"gpu_method={gpu_method}")
    print(f"raw_d2h_bytes={raw_d2h_nbytes}")
    print(f"gpu_d2h_bytes={gpu_d2h_nbytes}")
    print(
        "d2h_reduction_percent="
        f"{(1.0 - gpu_d2h_nbytes / raw_d2h_nbytes) * 100.0:.6f}"
    )
    for label, timing in (("raw", raw_timing), ("gpu", gpu_timing)):
        print(
            f"{label}_mean_ms={timing['mean_ms']:.6f} "
            f"{label}_p50_ms={timing['p50_ms']:.6f} "
            f"{label}_p95_ms={timing['p95_ms']:.6f} "
            f"{label}_p99_ms={timing['p99_ms']:.6f} "
            f"{label}_fps={timing['fps']:.6f}"
        )
    if mismatch_pixels or control_mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
