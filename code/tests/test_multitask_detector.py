from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from multitask_detector import (  # noqa: E402
    CRACK_MAP_NAME,
    CRACK_PIXELS_NAME,
    CRACK_THRESHOLD_NAME,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUTS_FINITE_NAME,
    OUTPUT_CONTRACTS,
    RAW_LOGITS_NAME,
    RUST_BLOCKED_NAME,
    RUST_COUNTS_NAME,
    RUST_MAP_NAME,
    OptimizedMultitaskDetector,
    results_from_postprocessed_outputs,
)


def _valid_outputs() -> dict[str, np.ndarray]:
    rust_map = np.zeros((1, 240, 1280), dtype=np.uint8)
    rust_map[0, 10:20, 30:40] = 2
    crack_map = np.zeros((1, 128, 1280), dtype=np.uint8)
    crack_map[0, 5:8, 10:20] = 1
    return {
        RUST_MAP_NAME: rust_map,
        RUST_COUNTS_NAME: np.array(
            [240 * 1280 - 100, 0, 100, 0], dtype=np.int32
        ),
        RUST_BLOCKED_NAME: np.array([1], dtype=np.uint8),
        CRACK_MAP_NAME: crack_map,
        CRACK_PIXELS_NAME: np.array([30], dtype=np.int32),
        CRACK_THRESHOLD_NAME: np.array([0.5], dtype=np.float32),
        OUTPUTS_FINITE_NAME: np.array([1], dtype=np.uint8),
    }


class PostprocessedOutputTests(unittest.TestCase):
    def test_builds_results_without_raw_logits(self) -> None:
        rust, crack = results_from_postprocessed_outputs(
            _valid_outputs(), "rust/method", "crack/method", 0.5, 20
        )

        self.assertEqual(rust.class_map.dtype, np.uint8)
        self.assertAlmostEqual(rust.rust_ratio, 100 / (240 * 1280))
        self.assertEqual(rust.class_ratios["Poor"], rust.rust_ratio)
        self.assertTrue(crack.detected)
        self.assertEqual(crack.crack_pixels, 30)
        self.assertEqual(crack.mask.shape, (128, 1280))

    def test_rejects_mismatched_control_scalars_fail_closed(self) -> None:
        cases = (
            (RUST_COUNTS_NAME, np.zeros((4,), dtype=np.int32), "counts"),
            (RUST_BLOCKED_NAME, np.array([0], dtype=np.uint8), "flag"),
            (CRACK_PIXELS_NAME, np.array([29], dtype=np.int32), "count"),
            (
                CRACK_THRESHOLD_NAME,
                np.array([0.6], dtype=np.float32),
                "threshold",
            ),
            (OUTPUTS_FINITE_NAME, np.array([0], dtype=np.uint8), "NaN"),
        )
        for name, value, message in cases:
            with self.subTest(name=name):
                outputs = _valid_outputs()
                outputs[name] = value
                with self.assertRaisesRegex(ValueError, message):
                    results_from_postprocessed_outputs(
                        outputs, "rust", "crack", 0.5, 20
                    )

    def test_rejects_float_or_nonbinary_maps(self) -> None:
        outputs = _valid_outputs()
        outputs[CRACK_MAP_NAME] = outputs[CRACK_MAP_NAME].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            results_from_postprocessed_outputs(outputs, "rust", "crack", 0.5, 20)


class _TensorIOMode:
    INPUT = "input"
    OUTPUT = "output"


class _TensorFormat:
    LINEAR = "linear"


class _TensorLocation:
    DEVICE = "device"


class _TensorRT:
    TensorIOMode = _TensorIOMode
    TensorFormat = _TensorFormat
    TensorLocation = _TensorLocation

    @staticmethod
    def nptype(dtype):
        return dtype


class _Engine:
    def __init__(self) -> None:
        self.names = [INPUT_NAME, *OUTPUT_CONTRACTS]
        self.modes = {INPUT_NAME: _TensorIOMode.INPUT}
        self.modes.update({name: _TensorIOMode.OUTPUT for name in OUTPUT_CONTRACTS})
        self.shapes = {INPUT_NAME: INPUT_SHAPE}
        self.shapes.update({name: value[0] for name, value in OUTPUT_CONTRACTS.items()})
        self.dtypes = {INPUT_NAME: np.float32}
        self.dtypes.update({name: value[1] for name, value in OUTPUT_CONTRACTS.items()})
        self.formats = {name: _TensorFormat.LINEAR for name in self.names}
        self.locations = {name: _TensorLocation.DEVICE for name in self.names}

    @property
    def num_io_tensors(self) -> int:
        return len(self.names)

    def get_tensor_name(self, index):
        return self.names[index]

    def get_tensor_mode(self, name):
        return self.modes[name]

    def get_tensor_shape(self, name):
        return self.shapes[name]

    def get_tensor_dtype(self, name):
        return self.dtypes[name]

    def get_tensor_format(self, name):
        return self.formats[name]

    def get_tensor_location(self, name):
        return self.locations[name]


class EngineContractTests(unittest.TestCase):
    @staticmethod
    def _detector() -> OptimizedMultitaskDetector:
        detector = OptimizedMultitaskDetector.__new__(OptimizedMultitaskDetector)
        detector._trt = _TensorRT()
        detector._engine = _Engine()
        return detector

    def test_accepts_only_small_postprocessed_outputs(self) -> None:
        self._detector()._validate_engine_contract()

    def test_rejects_raw_five_channel_output(self) -> None:
        detector = self._detector()
        detector._engine.names.append(RAW_LOGITS_NAME)
        with self.assertRaisesRegex(RuntimeError, "Raw 5-channel logits"):
            detector._validate_engine_contract()

    def test_rejects_wrong_map_dtype(self) -> None:
        detector = self._detector()
        detector._engine.dtypes[RUST_MAP_NAME] = np.float32
        with self.assertRaisesRegex(RuntimeError, "uint8"):
            detector._validate_engine_contract()

    def test_rejects_wrong_input_shape_or_mode(self) -> None:
        detector = self._detector()
        detector._engine.shapes[INPUT_NAME] = (1, 3, 128, 1280)
        with self.assertRaisesRegex(RuntimeError, "input shape"):
            detector._validate_engine_contract()

        detector = self._detector()
        detector._engine.modes[INPUT_NAME] = _TensorIOMode.OUTPUT
        with self.assertRaisesRegex(RuntimeError, "not an input"):
            detector._validate_engine_contract()

    def test_rejects_wrong_output_shape_mode_format_or_location(self) -> None:
        mutations = (
            ("got", lambda engine: engine.shapes.__setitem__(RUST_MAP_NAME, (1, 1))),
            ("not an output", lambda engine: engine.modes.__setitem__(RUST_MAP_NAME, _TensorIOMode.INPUT)),
            ("LINEAR", lambda engine: engine.formats.__setitem__(RUST_MAP_NAME, "vectorized")),
            ("GPU device", lambda engine: engine.locations.__setitem__(RUST_MAP_NAME, "host")),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                detector = self._detector()
                mutate(detector._engine)
                with self.assertRaisesRegex(RuntimeError, expected):
                    detector._validate_engine_contract()


if __name__ == "__main__":
    unittest.main()
