from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from optimized_rust_detector import (  # noqa: E402
    INPUT_NAME,
    INPUT_SHAPE,
    LOGITS_FINITE_ABS_NAME,
    LOGITS_NOT_NAN_NAME,
    OUTPUT_CONTRACTS,
    RAW_LOGITS_NAME,
    RUST_MAP_NAME,
    OptimizedRustDetector,
    result_from_postprocessed_outputs,
)


def _valid_outputs() -> dict[str, np.ndarray]:
    class_map = np.zeros((1, 240, 1280), dtype=np.uint8)
    class_map[0, 10:20, 30:40] = 2
    return {
        RUST_MAP_NAME: class_map,
        LOGITS_NOT_NAN_NAME: np.array([1], dtype=np.uint8),
        LOGITS_FINITE_ABS_NAME: np.array([1], dtype=np.uint8),
    }


class PostprocessedOutputTests(unittest.TestCase):
    def test_builds_same_detection_semantics_without_raw_logits(self) -> None:
        result = result_from_postprocessed_outputs(_valid_outputs(), "optimized")

        self.assertEqual(result.class_map.dtype, np.uint8)
        self.assertEqual(result.class_map.shape, (240, 1280))
        self.assertAlmostEqual(result.rust_ratio, 100 / (240 * 1280))
        self.assertEqual(result.class_ratios["Poor"], result.rust_ratio)

    def test_either_finite_guard_fails_closed(self) -> None:
        cases = (
            (LOGITS_NOT_NAN_NAME, np.array([0], dtype=np.uint8)),
            (LOGITS_FINITE_ABS_NAME, np.array([0], dtype=np.uint8)),
        )
        for name, value in cases:
            with self.subTest(name=name):
                outputs = _valid_outputs()
                outputs[name] = value
                with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                    result_from_postprocessed_outputs(outputs, "optimized")

    def test_rejects_wrong_dtype_or_class_id(self) -> None:
        outputs = _valid_outputs()
        outputs[RUST_MAP_NAME] = outputs[RUST_MAP_NAME].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            result_from_postprocessed_outputs(outputs, "optimized")

        outputs = _valid_outputs()
        outputs[RUST_MAP_NAME][0, 0, 0] = 4
        with self.assertRaisesRegex(ValueError, "class ID"):
            result_from_postprocessed_outputs(outputs, "optimized")


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
        self.shapes.update({name: contract[0] for name, contract in OUTPUT_CONTRACTS.items()})
        self.dtypes = {INPUT_NAME: np.float32}
        self.dtypes.update({name: contract[1] for name, contract in OUTPUT_CONTRACTS.items()})
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
    def _detector() -> OptimizedRustDetector:
        detector = OptimizedRustDetector.__new__(OptimizedRustDetector)
        detector._trt = _TensorRT()
        detector._engine = _Engine()
        return detector

    def test_accepts_exact_small_device_linear_contract(self) -> None:
        self._detector()._validate_engine_contract()

    def test_rejects_raw_logits_output(self) -> None:
        detector = self._detector()
        detector._engine.names.append(RAW_LOGITS_NAME)
        with self.assertRaisesRegex(RuntimeError, "Raw four-channel logits"):
            detector._validate_engine_contract()

    def test_rejects_wrong_shape_dtype_mode_format_or_location(self) -> None:
        mutations = (
            ("got", lambda engine: engine.shapes.__setitem__(RUST_MAP_NAME, (1, 1))),
            ("uint8", lambda engine: engine.dtypes.__setitem__(RUST_MAP_NAME, np.float32)),
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
