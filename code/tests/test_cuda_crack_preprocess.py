"""실시간 균열 GPU 전처리: 가중치 표와 커널 계산이 CPU 경로(OpenCV INTER_AREA + 정규화)와 같은지,
검출기가 GPU 경로를 올바르게 고르는지 확인한다. 실제 GPU 실행은 Jetson에서 따로 비교했다."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from cuda_crack_preprocess import (  # noqa: E402
    RAW_SHAPE,
    _kernel_source,
    area_resize_table,
)
from hrsegnet_crack_detector import (  # noqa: E402
    INPUT_SHAPE,
    HrSegNetCrackDetector,
    preprocess_into,
    resize_realtime_roi,
)


def _emulate_kernel(frame: np.ndarray) -> np.ndarray:
    """커널과 같은 순서(float32 곱 → 더하기, 가까운 짝수 반올림, x/127.5-1)를 numpy로 따라 한다."""
    starts, counts, rows, alphas = area_resize_table()
    source = frame.astype(np.float32)
    resized = np.empty((128, 1280, 3), dtype=np.uint8)
    for y, (start, count) in enumerate(zip(starts, counts)):
        total = np.zeros((1280, 3), dtype=np.float32)
        for k in range(start, start + count):
            total = (total + np.float32(alphas[k]) * source[rows[k]]).astype(np.float32)
        resized[y] = np.clip(np.rint(total), 0, 255).astype(np.uint8)
    rgb = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    return (rgb / np.float32(127.5) - np.float32(1.0))[None]


class CrackAreaTableTests(unittest.TestCase):
    def test_table_matches_opencv_area_weights(self):
        starts, counts, rows, alphas = area_resize_table()
        self.assertEqual(len(starts), 128)
        self.assertEqual(set(counts), {2, 3})
        for start, count in zip(starts, counts):
            self.assertAlmostEqual(sum(alphas[start:start + count]), 1.0, places=5)
            self.assertEqual(rows[start:start + count], sorted(rows[start:start + count]))
        self.assertEqual((rows[0], rows[-1]), (0, 179))

    def test_kernel_math_is_bit_identical_to_cpu_path(self):
        rng = np.random.default_rng(21)
        frames = [rng.integers(0, 256, RAW_SHAPE, dtype=np.uint8) for _ in range(4)]
        frames += [np.zeros(RAW_SHAPE, np.uint8), np.full(RAW_SHAPE, 255, np.uint8)]
        expected = np.empty(INPUT_SHAPE, dtype=np.float32)
        for frame in frames:
            preprocess_into(resize_realtime_roi(frame), expected)
            np.testing.assert_array_equal(_emulate_kernel(frame), expected)

    def test_kernel_source_keeps_cpu_rounding(self):
        source = _kernel_source()
        self.assertIn("__fadd_rn(sum0, __fmul_rn(beta", source)
        self.assertIn("__float2int_rn(accumulated)", source)
        self.assertIn("__fdiv_rn((float)rounded, 127.5f)", source)
        # RGB 평면 순서: R=원본 2번 채널.
        self.assertIn("input[index] = normalized(sum2);", source)
        self.assertIn("__fmaf_rn", _kernel_source(fused=True))


class DetectorGpuPreprocessRoutingTests(unittest.TestCase):
    def _detector(self, candidate):
        detector = HrSegNetCrackDetector.__new__(HrSegNetCrackDetector)
        detector.role = "realtime"
        detector.input_shape = INPUT_SHAPE
        detector.method = "test"
        detector.probability_threshold = 0.55
        detector.min_component_pixels = 20
        detector.stage_timing = False
        detector._context = Mock()
        detector._context.execute_async_v3.return_value = True
        detector._stream = 17
        detector._device_input = 100
        detector._device_output = 200
        detector._host_input = np.full(INPUT_SHAPE, 7.0, dtype=np.float32)
        detector._trt_timing_events = ()
        detector._cudart = Mock()
        detector._cuda_call = Mock()
        detector._gpu_postprocessor = Mock()
        detector._gpu_postprocessor.process.return_value = SimpleNamespace(
            candidate_map=candidate, max_margin=2.0
        )
        detector._gpu_preprocessor = Mock()
        return detector

    def test_180_row_roi_goes_to_gpu_and_skips_cpu_input_copy(self):
        candidate = np.zeros((128, 1280), dtype=np.uint8)
        candidate[:64] = 1
        detector = self._detector(candidate)
        frame = np.zeros(RAW_SHAPE, dtype=np.uint8)
        result = detector.detect(frame)
        detector._gpu_preprocessor.stage.assert_called_once_with(frame)
        detector._gpu_preprocessor.enqueue.assert_called_once_with(100, 17)
        detector._cudart.cudaMemcpyAsync.assert_not_called()
        self.assertTrue(np.all(detector._host_input == 7.0))
        self.assertEqual(result.mask.shape, (180, 1280))
        self.assertEqual(result.crack_pixels, 90 * 1280)

    def test_native_128_row_frame_keeps_cpu_path(self):
        detector = self._detector(np.zeros((128, 1280), dtype=np.uint8))
        detector.detect(np.zeros((128, 1280, 3), dtype=np.uint8))
        detector._gpu_preprocessor.stage.assert_not_called()
        detector._gpu_preprocessor.enqueue.assert_not_called()
        detector._cudart.cudaMemcpyAsync.assert_called_once()

    def test_invalid_roi_is_rejected_before_gpu_work(self):
        detector = self._detector(np.zeros((128, 1280), dtype=np.uint8))
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((179, 1280, 3), dtype=np.uint8))
        detector._gpu_preprocessor.enqueue.assert_not_called()
        detector._context.execute_async_v3.assert_not_called()


class LinearRestoreTests(unittest.TestCase):
    def test_restore_rows_match_opencv_linear_resize(self):
        from cuda_logit_margin import linear_restore_rows

        rows = linear_restore_rows(128, 180)
        self.assertEqual(len(rows), 180)
        rng = np.random.default_rng(4)
        margin = rng.normal(0, 2, (128, 1280)).astype(np.float32)
        emulated = np.stack([
            (margin[upper] * np.float32(upper_w)).astype(np.float32)
            + (margin[lower] * np.float32(lower_w)).astype(np.float32)
            for upper, lower, upper_w, lower_w in rows
        ])
        expected = cv2.resize(margin, (1280, 180), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(emulated, expected, atol=1e-4)
        # 경계 행은 가장자리 행으로 고정된다.
        self.assertEqual(rows[0][:2], (0, 0))
        self.assertEqual(rows[-1][:2], (127, 127))
        for _, _, upper_w, lower_w in rows:
            self.assertAlmostEqual(upper_w + lower_w, 1.0, places=6)

    def test_restore_kernel_thresholds_interpolated_margin(self):
        from cuda_logit_margin import _restore_kernel_source

        source = _restore_kernel_source(128, 1280, 180, "0.200670689f")
        self.assertIn("margin >= 0.200670689f", source)
        self.assertIn("__fsub_rn(logits[163840 + upper], logits[upper])", source)
        self.assertIn("index >= 230400", source)

    def test_restore_height_must_exceed_logit_height(self):
        from cuda_logit_margin import CudaLogitMarginPostprocessor

        for value in (128, 100, True, 180.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CudaLogitMarginPostprocessor(object(), (1, 2, 128, 1280), 0.2, restore_height=value)

    def test_detector_uses_gpu_restored_map_without_nearest_resize(self):
        restored = np.zeros((180, 1280), dtype=np.uint8)
        restored[:3, 10:20] = 1  # 30 ROI pixels survive the 20-pixel filter.
        detector = DetectorGpuPreprocessRoutingTests._detector(None, restored)
        detector.linear_restore = True
        result = detector.detect(np.zeros(RAW_SHAPE, dtype=np.uint8))
        detector._gpu_postprocessor.process.assert_called_once_with(200, 17, restore=True)
        self.assertEqual(result.mask.shape, (180, 1280))
        self.assertEqual(result.crack_pixels, 30)
        self.assertEqual(result.boxes, [(10, 0, 10, 3)])

    def test_native_128_row_frame_never_requests_restore(self):
        detector = DetectorGpuPreprocessRoutingTests._detector(None, np.zeros((128, 1280), np.uint8))
        detector.linear_restore = True
        detector.detect(np.zeros((128, 1280, 3), dtype=np.uint8))
        detector._gpu_postprocessor.process.assert_called_once_with(200, 17)


if __name__ == "__main__":
    unittest.main()
