from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from cuda_rust_preprocess import (  # noqa: E402
    INPUT_SHAPE,
    RAW_NBYTES,
    RAW_SHAPE,
    CudaRustPreprocessor,
    _kernel_source,
)


class CudaRustPreprocessSourceTests(unittest.TestCase):
    def test_kernel_preserves_bgr_to_rgb_imagenet_nchw_contract(self) -> None:
        pixel_count = 240 * 1280
        source = _kernel_source(pixel_count)

        self.assertIn("bgr[source + 2]", source)
        self.assertIn("bgr[source + 1]", source)
        self.assertIn("bgr[source]", source)
        self.assertIn("input[pixel] = (red - 0.485f) / 0.229f", source)
        self.assertIn(
            f"input[{pixel_count} + pixel] = (green - 0.456f) / 0.224f",
            source,
        )
        self.assertIn(
            f"input[{2 * pixel_count} + pixel] = (blue - 0.406f) / 0.225f",
            source,
        )
        self.assertIn("/ 255.0f", source)

    def test_kernel_never_emits_invalid_integer_float_literal(self) -> None:
        source = _kernel_source(240 * 1280)

        self.assertIsNone(re.search(r"(?<![.0-9])[-+]?\d+f\b", source))

    def test_raw_upload_is_one_quarter_of_old_fp32_tensor(self) -> None:
        old_fp32_nbytes = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize

        self.assertEqual(RAW_SHAPE, (240, 1280, 3))
        self.assertEqual(RAW_NBYTES * 4, old_fp32_nbytes)

    def test_invalid_shape_fails_before_cuda_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "static input"):
            CudaRustPreprocessor(object(), (1, 3, 239, 1280))

    def test_stage_copies_noncontiguous_bgr_into_pinned_view(self) -> None:
        source = np.zeros((240, 1280, 6), dtype=np.uint8)[:, :, ::2]
        source[0, 0] = (5, 127, 250)
        preprocessor = CudaRustPreprocessor.__new__(CudaRustPreprocessor)
        preprocessor._host_raw = np.empty(RAW_SHAPE, dtype=np.uint8)

        preprocessor.stage(source)

        self.assertTrue(preprocessor._host_raw.flags.c_contiguous)
        np.testing.assert_array_equal(preprocessor._host_raw, source)

    def test_stage_rejects_wrong_dtype(self) -> None:
        preprocessor = CudaRustPreprocessor.__new__(CudaRustPreprocessor)
        preprocessor._host_raw = np.empty(RAW_SHAPE, dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "uint8 BGR"):
            preprocessor.stage(np.zeros(RAW_SHAPE, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
