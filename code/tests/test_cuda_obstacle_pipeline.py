from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from cuda_obstacle_pipeline import (  # noqa: E402
    COMPACT_NBYTES,
    COMPACT_SHAPE,
    MODEL_INPUT_NBYTES,
    OUTPUT_COLUMNS,
    OUTPUT_ROWS,
    RAW_INPUT_NBYTES,
    RAW_INPUT_SHAPE,
    SUMMARY_NBYTES,
    SUMMARY_SHAPE,
    CudaObstaclePipeline,
    _kernel_source,
)


class ObstacleKernelContractTests(unittest.TestCase):
    def test_header_free_kernels_preserve_locked_semantics(self) -> None:
        source = _kernel_source(0.424)

        self.assertNotIn("#include", source)
        self.assertIn("obstacle_bgr_to_rgb_nchw", source)
        self.assertIn("obstacle_validate_compact", source)
        self.assertIn("bgr[source + 2]", source)
        self.assertIn("bgr[source + 1]", source)
        self.assertIn("bgr[source]", source)
        self.assertIn("y < 8 || y >= 728", source)
        self.assertIn("114.0f / 255.0f", source)
        self.assertIn("0x3ed91687u", source)
        self.assertIn("score >= threshold", source)
        self.assertIn("score >= 0.0f && score <= 1.0f", source)
        self.assertIn("raw_x2 >= raw_x1 && raw_y2 >= raw_y1", source)
        self.assertIn("class_value == 0.0f", source)
        self.assertIn("raw_y1 - 8.0f", source)
        self.assertIn("y1 < 720.0f && y2 > 480.0f", source)
        self.assertNotIn("nms", source.lower())

    def test_transfer_sizes_are_fixed_by_the_onnx_contract(self) -> None:
        self.assertEqual(MODEL_INPUT_NBYTES, 11_304_960)
        self.assertEqual(RAW_INPUT_NBYTES, 2_764_800)
        self.assertEqual(SUMMARY_NBYTES, 12)
        self.assertEqual(COMPACT_NBYTES, 7_200)


class _FakeCudaRuntime:
    class cudaError_t:
        cudaSuccess = 0

    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = "h2d"
        cudaMemcpyDeviceToHost = "d2h"

    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def cudaMemcpyAsync(
        self,
        destination,
        source,
        nbytes,
        kind,
        stream,
    ):
        self.events.append(("memcpy", int(nbytes), kind, stream))
        return (0,)

    def cudaStreamSynchronize(self, stream):
        self.events.append(("sync", stream))
        return (0,)

    def cudaEventRecord(self, event, stream):
        self.events.append(("event", event, stream))
        return (0,)

    def cudaFree(self, allocation):
        self.events.append(("free", allocation))
        return (0,)

    def cudaFreeHost(self, allocation):
        self.events.append(("free_host", allocation))
        return (0,)


class _FakeCudaDriver:
    class CUresult:
        CUDA_SUCCESS = 0

    @staticmethod
    def CUdeviceptr(pointer):
        return int(pointer)

    @staticmethod
    def CUstream(stream):
        return int(stream)

    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def cuLaunchKernel(
        self,
        function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_memory,
        stream,
        arguments,
        extra,
    ):
        self.events.append(
            ("launch", function, int(grid_x), int(block_x), int(stream))
        )
        return (0,)

    def cuModuleUnload(self, module):
        self.events.append(("unload", module))
        return (0,)


def _mock_pipeline() -> tuple[CudaObstaclePipeline, list[tuple]]:
    events: list[tuple] = []
    pipeline = CudaObstaclePipeline.__new__(CudaObstaclePipeline)
    pipeline._cudart = _FakeCudaRuntime(events)
    pipeline._cuda = _FakeCudaDriver(events)
    pipeline._nvrtc = None
    pipeline._module = "module"
    pipeline._preprocess_function = "preprocess"
    pipeline._postprocess_function = "postprocess"
    pipeline._host_raw_pointer = "host_raw"
    pipeline._host_summary_pointer = "host_summary"
    pipeline._host_compact_pointer = "host_compact"
    pipeline._host_raw = np.empty(RAW_INPUT_SHAPE, dtype=np.uint8)
    pipeline._host_summary = np.zeros(SUMMARY_SHAPE, dtype=np.uint32)
    pipeline._host_compact = np.zeros(COMPACT_SHAPE, dtype=np.float32)
    pipeline._device_raw = 101
    pipeline._device_summary = 102
    pipeline._device_compact = 103
    return pipeline, events


class ObstaclePipelineTransferTests(unittest.TestCase):
    def test_upload_copies_only_raw_uint8_then_launches_preprocess(self) -> None:
        pipeline, events = _mock_pipeline()
        frame = np.zeros(RAW_INPUT_SHAPE, dtype=np.uint8)
        frame[0, 0] = (1, 2, 3)

        pipeline.upload_and_preprocess(frame, tensor_pointer=201, stream=301)

        self.assertEqual(pipeline._host_raw[0, 0].tolist(), [1, 2, 3])
        self.assertEqual(
            events,
            [
                ("memcpy", RAW_INPUT_NBYTES, "h2d", 301),
                ("launch", "preprocess", 3680, 256, 301),
            ],
        )
        self.assertNotIn(MODEL_INPUT_NBYTES, [event[1] for event in events])

    def test_debug_upload_marks_h2d_and_gpu_preprocess_boundaries(self) -> None:
        pipeline, events = _mock_pipeline()
        frame = np.zeros(RAW_INPUT_SHAPE, dtype=np.uint8)

        cpu_ms = pipeline.upload_and_preprocess(
            frame,
            tensor_pointer=201,
            stream=301,
            timing_events=("h2d-start", "h2d-end", "preprocess-end"),
        )

        self.assertGreaterEqual(cpu_ms, 0.0)
        self.assertEqual(
            events,
            [
                ("event", "h2d-start", 301),
                ("memcpy", RAW_INPUT_NBYTES, "h2d", 301),
                ("event", "h2d-end", 301),
                ("launch", "preprocess", 3680, 256, 301),
                ("event", "preprocess-end", 301),
            ],
        )

    def test_postprocess_copies_summary_then_only_live_compact_rows(self) -> None:
        pipeline, events = _mock_pipeline()
        pipeline._host_summary[:] = (2, 1, 0)
        pipeline._host_compact[0] = (10, 500, 30, 520, 0.8, 0)
        pipeline._host_compact[1] = (50, 600, 70, 620, 0.7, 0)

        result = pipeline.postprocess(output_pointer=202, stream=302)

        self.assertEqual(result.records.shape, (2, OUTPUT_COLUMNS))
        self.assertTrue(result.control_roi_detected)
        self.assertEqual(
            events,
            [
                ("launch", "postprocess", 1, 1, 302),
                ("memcpy", SUMMARY_NBYTES, "d2h", 302),
                ("sync", 302),
                ("memcpy", 2 * OUTPUT_COLUMNS * 4, "d2h", 302),
                ("sync", 302),
            ],
        )
        self.assertNotIn(COMPACT_NBYTES, [event[1] for event in events])

    def test_debug_postprocess_marks_kernel_and_complete_d2h_boundaries(self) -> None:
        pipeline, events = _mock_pipeline()
        pipeline._host_summary[:] = (2, 1, 0)

        pipeline.postprocess(
            output_pointer=202,
            stream=302,
            timing_events=("postprocess-end", "d2h-end"),
        )

        self.assertEqual(
            events,
            [
                ("launch", "postprocess", 1, 1, 302),
                ("event", "postprocess-end", 302),
                ("memcpy", SUMMARY_NBYTES, "d2h", 302),
                ("event", "d2h-end", 302),
                ("sync", 302),
                ("memcpy", 2 * OUTPUT_COLUMNS * 4, "d2h", 302),
                ("event", "d2h-end", 302),
                ("sync", 302),
            ],
        )

    def test_postprocess_invalid_flag_stops_before_compact_copy(self) -> None:
        pipeline, events = _mock_pipeline()
        pipeline._host_summary[:] = (1, 0, 1)

        with self.assertRaisesRegex(ValueError, "failed GPU validation"):
            pipeline.postprocess(output_pointer=202, stream=302)

        self.assertEqual(
            events,
            [
                ("launch", "postprocess", 1, 1, 302),
                ("memcpy", SUMMARY_NBYTES, "d2h", 302),
                ("sync", 302),
            ],
        )

    def test_zero_count_does_not_copy_compact_buffer(self) -> None:
        pipeline, events = _mock_pipeline()
        pipeline._host_summary[:] = (0, 0, 0)

        result = pipeline.postprocess(output_pointer=202, stream=302)

        self.assertEqual(result.records.shape, (0, OUTPUT_COLUMNS))
        self.assertFalse(result.control_roi_detected)
        self.assertEqual(
            [event[0] for event in events],
            ["launch", "memcpy", "sync"],
        )

    def test_close_handles_partial_initialization_and_is_idempotent(self) -> None:
        pipeline, events = _mock_pipeline()
        pipeline._device_summary = None
        pipeline._host_compact_pointer = None

        pipeline.close()
        first_events = list(events)
        pipeline.close()

        self.assertEqual(events, first_events)
        self.assertEqual(
            first_events,
            [
                ("free", 101),
                ("free", 103),
                ("free_host", "host_raw"),
                ("free_host", "host_summary"),
                ("unload", "module"),
            ],
        )
        self.assertIsNone(pipeline._preprocess_function)
        self.assertIsNone(pipeline._postprocess_function)
        self.assertIsNone(pipeline._host_raw)


if __name__ == "__main__":
    unittest.main()
