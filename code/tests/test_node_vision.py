from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from node_runtime import camera, inspection, vision  # noqa: E402
from node_runtime.data import CameraFrame  # noqa: E402
from inspection_report import INITIAL_PHASE  # noqa: E402


class NodeVisionBoundaryTests(unittest.TestCase):
    def test_cores_import_without_starting_run_or_loading_uart_or_ros(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(JETSON_CODE), environment.get("PYTHONPATH", ""))
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; from node_runtime import camera, vision, display, inspection; "
                "assert 'run' not in sys.modules; "
                "assert 'serial' not in sys.modules; "
                "assert 'rclpy' not in sys.modules",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_delayed_inference_retains_capture_time_and_cannot_be_ready(self):
        frame = CameraFrame(np.zeros((720, 1280, 3), dtype=np.uint8), 42, 10.0)
        rust_roi, crack_roi = vision.extract_realtime_control_rois(frame.frame)
        rust_result = object()
        crack_result = object()
        rust_detector = Mock(detect=Mock(return_value=rust_result))
        crack_detector = Mock(detect=Mock(return_value=crack_result))
        inference = vision.AlternatingRealtimeInference(
            rust_detector,
            crack_detector,
            multitask_enabled=False,
            hybrid_enabled=False,
            max_result_age_seconds=0.8,
        )

        with patch.object(vision.time, "monotonic", return_value=10.1):
            first = inference.process(frame, rust_roi, crack_roi)
        with patch.object(vision.time, "monotonic", return_value=10.9):
            delayed = inference.process(frame, rust_roi, crack_roi)

        self.assertFalse(first.ready)
        self.assertFalse(delayed.ready)
        self.assertEqual(delayed.frame_read_completed_at, 10.0)
        self.assertIs(delayed.rust_result, rust_result)
        self.assertIs(delayed.crack_result, crack_result)
        self.assertEqual(rust_detector.detect.call_count, 1)
        self.assertEqual(crack_detector.detect.call_count, 1)
        self.assertEqual(rust_detector.detect.call_args.args[0].shape, (240, 1280, 3))
        # 실시간 균열 ROI는 원본 하단 180행이며, 128행 엔진 크기로의 축소는 detector가 맡는다.
        self.assertEqual(crack_detector.detect.call_args.args[0].shape, (180, 1280, 3))
        self.assertFalse(camera.realtime_camera_frame_is_fresh(frame, now=10.9))

    def test_dual_capture_is_persisted_before_atomic_worker_acceptance(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        accepted = []

        def accept(tasks):
            self.assertEqual(len(tasks), 2)
            self.assertTrue(all(task.raw_capture_path.is_file() for task in tasks))
            self.assertEqual({task.camera_role for task in tasks}, {"side", "top"})
            accepted.extend(tasks)

        with tempfile.TemporaryDirectory() as temporary:
            worker = Mock(submit_batch=Mock(side_effect=accept))
            tasks = inspection.queue_dual_capture_for_analysis(
                frame,
                frame,
                INITIAL_PHASE,
                1,
                "CAMERA_CAPTURE",
                worker,
                captured_at=datetime(2026, 9, 5, 12, 0),
                side_frame_read_completed_at=100.0,
                top_frame_read_completed_at=100.02,
                output_directory=Path(temporary),
            )
            self.assertEqual(tuple(accepted), tasks)
            self.assertEqual(tasks[0].frame_read_completed_at, 100.0)
            self.assertEqual(tasks[1].frame_read_completed_at, 100.02)
            self.assertEqual(tasks[0].captured_at, tasks[1].captured_at)
            for task in tasks:
                self.assertEqual(cv2.imread(str(task.raw_capture_path)).shape, frame.shape)
            worker.submit_batch.assert_called_once()
            worker.submit.assert_not_called()

    def test_rejected_pair_keeps_both_originals_recoverable(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        worker = Mock(submit_batch=Mock(side_effect=RuntimeError("queue is full")))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no acknowledgement/count increment"):
                inspection.queue_dual_capture_for_analysis(
                    frame,
                    frame,
                    INITIAL_PHASE,
                    1,
                    "CAMERA_CAPTURE",
                    worker,
                    captured_at=datetime(2026, 9, 5, 12, 0),
                    side_frame_read_completed_at=100.0,
                    top_frame_read_completed_at=100.02,
                    output_directory=Path(temporary),
                )
            originals = list(Path(temporary).rglob("*.jpg"))
            self.assertEqual(len(originals), 2)
            self.assertEqual({path.parent.name for path in originals}, {"side", "top"})
            worker.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
