from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))

from obstacle_detector import ObstacleDetection, ObstacleDetectionResult
from node_runtime.data import CameraFrame
from node_runtime.display import annotate_top_realtime_results
from node_runtime.vision import (
    AlternatingTopRealtimeInference,
    clip_realtime_obstacles_to_roi,
    extract_realtime_control_rois,
)


def result_with_boxes(*boxes):
    return ObstacleDetectionResult(
        detections=tuple(ObstacleDetection(box, 0.8, 0) for box in boxes),
        method="test",
        confidence_threshold=0.424,
        control_roi_detected=any(box[1] < 720 and box[3] > 480 for box in boxes),
    )


class RealtimeObstacleRoiTests(unittest.TestCase):
    def test_clips_intersections_and_drops_outside_or_boundary_only_boxes(self):
        source = result_with_boxes(
            (10.0, 500.0, 30.0, 520.0),
            (-5.0, -8.0, 1300.0, 720.0),
            (10.0, 470.0, 30.0, 480.5),
            (10.0, 400.0, 30.0, 480.0),
            (10.0, 300.0, 30.0, 400.0),
            (1280.0, 500.0, 1300.0, 520.0),
            (-20.0, 500.0, 0.0, 520.0),
            (10.0, 720.0, 30.0, 730.0),
        )
        clipped = clip_realtime_obstacles_to_roi(source)
        self.assertEqual(clipped.boxes, (
            (10.0, 500.0, 30.0, 520.0),
            (0.0, 480.0, 1280.0, 720.0),
            (10.0, 480.0, 30.0, 480.5),
        ))
        self.assertTrue(clipped.control_roi_detected)
        self.assertEqual(clipped.scores, (0.8,) * 3)
        self.assertEqual(clipped.class_ids, (0,) * 3)
        self.assertEqual(clipped.method, source.method)
        self.assertEqual(clipped.confidence_threshold, source.confidence_threshold)
        self.assertEqual(len(source.detections), 8)
        self.assertEqual(source.boxes[1], (-5.0, -8.0, 1300.0, 720.0))
        self.assertEqual(clip_realtime_obstacles_to_roi(clipped), clipped)

    def test_outside_only_and_empty_results_are_not_detected(self):
        for source in (result_with_boxes(), result_with_boxes((10, 240, 30, 400))):
            with self.subTest(source=source):
                clipped = clip_realtime_obstacles_to_roi(source)
                self.assertFalse(clipped.detected)
                self.assertFalse(clipped.control_roi_detected)

    def test_realtime_scheduler_caches_and_displays_only_roi_boxes(self):
        source = result_with_boxes((10, 479, 30, 500), (40, 300, 60, 400))
        obstacle = Mock()
        obstacle.detect.return_value = source
        scheduler = AlternatingTopRealtimeInference(obstacle, Mock())
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = CameraFrame(frame, 1, time.monotonic())
        scheduler.process(camera, frame, frame[540:720])  # Crack phase first.
        outcome = scheduler.process(camera, frame, frame[540:720])
        self.assertEqual(outcome.display_obstacle_result.boxes, ((10, 480, 30, 500),))
        self.assertEqual(outcome.obstacle_result, outcome.display_obstacle_result)
        obstacle.detect.assert_called_once_with(frame)
        self.assertEqual(len(source.detections), 2)  # Capture/raw result untouched.

    def test_preview_stroke_never_leaks_above_roi_and_input_is_unchanged(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        source = result_with_boxes((-5, -8, 1300, 720), (10, 300, 30, 400))
        # ROS preview objects carry detections only, not detector metadata.
        shown = annotate_top_realtime_results(
            frame, SimpleNamespace(detections=source.detections), None
        )
        self.assertTrue(np.any(shown[480:]))
        self.assertTrue(np.any(shown[719]))  # Clipped bottom border is visible.
        self.assertFalse(np.any(shown[:480]))
        self.assertFalse(np.any(frame))

    def test_bottom_quarter_is_shared_and_extra_sixty_rows_are_primary_only(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[480:540] = 70
        frame[540:] = 140
        primary, crack = extract_realtime_control_rois(frame)
        self.assertEqual(primary.shape, (240, 1280, 3))
        self.assertEqual(crack.shape, (180, 1280, 3))
        self.assertTrue(np.all(primary[:60] == 70))
        self.assertTrue(np.all(crack == 140))
        np.testing.assert_array_equal(primary[60:], crack)


if __name__ == "__main__":
    unittest.main()
