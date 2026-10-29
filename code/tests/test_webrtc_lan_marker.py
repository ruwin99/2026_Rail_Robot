"""Local timestamp marker checks; no SSH or native WebRTC is started."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from webrtc_lan_check import read_latency_marker, write_latency_marker


class LatencyMarkerTests(unittest.TestCase):
    def test_new_and_legacy_layout_survive_jpeg_round_trip(self):
        stamp = 1788732345123456000
        for height in (360, 720):
            with self.subTest(height=height):
                frame = np.full((height, 1280, 3), 64, np.uint8)
                write_latency_marker(frame, stamp)
                self.assertTrue(np.all(frame[:-16] == 64))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.assertTrue(ok)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                self.assertAlmostEqual(read_latency_marker(decoded), stamp / 1e9, places=6)

    def test_missing_corrupt_or_small_marker_is_not_a_latency_sample(self):
        frame = np.zeros((360, 1280), np.uint8)
        self.assertIsNone(read_latency_marker(frame))
        write_latency_marker(frame, 1788732345123456000)
        frame[-16:, 24*8:25*8] = 255 - frame[-16:, 24*8:25*8]
        self.assertIsNone(read_latency_marker(frame))
        self.assertIsNone(read_latency_marker(np.zeros((8, 1280), np.uint8)))


if __name__ == "__main__":
    unittest.main()
