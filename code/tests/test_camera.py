"""Camera read completion timing contract without a physical device."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"


class CameraTimingTests(unittest.TestCase):
    def test_reader_preserves_ns_sample_taken_immediately_after_read(self):
        source = textwrap.dedent('''
            from unittest.mock import patch
            import numpy as np
            from node_runtime import camera

            class Capture:
                def __init__(self): self.calls = []
                def read(self):
                    self.calls.append('read')
                    if len(self.calls) > 1: return False, None
                    return True, np.zeros((2, 2, 3), np.uint8)
                def release(self): pass

            capture = Capture()
            def clock_ns():
                capture.calls.append('clock_ns')
                return 12345678901
            with patch.object(camera.time, 'monotonic_ns', side_effect=clock_ns):
                reader = camera.LatestFrameCamera(capture)
                frame = reader.read_latest(timeout=1)
                try: reader.close()
                except RuntimeError: pass
            assert capture.calls[:2] == ['read', 'clock_ns'], capture.calls
            assert frame.read_completed_monotonic_ns == 12345678901
            assert frame.read_completed_at > 0
        ''')
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), env.get("PYTHONPATH", "")))
        result = subprocess.run([sys.executable, "-B", "-c", source], env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
