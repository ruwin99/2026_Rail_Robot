from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "jetson_code" / "training_capture.py"


class _Frame:
    shape = (720, 1280, 3)


class _Camera:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.frame = _Frame()
        self.released = False
        self.set_calls: list[tuple[int, object]] = []

    def set(self, property_id: int, value: object) -> bool:
        self.set_calls.append((property_id, value))
        return True

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, _Frame]:
        return True, self.frame

    def release(self) -> None:
        self.released = True


class _FakeCv2(SimpleNamespace):
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_AUTOFOCUS = 39
    WINDOW_NORMAL = 0

    def __init__(self, camera: _Camera, keys: list[int]) -> None:
        super().__init__()
        self.camera = camera
        self.keys = iter(keys)
        self.imwrite_calls: list[tuple[Path, object]] = []
        self.destroyed = False

    def VideoCapture(self, index: int, backend: int) -> _Camera:
        self.open_call = (index, backend)
        return self.camera

    @staticmethod
    def VideoWriter_fourcc(*_letters: str) -> int:
        return 1234

    def namedWindow(self, *_args: object) -> None:
        pass

    def imshow(self, *_args: object) -> None:
        pass

    def waitKey(self, _delay: int) -> int:
        return next(self.keys)

    def imwrite(self, path: str, frame: object) -> bool:
        self.imwrite_calls.append((Path(path), frame))
        return True

    def destroyAllWindows(self) -> None:
        self.destroyed = True


def _load_module(fake_cv2: _FakeCv2):
    spec = importlib.util.spec_from_file_location("training_capture_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load training_capture.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        spec.loader.exec_module(module)
    return module


class TrainingCaptureTests(unittest.TestCase):
    def test_s_saves_one_raw_frame_and_q_releases_camera(self) -> None:
        camera = _Camera()
        fake_cv2 = _FakeCv2(camera, [ord("s"), ord("q")])
        module = _load_module(fake_cv2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "for model"
            module.OUTPUT_DIRECTORY = output_directory
            output = io.StringIO()
            with redirect_stdout(output):
                result = module.run_training_capture()

        self.assertEqual(result, 0)
        self.assertEqual(len(fake_cv2.imwrite_calls), 1)
        saved_path, saved_frame = fake_cv2.imwrite_calls[0]
        self.assertEqual(saved_path.parent, output_directory)
        self.assertEqual(saved_path.suffix, ".jpg")
        self.assertIs(saved_frame, camera.frame)
        self.assertIn(str(saved_path.resolve()), output.getvalue())
        self.assertTrue(camera.released)
        self.assertTrue(fake_cv2.destroyed)

    def test_closed_camera_returns_error_and_releases_resource(self) -> None:
        camera = _Camera(opened=False)
        fake_cv2 = _FakeCv2(camera, [])
        module = _load_module(fake_cv2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            module.OUTPUT_DIRECTORY = Path(temporary_directory) / "for model"
            error = io.StringIO()
            with redirect_stderr(error):
                result = module.run_training_capture()

        self.assertEqual(result, 1)
        self.assertIn("Could not open camera", error.getvalue())
        self.assertEqual(fake_cv2.imwrite_calls, [])
        self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
