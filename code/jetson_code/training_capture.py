from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30
CAMERA_INDEX = 0
WINDOW_NAME = "Training Image Capture"
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "for model"


def _capture_path(output_directory: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_directory / f"training_{timestamp}.jpg"


def run_training_capture() -> int:
    output_directory = OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)

    camera = None
    try:
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        if not camera.isOpened():
            print(
                f"Could not open camera index {CAMERA_INDEX}. Check the USB camera connection.",
                file=sys.stderr,
            )
            return 1

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print(f"Training images: {output_directory}")
        print("Press S to save the current raw frame. Press Q or Esc to quit.")
        resolution_reported = False

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a frame from the camera.", file=sys.stderr)
                return 1

            if not resolution_reported:
                height, width = frame.shape[:2]
                if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
                    print(
                        "Camera returned "
                        f"{width}x{height}; saving that native frame without resizing.",
                        file=sys.stderr,
                    )
                resolution_reported = True

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                capture_path = _capture_path(output_directory)
                if not cv2.imwrite(str(capture_path), frame):
                    print(f"Could not save image: {capture_path}", file=sys.stderr)
                    return 1
                print(f"Saved: {capture_path.resolve()}")
            elif key in (ord("q"), ord("Q"), 27):
                return 0
    except Exception as exc:
        print(f"Training capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(run_training_capture())
