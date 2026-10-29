"""Explicit BGR/mono image transport with stride validation."""
from __future__ import annotations

from array import array as byte_array

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from rail_robot_interfaces.msg import CameraFrame as FrameMessage

from ..data import CameraFrame
from ..settings import FRAME_HEIGHT, FRAME_WIDTH, REALTIME_INPUT_MAX_AGE_SECONDS
from .common import source_monotonic, stamp_from_seconds


def encode_image(array: np.ndarray, header=None) -> Image:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
        raise ValueError("Only uint8 BGR/mono images are supported.")
    if array.ndim == 2:
        channels, encoding = 1, "mono8"
    elif array.ndim == 3 and array.shape[2] == 3:
        channels, encoding = 3, "bgr8"
    else:
        raise ValueError("Expected HxW or HxWx3 image.")
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("Empty images cannot be published.")
    message = Image()
    message.header = header if header is not None else Header()
    message.height, message.width = array.shape[:2]
    message.encoding, message.is_bigendian = encoding, 0
    message.step = message.width * channels
    # ROS uint8[] setters accept array('B') directly. Passing bytes instead
    # validates millions of individual pixels in Python before copying them.
    message.data = byte_array('B', np.ascontiguousarray(array).tobytes())
    return message


def decode_image(message: Image) -> np.ndarray:
    if message.encoding not in ("bgr8", "mono8"):
        raise ValueError("Unsupported image encoding.")
    channels = 3 if message.encoding == "bgr8" else 1
    if message.height <= 0 or message.width <= 0:
        raise ValueError("Empty image.")
    row_bytes = message.width * channels
    if message.step < row_bytes or len(message.data) != message.step * message.height:
        raise ValueError("Image stride or data length is invalid.")
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(
        message.height, message.step)
    pixels = rows[:, :row_bytes].copy()
    return pixels.reshape((message.height, message.width, channels)
                          if channels == 3 else (message.height, message.width))


def frame_to_message(frame: CameraFrame, camera_role: str, ros_now_seconds: float,
                     monotonic_now: float) -> FrameMessage:
    if camera_role not in ("side", "top"):
        raise ValueError("Unknown camera role.")
    age = monotonic_now - frame.read_completed_at
    if not np.isfinite(age) or age < 0:
        raise ValueError("Invalid local frame age.")
    message = FrameMessage()
    message.header = Header()
    message.header.stamp = stamp_from_seconds(ros_now_seconds - age)
    message.header.frame_id = camera_role
    message.camera_role, message.sequence = camera_role, frame.sequence
    message.capture_complete_monotonic_ns = frame.read_completed_monotonic_ns
    message.image = encode_image(frame.frame, message.header)
    return message


def message_to_frame(message: FrameMessage, ros_now_seconds: float,
                     monotonic_now: float,
                     max_age: float = REALTIME_INPUT_MAX_AGE_SECONDS) -> CameraFrame:
    if message.camera_role not in ("side", "top"):
        raise ValueError("Unknown camera role.")
    observed = source_monotonic(message.header.stamp, ros_now_seconds,
                                monotonic_now, max_age)
    frame = decode_image(message.image)
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ValueError("Camera frame must remain native 1280x720 BGR.")
    return CameraFrame(
        frame, message.sequence, observed, message.capture_complete_monotonic_ns
    )
