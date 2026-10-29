"""Save camera originals on a ROS Trigger without loading inference or UART."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from uuid import uuid4

import cv2
from rclpy.node import Node
from rail_robot_interfaces.msg import CameraFrame as CameraFrameMessage
from std_srvs.srv import Trigger

from ..camera import realtime_camera_frame_is_fresh
from .common import qos_sensor, stamp_seconds
from .deployment import read_parameters
from .images import message_to_frame


class TrainingCaptureNode(Node):
    """Own raw JPEG persistence only; camera selection is a startup parameter."""

    def __init__(self, *, camera_role=None, output_directory=None,
                 node_name="rail_training_capture", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self._resource_lock = threading.RLock()
        self._closed = False
        self._stopping = False
        self._latest = None
        self._latest_key = None
        try:
            if camera_role is None or output_directory is None:
                values = read_parameters(self, ("training_camera_role", "training_output_directory"))
                camera_role = values["training_camera_role"] if camera_role is None else camera_role
                output_directory = values["training_output_directory"] if output_directory is None else output_directory
            if camera_role not in ("side", "top"):
                raise ValueError("training_camera_role must be side or top.")
            if not str(output_directory).strip():
                raise ValueError("training_output_directory must not be empty.")
            self._role = camera_role
            self._output = Path(output_directory).expanduser().resolve()
            self._camera_subscription = self.create_subscription(
                CameraFrameMessage, f"camera/{camera_role}", self._receive, qos_sensor)
            self._save_service = self.create_service(Trigger, "training/save", self._save)
        except Exception:
            super().destroy_node()
            raise

    def _receive(self, message):
        with self._resource_lock:
            if self._closed or self._stopping or message.camera_role != self._role:
                return
            try:
                if not message.stream_id:
                    raise ValueError("Camera stream ID is missing.")
                key = (message.stream_id, message.sequence, stamp_seconds(message.header.stamp))
                if self._latest_key is not None:
                    old_stream, old_sequence, old_stamp = self._latest_key
                    if key[2] < old_stamp or (key[0] == old_stream and key[1] <= old_sequence):
                        return
                frame = message_to_frame(
                    message, self.get_clock().now().nanoseconds / 1e9, time.monotonic())
                self._latest, self._latest_key = frame, key
            except (ValueError, TypeError) as exc:
                self.get_logger().warning(f"Training frame rejected: {exc}")

    def _save(self, request, response):
        del request
        with self._resource_lock:
            try:
                ready, reason = self.readiness()
                if not ready:
                    raise RuntimeError(reason)
                frame = self._latest.frame
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if not encoded_ok:
                    raise RuntimeError("Could not encode the original as JPEG.")
                captured_at = datetime.fromtimestamp(self._latest_key[2], tz=timezone.utc)
                self._output.mkdir(parents=True, exist_ok=True)
                path = self._output / (
                    f"training_{self._role}_{captured_at:%Y%m%d_%H%M%S_%f}_"
                    f"{self._latest.sequence}_{uuid4().hex[:8]}.jpg"
                )
                try:
                    with path.open("xb") as stream:
                        data = encoded.tobytes()
                        if stream.write(data) != len(data):
                            raise OSError("Incomplete original JPEG write.")
                except FileExistsError:
                    raise
                except Exception:
                    path.unlink(missing_ok=True)
                    raise
                response.success, response.message = True, str(path)
                self.get_logger().info(f"Training original saved: {path}")
            except Exception as exc:
                response.success, response.message = False, str(exc)
            return response

    def readiness(self):
        with self._resource_lock:
            if self._closed or self._stopping:
                return False, "Training capture is stopping."
            if self._latest is None or not realtime_camera_frame_is_fresh(self._latest):
                return False, "No fresh camera original is available."
            return True, f"Fresh {self._role.upper()} original available."

    def prepare_shutdown(self):
        self._stopping = True

    def destroy_node(self):
        with self._resource_lock:
            self._closed = True
            self._latest = None
        return super().destroy_node()
