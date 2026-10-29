"""Optional five-second Firebase fallback, consuming only composed GUI JPEGs."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from firebase_uploader import FirebaseLivePreviewPublisher, MAX_IMAGE_BYTES
from .common import qos_sensor, qos_state, source_monotonic, stamp_seconds


class FirebaseSnapshotNode(Node):
    """A latest-only consumer; network work stays in the existing upload worker.

    No connection state or WebRTC switching is invented here. When enabled,
    snapshots remain available at the existing five-second interval.
    """

    def __init__(self, *, publisher_factory=FirebaseLivePreviewPublisher.from_environment,
                 clock=time.monotonic, node_name="rail_firebase_snapshot", **kwargs):
        super().__init__(node_name, **kwargs)
        self._lock = threading.RLock()
        self._monotonic_clock = clock
        self._closed = self._stopping = False
        self._latest = None
        self._last_submitted = float("-inf")
        self._last_stamp = None
        self._uploader = None
        self._upload_pending = False
        descriptor = ParameterDescriptor(read_only=True)
        topic = self.declare_parameter("gui_output_topic", "/rail_robot/gui/image/compressed",
                                       descriptor=descriptor).value
        self._status = self.create_publisher(String, "gui/firebase_snapshot/status", qos_state)
        try:
            self._uploader = publisher_factory()
            self._report("ready" if self._uploader is not None else "disabled: Firebase configuration unavailable")
        except Exception as exc:
            self._report(f"disabled: Firebase configuration error ({type(exc).__name__})")
        self.create_subscription(CompressedImage, topic, self._receive, qos_sensor)
        self.create_timer(0.2, self._tick)

    def _report(self, detail):
        message = String()
        message.data = detail
        self._status.publish(message)

    def _receive(self, message):
        with self._lock:
            if self._closed or self._stopping or self._uploader is None:
                return
            if not message.header.frame_id or message.format != "bgr8; jpeg compressed bgr8":
                return
            if not 0 < len(message.data) <= MAX_IMAGE_BYTES:
                return
            try:
                source_monotonic(message.header.stamp, self.get_clock().now().nanoseconds / 1e9,
                                 self._monotonic_clock(), 1.0)
                stamp = stamp_seconds(message.header.stamp)
            except ValueError:
                return
            if self._last_stamp is not None and stamp <= self._last_stamp:
                return
            self._last_stamp = stamp
            self._latest = message

    def _tick(self):
        with self._lock:
            if self._closed or self._stopping or self._uploader is None:
                return
            try:
                if self._uploader.poll():
                    self._upload_pending = False
                    self._report("uploaded")
            except Exception as exc:
                self._upload_pending = False
                # Never emit runtime/fault or expose credentials/HTTP bodies.
                self._report(f"upload failed ({type(exc).__name__}); retry on next fresh snapshot")
            if self._upload_pending or self._monotonic_clock() - self._last_submitted < 5.0 or self._latest is None:
                return
            message = self._latest
            self._latest = None
            try:
                source_monotonic(message.header.stamp, self.get_clock().now().nanoseconds / 1e9,
                                 self._monotonic_clock(), 1.0)
                frame = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape not in ((720, 640, 3), (360, 1280, 3), (720, 1280, 3)):
                    raise ValueError("Invalid composed GUI JPEG dimensions")
                # The existing publisher also enforces 5 s and one active upload.
                if self._uploader.submit(frame):
                    self._upload_pending = True
                    self._last_submitted = self._monotonic_clock()
                    self._report("queued")
            except Exception as exc:
                self._report(f"snapshot skipped ({type(exc).__name__})")

    def prepare_shutdown(self):
        with self._lock:
            self._stopping = True
            self._latest = None

    def destroy_node(self):
        with self._lock:
            if self._closed:
                return True
            self._closed = self._stopping = True
            self._latest = None
            if self._uploader is not None:
                self._uploader.close()
        return super().destroy_node()
