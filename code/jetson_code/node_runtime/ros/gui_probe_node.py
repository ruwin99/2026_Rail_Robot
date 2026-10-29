"""Bounded observation of the GUI topic; never opens cameras, GPU or UART."""
from __future__ import annotations

from collections import deque
import json
import math
import time

import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .common import qos_sensor, stamp_seconds


class GuiProbeNode(Node):
    """Measure delivery FPS and composition-to-receive age, not camera latency."""

    def __init__(self, *, clock=time.monotonic, node_name="rail_gui_probe", **kwargs):
        super().__init__(node_name, **kwargs)
        descriptor = ParameterDescriptor(read_only=True)
        self._duration = self.declare_parameter("duration_seconds", 60.0, descriptor=descriptor).value
        if not isinstance(self._duration, (int, float)) or not math.isfinite(self._duration) or not 1 <= self._duration <= 3600:
            raise ValueError("duration_seconds must be between 1 and 3600")
        topic = self.declare_parameter("gui_output_topic", "/rail_robot/gui/image/compressed", descriptor=descriptor).value
        status_topic = self.declare_parameter("gui_status_topic", "/rail_robot/gui/status", descriptor=descriptor).value
        self._monotonic_clock = clock
        self._started = clock()
        self._first = self._last = self._last_stamp = None
        self._received = self._invalid = self._bytes = 0
        self._ages = deque(maxlen=60000)
        self._intervals = deque(maxlen=60000)
        self._frame_ids = set()
        self._first_image_shape = None
        self._last_decode = float("-inf")
        self._decoded = self._status_count = self._stale_sources = 0
        self._invalid_status = 0
        self._status_stamp = self._status_sequence = self._last_status_received = None
        self._sources = {}
        self._source_keys = {}
        self._source_advances = {"side": 0, "front": 0}
        self._source_stream_changes = {"side": 0, "front": 0}
        self._older_source_observations = {"side": 0, "front": 0}
        self._max_source_age_ms = None
        self._finished = False
        self.summary = None
        self.create_subscription(CompressedImage, topic, self._receive, qos_sensor)
        self.create_subscription(String, status_topic, self._receive_status, qos_sensor)
        self.create_timer(0.2, self._tick)

    def _receive(self, message):
        if self._finished:
            return
        now = self._monotonic_clock()
        try:
            stamp = stamp_seconds(message.header.stamp)
            age = self.get_clock().now().nanoseconds / 1e9 - stamp
            if not message.header.frame_id or not message.data or message.format != "bgr8; jpeg compressed bgr8":
                raise ValueError("Invalid GUI image envelope")
            if not math.isfinite(age) or age < 0 or (self._last_stamp is not None and stamp <= self._last_stamp):
                raise ValueError("Nonmonotonic/future composition stamp")
            if now - self._last_decode >= 1.0:
                image = cv2.imdecode(np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape not in ((720, 640, 3), (360, 1280, 3), (720, 1280, 3)):
                    raise ValueError("Expected 640x720, 1280x360 or 1280x720 JPEG")
                self._first_image_shape = list(image.shape)
                self._last_decode = now
                self._decoded += 1
            if self._first is None:
                self._first = now
            if self._last is not None:
                self._intervals.append((now - self._last) * 1000)
            self._last, self._last_stamp = now, stamp
            self._received += 1
            self._bytes += len(message.data)
            self._ages.append(age * 1000)
            if len(self._frame_ids) < 8:
                self._frame_ids.add(message.header.frame_id)
        except (ValueError, TypeError, cv2.error):
            self._invalid += 1

    def _receive_status(self, message):
        if self._finished or len(message.data) > 100000:
            return
        try:
            status = json.loads(message.data)
            if status.get("schema_version") != 1 or status.get("state") != "published":
                return
            stamp, sequence = status["image_stamp_seconds"], status["sequence"]
            if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError("Invalid status timestamp")
            age_at_receive = self.get_clock().now().nanoseconds / 1e9 - stamp
            if (not 0 <= age_at_receive <= 1.0 or type(sequence) is not int or sequence < 1
                    or not status.get("frame_id")
                    or (self._frame_ids and status["frame_id"] not in self._frame_ids)
                    or (self._status_stamp is not None and stamp <= self._status_stamp)
                    or (self._status_sequence is not None and sequence <= self._status_sequence)):
                raise ValueError("Stale, future or reordered status")
            sources = status["sources"]
            if not isinstance(sources, dict):
                raise ValueError("Invalid source metadata")
            self._status_stamp, self._status_sequence = stamp, sequence
            self._last_status_received = self._monotonic_clock()
            self._status_count += 1
            for role in ("side", "front"):
                source = sources.get(role) or {}
                age = source.get("source_age_ms")
                if isinstance(age, (int, float)):
                    age += age_at_receive * 1000
                if (source.get("fresh") is not True or not isinstance(age, (int, float))
                        or not math.isfinite(age) or not 0 <= age <= 800):
                    self._stale_sources += 1
                else:
                    self._max_source_age_ms = max(self._max_source_age_ms or 0.0, age)
                key = (source.get("stream_id"), source.get("sequence"))
                previous = self._source_keys.get(role)
                if key[0] and type(key[1]) is int:
                    if previous is not None and previous[0] == key[0] and key[1] > previous[1]:
                        self._source_advances[role] += 1
                    elif previous is not None and previous[0] != key[0]:
                        self._source_stream_changes[role] += 1
                    elif previous is not None and key[1] < previous[1]:
                        # A late exact-frame overlay can legitimately select an
                        # older original. Do not call it a transport error or
                        # count its later return as new camera progress.
                        self._older_source_observations[role] += 1
                    if previous is None or key[0] != previous[0] or key[1] >= previous[1]:
                        self._source_keys[role] = key
                self._sources[role] = dict(source, age_at_receive_ms=age)
        except (ValueError, KeyError, TypeError, AttributeError):
            self._invalid_status += 1

    def _tick(self):
        if self._finished or self._monotonic_clock() - self._started < self._duration:
            return
        self._finished = True
        elapsed = self._monotonic_clock() - self._started
        active = 0 if self._first is None else self._last - self._first
        self.summary = {
            "duration_seconds": elapsed, "received": self._received,
            "invalid_messages": self._invalid,
            "full_window_fps": self._received / elapsed,
            "active_window_fps": (self._received - 1) / active if active > 0 else 0.0,
            "first_image_shape": self._first_image_shape,
            "decoded_checks": self._decoded,
            "status_messages": self._status_count,
            "invalid_status_messages": self._invalid_status,
            "time_since_last_status_seconds": None if self._last_status_received is None else self._monotonic_clock() - self._last_status_received,
            "last_sources": self._sources,
            "source_sequence_advances": self._source_advances,
            "source_stream_changes": self._source_stream_changes,
            "source_older_frame_observations": self._older_source_observations,
            "missing_or_stale_source_observations": self._stale_sources,
            "max_fresh_source_age_ms": self._max_source_age_ms,
            "frame_ids": sorted(self._frame_ids),
            "composition_to_receive_ms_p95": float(np.percentile(self._ages, 95)) if self._ages else None,
            "composition_to_receive_ms_max": max(self._ages) if self._ages else None,
            "interarrival_ms_p95": float(np.percentile(self._intervals, 95)) if self._intervals else None,
            "interarrival_ms_max": max(self._intervals) if self._intervals else None,
            "megabits_per_second": self._bytes * 8 / elapsed / 1e6,
            "time_since_last_frame_seconds": None if self._last is None else self._monotonic_clock() - self._last,
            "clock_note": "ROS clocks must be synchronized; image stamp is composition time, not camera capture time.",
        }
        self.get_logger().info(json.dumps(self.summary, sort_keys=True, allow_nan=False))

    def shutdown_requested(self):
        return self._finished
