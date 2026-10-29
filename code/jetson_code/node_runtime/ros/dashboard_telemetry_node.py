"""Read-only ROS observations to an opt-in Firebase telemetry worker."""
import math
import threading

from rclpy.node import Node
from std_msgs.msg import Float64, String
from rail_robot_interfaces.msg import EncoderState, MissionStatus, ModelTiming

from .common import qos_sensor, qos_events, qos_state, stamp_seconds
from ..live_dashboard import LatestTelemetryWorker, TelemetrySnapshot
from ..settings import AUTOMATIC_RAIL_SECTION_TARGET


class DashboardTelemetryNode(Node):
    def __init__(self, **kwargs):
        super().__init__("rail_dashboard_telemetry", **kwargs)
        self._lock = threading.Lock()
        self._snapshot = TelemetrySnapshot(AUTOMATIC_RAIL_SECTION_TARGET)
        self._worker = LatestTelemetryWorker()
        self._closed = False
        self._last_error = None
        for role in ("side", "front"):
            self.create_subscription(Float64, f"debug/camera/{role}/fps",
                                     lambda msg, role=role: self._fps(role, msg), qos_sensor)
        self.create_subscription(ModelTiming, "debug/vision/timing", self._timing, qos_events)
        self.create_subscription(MissionStatus, "mission/state", self._mission, qos_state)
        self.create_subscription(String, "runtime/stm_bridge_health", self._stm_health, qos_state)
        self.create_subscription(EncoderState, "debug/encoder/state", self._encoder, qos_state)
        self.create_timer(2.0, self._publish)

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _fps(self, role, message):
        # Float64 has no source header: this is explicitly reception time.
        observed_at = self._now()
        with self._lock:
            self._snapshot.metric(role, "fps", message.data, observed_at)
            self._snapshot.health_signal(
                f"camera_{role}", "ok" if math.isfinite(message.data) and message.data > 0 else "unknown",
                observed_at)

    def _timing(self, message):
        role = "front" if message.camera_role == "top" else message.camera_role
        model = {"rust": "rustMs", "crack": "crackMs", "obstacle": "obstacleMs"}.get(message.model)
        if model:
            observed_at = stamp_seconds(message.header.stamp)
            with self._lock:
                self._snapshot.metric(role, model, message.processing_ms, observed_at)
                if role in ("side", "front") and math.isfinite(message.processing_ms) and message.processing_ms >= 0:
                    self._snapshot.health_signal("vision", "ok", observed_at)

    def _mission(self, message):
        observed_at = stamp_seconds(message.header.stamp)
        with self._lock:
            self._snapshot.mission_status(message.mode, message.initial_count, message.rescan_count,
                                          message.failed, observed_at)
            self._snapshot.health_signal("mission", "fault" if message.failed or message.error else "ok",
                                         observed_at)

    def _stm_health(self, message):
        state = {"ready": "ok", "fault": "fault", "unknown": "unknown"}.get(message.data, "unknown")
        with self._lock:
            self._snapshot.health_signal("stm_bridge", state, self._now())

    def _encoder(self, message):
        # Only actuator-debug mode publishes actual MCU feedback. Mission commands do not.
        state = "fault" if message.error else "observed" if message.feedback_valid else "unknown"
        with self._lock:
            observed_at = self._now()
            for side in ("left", "right"):
                self._snapshot.health_signal(f"encoder_{side}", state, observed_at)

    def _publish(self):
        if self._closed:
            return
        topics = [(name, self.count_publishers(name), types) for name, types in self.get_topic_names_and_types()]
        with self._lock:
            payload = self._snapshot.payload(topics, self._now())
        self._worker.submit(payload)
        error = self._worker.last_error
        if error and error != self._last_error:
            self.get_logger().warning("Live dashboard telemetry unavailable: " + error)
        self._last_error = error

    def prepare_shutdown(self):
        if not self._closed:
            self._closed = True
            self._worker.close()

    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
