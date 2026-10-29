"""Presentation-only GUI composition and bounded JPEG output with source identity."""
from __future__ import annotations

from array import array

from collections import OrderedDict
from copy import deepcopy
import json
import time
import threading
from types import SimpleNamespace

import cv2
import numpy as np
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rail_robot_interfaces.msg import (
    ActuatorDecision, CameraFrame as CameraFrameMessage, DetectionSample, MissionStatus, ModelTiming,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, Header, String

from obstacle_detector import ObstacleDetection
from rust_detector import LetterboxTransform, result_from_class_map

from ..display import (
    annotate_realtime_control_results,
    annotate_top_realtime_results,
    draw_realtime_roi_guide,
    open_latest_frame_display,
    stack_dual_camera_displays,
)
from ..settings import (
    CAPTURE_SCAN_MODE, FRAME_HEIGHT, FRAME_RATE, FRAME_WIDTH,
    REALTIME_RESULT_MAX_AGE_SECONDS, RESCAN_MODE, RESCAN_RETURN_MODE, RETURN_MODE,
    REALTIME_RUST_ROI_TOP, REALTIME_RUST_ROI_BOTTOM,
    REALTIME_CRACK_ROI_TOP, REALTIME_CRACK_ROI_BOTTOM,
)
from ..metrics import FrameRateMeter
from .common import qos_events, qos_sensor, qos_state, source_monotonic, stamp_from_seconds, stamp_seconds
from .images import decode_image, message_to_frame


# 카메라 원본과 분석·상태 토픽을 화면으로 합성한다. WebRTC 전송은 별도 소비자가 담당한다.
# 박스·마스크 그리기와 창 작업자는 node_runtime/display.py, ROS 이미지 해석은 ros/images.py를 본다.
# 부식 class_map의 표시 객체 복원은 rust_detector.py, 장애물 표시 객체는 obstacle_detector.py로 이어진다.
GUI_DEFAULTS = {
    "local_gui": True,
    "gui_output": True,
    "gui_publish_rate_hz": 12.0,
    "gui_jpeg_quality": 80,
    "gui_output_topic": "/rail_robot/gui/image/compressed",
    "gui_status_topic": "/rail_robot/gui/status",
    "gui_frame_id": "rail_robot_gui",
    "profile_trace": False,
}


def _parameter(node, name, supplied):
    if supplied is not None:
        return supplied
    if node.has_parameter(name):
        return node.get_parameter(name).value
    return node.declare_parameter(name, GUI_DEFAULTS[name],
                                  descriptor=ParameterDescriptor(read_only=True)).value


class DisplayNode(Node):
    """Render masks only over the original stream, sequence, and timestamp.

    A bounded frame history allows delayed inference results to find their own
    original. It is never used by control. If no matching result is available,
    the preview shows the newest raw frame without a historical overlay.
    """

    def __init__(
        self,
        *,
        local_gui=None,
        gui_output=None,
        gui_publish_rate_hz=None,
        gui_jpeg_quality=None,
        gui_output_topic=None,
        gui_status_topic=None,
        gui_frame_id=None,
        profile_trace=None,
        display_factory=open_latest_frame_display,
        node_name: str = "rail_display",
        **node_kwargs,
    ) -> None:
        super().__init__(node_name, **node_kwargs)
        self._resource_lock = threading.RLock()
        self._render_guard = threading.Lock()
        self._closed = False
        self._stopping = False
        self._exit_requested = False
        self._display = None
        self._frames = {"side": OrderedDict(), "top": OrderedDict()}
        self._frame_ids = {"side": {}, "top": {}}
        self._samples = {"side": {}, "top": {}}
        # 카메라·STM이 끊기면 거부 경고가 초당 수십 줄 쌓인다. 5초에 한 번, 누적 건수와 함께 남긴다.
        self._rejected_samples = 0
        self._reject_logged_at = float("-inf")
        self._model_timings = {}
        self._trace_model_timings = OrderedDict()
        self._trace_previous_sources = {}
        self._trace_render_drop_count = 0
        self._camera_fps = {}
        self._mission = None
        self._control = {}
        self._source_metadata = {}
        self._latest_sources = {}
        self._output_sequence = 0
        self._published_count = 0
        self._published_meter = FrameRateMeter()
        self._previous_telemetry = None
        self._last_presentation_error = None
        self._frame_capacity = int(FRAME_RATE * REALTIME_RESULT_MAX_AGE_SECONDS) + 2
        try:
            self._local_gui = _parameter(self, "local_gui", local_gui)
            self._gui_output = _parameter(self, "gui_output", gui_output)
            self._publish_rate = _parameter(self, "gui_publish_rate_hz", gui_publish_rate_hz)
            self._jpeg_quality = _parameter(self, "gui_jpeg_quality", gui_jpeg_quality)
            self._output_topic = _parameter(self, "gui_output_topic", gui_output_topic)
            self._status_topic = _parameter(self, "gui_status_topic", gui_status_topic)
            self._frame_id = _parameter(self, "gui_frame_id", gui_frame_id)
            self._profile_trace = _parameter(self, "profile_trace", profile_trace)
            if not isinstance(self._local_gui, bool):
                raise ValueError("local_gui must be a boolean.")
            if not isinstance(self._gui_output, bool):
                raise ValueError("gui_output must be a boolean.")
            if not isinstance(self._profile_trace, bool):
                raise ValueError("profile_trace must be a boolean.")
            if isinstance(self._publish_rate, bool) or not isinstance(self._publish_rate, (float, int)) or not np.isfinite(self._publish_rate) or not 10 <= self._publish_rate <= 15:
                raise ValueError("gui_publish_rate_hz must be 10 through 15.")
            if isinstance(self._jpeg_quality, bool) or not isinstance(self._jpeg_quality, int) or not 1 <= self._jpeg_quality <= 100:
                raise ValueError("gui_jpeg_quality must be an integer from 1 through 100.")
            for name, value in (("gui_output_topic", self._output_topic), ("gui_status_topic", self._status_topic), ("gui_frame_id", self._frame_id)):
                if not isinstance(value, str) or not value.strip() or any(character.isspace() for character in value):
                    raise ValueError(f"{name} must be a nonempty identifier without whitespace.")
            if self._output_topic == self._status_topic:
                raise ValueError("GUI image and status topics must differ.")
            self._image_output = self._status_output = self._trace_output = None
            if self._gui_output:
                self._image_output = self.create_publisher(CompressedImage, self._output_topic, qos_sensor)
                self._status_output = self.create_publisher(String, self._status_topic, qos_sensor)
                if self._profile_trace:
                    from rail_robot_interfaces.msg import FrameTrace
                    self._trace_output = self.create_publisher(
                        FrameTrace, "/rail_robot/debug/frame_trace", qos_sensor)
            # 로컬 창과 JPEG 토픽은 독립 옵션이므로 창 없이도 gui_output으로 전송할 수 있다.
            # 창 초기화 실패를 기록해도 토픽 생성은 유지된다. 영상 전송이 없으면 _gui_output과
            # _output_topic을 먼저 확인하고, 창 문제와 JPEG 발행 문제를 분리해 추적한다.
            if self._local_gui:
                try:
                    self._display = display_factory()
                except Exception as exc:
                    self.get_logger().error(f"Local preview unavailable; JPEG output remains enabled: {exc}")
            self._input_subscriptions = [
                self.create_subscription(
                    CameraFrameMessage,
                    f"camera/{role}",
                    lambda message, role=role: self._receive_frame(role, message),
                    qos_sensor,
                )
                for role in ("side", "top")
            ]
            self._input_subscriptions.append(
                self.create_subscription(DetectionSample, "vision/results", self._receive_sample, qos_events)
            )
            self._input_subscriptions.append(
                self.create_subscription(ModelTiming, "debug/vision/timing", self._receive_timing, qos_events)
            )
            for role, topic_role in (("side", "side"), ("top", "front")):
                self._input_subscriptions.append(self.create_subscription(
                    Float64, f"debug/camera/{topic_role}/fps",
                    lambda message, role=role: self._receive_fps(role, message), qos_sensor,
                ))
            self._input_subscriptions.append(self.create_subscription(
                MissionStatus, "mission/state", self._receive_mission, qos_state))
            self._input_subscriptions.append(self.create_subscription(
                ActuatorDecision, "control/decisions", self._receive_control, qos_events))
            self._timer = self.create_timer(1.0 / self._publish_rate, self._render)
        except Exception:
            self.destroy_node()
            raise

    @staticmethod
    def _key(message):
        # 마스크 대응 키: 카메라 실행 세션·프레임 번호·원본 시각이 모두 같아야 한다.
        # 마스크가 안 보일 때 sequence만 비교하지 말고 stream_id와 stamp도 함께 확인한다.
        return message.stream_id, message.sequence, stamp_seconds(message.header.stamp)

    def _receive_mission(self, message):
        with self._resource_lock:
            if self._closed or self._stopping:
                return
            try:
                observed = source_monotonic(message.header.stamp, self.get_clock().now().nanoseconds / 1e9,
                                            time.monotonic(), 86400.0)
                stamp = stamp_seconds(message.header.stamp)
                if self._mission is not None and stamp < self._mission[0]["source_stamp_seconds"]:
                    return
                self._mission = ({
                    "mode": message.mode, "initial_count": int(message.initial_count),
                    "rescan_count": int(message.rescan_count), "failed": bool(message.failed),
                    "error": message.error, "source_stamp_seconds": stamp,
                }, observed)
            except (ValueError, TypeError) as exc:
                self.get_logger().warning(f"Preview mission status rejected: {exc}")

    def _receive_control(self, message):
        with self._resource_lock:
            if self._closed or self._stopping or message.role not in ("side", "front"):
                return
            try:
                observed = source_monotonic(message.header.stamp, self.get_clock().now().nanoseconds / 1e9,
                                            time.monotonic(), 86400.0)
                stamp = stamp_seconds(message.header.stamp)
                valid_until = stamp_seconds(message.valid_until)
                if not np.isfinite(message.cleaner_duty_percent) or not 0 <= message.cleaner_duty_percent <= 100 or not np.isfinite(valid_until):
                    raise ValueError("Invalid presentation control request.")
                previous = self._control.get(message.role)
                if previous is not None and stamp < previous[0]["source_stamp_seconds"]:
                    return
                self._control[message.role] = ({
                    "session_id": message.session_id, "sequence": int(message.sequence),
                    "role": message.role, "cleaner_duty_percent": float(message.cleaner_duty_percent),
                    "pump_on": bool(message.pump_on), "reason": message.reason,
                    "source_stamp_seconds": stamp, "valid_until_seconds": valid_until,
                }, observed)
            except (ValueError, TypeError) as exc:
                self.get_logger().warning(f"Preview control status rejected: {exc}")

    def _observed_status(self, now, ros_now):
        # 화면 상태는 수신한 관측값에 나이와 유효 여부를 붙인 것이다. 과거값도 STALE/EXPIRED로
        # 설명할 수 있게 남겨두며, control 값은 MCU의 실제 동작 확인이 아니라 제어 요청이다.
        mission = None
        if self._mission is not None:
            value, observed = self._mission
            age = now - observed
            mission = dict(value, age_ms=age * 1000, fresh=0 <= age <= 0.8)
        control = {}
        for role in ("side", "front"):
            if role not in self._control:
                control[role] = None
                continue
            value, observed = self._control[role]
            age = now - observed
            control[role] = dict(value, age_ms=age * 1000,
                                 valid=0 <= age <= 0.8 and ros_now <= value["valid_until_seconds"])
        return mission, control

    def _receive_fps(self, role, message):
        with self._resource_lock:
            if not self._closed and not self._stopping and np.isfinite(message.data) and message.data >= 0:
                self._camera_fps[role] = (float(message.data), time.monotonic())

    def _receive_timing(self, message):
        with self._resource_lock:
            if self._closed or self._stopping:
                return
            key = (message.camera_role, message.model)
            if key not in (("side", "rust"), ("side", "crack"), ("top", "obstacle"), ("top", "crack")):
                return
            if not message.stream_id or not np.isfinite(message.processing_ms) or message.processing_ms < 0:
                return
            try:
                completed = source_monotonic(message.header.stamp,
                    self.get_clock().now().nanoseconds / 1e9, time.monotonic(), 2.0)
            except ValueError:
                return
            previous = self._model_timings.get(key)
            if previous is None or completed > previous[1]:
                self._model_timings[key] = (float(message.processing_ms), completed)
            # FrameTrace uses the exact camera source identity, not merely the latest model duration.
            # The cache is bounded independently of GUI rendering and keyed by stream + sequence so a
            # camera restart or delayed timing event cannot be paired with another frame.
            stream_id = getattr(message, "stream_id", "")
            sequence = getattr(message, "sequence", 0)
            execution_id = getattr(message, "execution_id", "")
            started_ns = getattr(message, "processing_start_monotonic_ns", 0)
            ended_ns = getattr(message, "processing_end_monotonic_ns", 0)
            if (stream_id and type(sequence) is int and sequence > 0 and execution_id
                    and type(started_ns) is int and started_ns > 0
                    and type(ended_ns) is int and ended_ns >= started_ns):
                source_key = (message.camera_role, message.model, stream_id, sequence)
                self._trace_model_timings[source_key] = (started_ns, ended_ns)
                self._trace_model_timings.move_to_end(source_key)
                while len(self._trace_model_timings) > self._frame_capacity * 4:
                    self._trace_model_timings.popitem(last=False)

    def _metrics_lines(self, now):
        # 카메라 FPS와 모델 시간은 각각의 측정 토픽에서 받는다. 화면 갱신 횟수로 추정하지 않는다.
        # 표본이 없거나 FPS 1초/모델 시간 2초의 유효기간을 넘으면 0 대신 N/A로 표시한다.
        def current(values, key, unit, max_age):
            sample = values.get(key)
            if sample is None or not 0 <= now - sample[1] <= max_age:
                return "N/A"
            return f"{sample[0]:.1f}{unit}"
        fps = lambda role: current(self._camera_fps, role, "", 1.0)
        ms = lambda role, model: current(self._model_timings, (role, model), " ms", 2.0)
        return [
            f"Camera FPS  SIDE {fps('side')}  FRONT {fps('top')}",
            f"SIDE   Rust {ms('side', 'rust')}   Crack {ms('side', 'crack')}",
            f"FRONT  Obstacle {ms('top', 'obstacle')}   Crack {ms('top', 'crack')}",
        ]

    def _draw_metrics(self, image, now):
        # Draw on the composed preview only; camera originals stay unchanged.
        lines = self._metrics_lines(now)
        scale, thickness, line_height, margin = 0.6, 1, 24, 12
        width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines)
        top = max(0, image.shape[0] - margin - line_height * len(lines) - 12)
        cv2.rectangle(image, (margin, top), (min(image.shape[1] - 1, margin + width + 20), image.shape[0] - margin), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            cv2.putText(image, line, (margin + 10, top + 22 + line_height * index),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 255, 230), thickness, cv2.LINE_AA)
        return image

    def _receive_frame(self, role, message) -> None:
        with self._resource_lock:
            if self._closed or self._stopping:
                return
            self._receive_frame_locked(role, message)

    def _receive_frame_locked(self, role, message) -> None:
        # 역할·stream_id·시각·이미지 형식을 검증한 원본만 역할별 OrderedDict에 넣는다.
        # 프레임 순서가 뒤로 가면 버리고 새 카메라 세션이면 이전 원본과 마스크 이력을 비운다.
        # _frame_capacity를 넘긴 오래된 원본부터 삭제하므로 느린 추론을 무한정 기다리지 않는다.
        if message.camera_role != role or not message.stream_id:
            return
        try:
            frame = message_to_frame(
                message,
                self.get_clock().now().nanoseconds / 1e9,
                time.monotonic(),
                max_age=REALTIME_RESULT_MAX_AGE_SECONDS,
            )
            frames = self._frames[role]
            key = self._key(message)
            if frames:
                latest = next(reversed(frames))
                if key[2] < latest[2] or (key[0] == latest[0] and key[1] <= latest[1]):
                    return
                if key[0] != latest[0]:
                    frames.clear()
                    self._frame_ids[role].clear()
                    self._samples[role].clear()
                    for timing_key in tuple(self._trace_model_timings):
                        if timing_key[0] == role and timing_key[2] != key[0]:
                            del self._trace_model_timings[timing_key]
            frames[key] = frame
            self._frame_ids[role][key] = message.header.frame_id
            self._latest_sources[role] = (key, frame.read_completed_at)
            while len(frames) > self._frame_capacity:
                old_key, _ = frames.popitem(last=False)
                self._frame_ids[role].pop(old_key, None)
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Preview frame rejected: {exc}")

    def _receive_sample(self, message) -> None:
        with self._resource_lock:
            if self._closed or self._stopping:
                return
            self._receive_sample_locked(message)

    def _receive_sample_locked(self, message) -> None:
        # 유효한 최신 모델 결과만 detector별로 보관한다. 여기서 원본 위에 바로 그리지 않고
        # _render_role에서 대응 원본을 찾으므로 메시지 도착 순서가 달라도 잘못 겹치지 않는다.
        # 표시에 빠진 결과는 valid, detector 이름, stamp 만료 여부부터 확인한다.
        role = message.camera_role
        if role not in self._samples or message.detector not in ("rust", "crack", "obstacle"):
            return
        if not message.valid or not message.stream_id:
            return
        try:
            source_monotonic(
                message.header.stamp,
                self.get_clock().now().nanoseconds / 1e9,
                time.monotonic(),
                REALTIME_RESULT_MAX_AGE_SECONDS,
            )
            previous = self._samples[role].get(message.detector)
            if previous is None or self._key(message)[2] > self._key(previous)[2]:
                self._samples[role][message.detector] = message
        except (ValueError, TypeError) as exc:
            self._rejected_samples += 1
            now = time.monotonic()
            if now - self._reject_logged_at >= 5.0:
                self.get_logger().warning(
                    f"Preview sample rejected: {exc} ({self._rejected_samples} since last report)")
                self._rejected_samples = 0
                self._reject_logged_at = now

    @staticmethod
    def _result(sample):
        # 공통 ROS 메시지를 그리기 함수용 객체로 복원한다. 부식·균열 마스크 높이가 ROI와
        # 다르거나 장애물 박스/점수/클래스 개수가 다르면 표시 오류로 처리한다.
        flat_boxes = list(sample.boxes_xyxy)
        if len(flat_boxes) % 4 or not np.isfinite(flat_boxes).all():
            raise ValueError("Malformed preview boxes")
        boxes = [tuple(flat_boxes[index:index + 4]) for index in range(0, len(flat_boxes), 4)]
        if sample.detector == "obstacle":
            if len(boxes) != len(sample.confidences) or len(boxes) != len(sample.class_ids):
                raise ValueError("Obstacle box metadata count mismatch")
            return SimpleNamespace(
                detections=tuple(
                    ObstacleDetection(box, score, class_id)
                    for box, score, class_id in zip(boxes, sample.confidences, sample.class_ids)
                ),
            )
        mask = decode_image(sample.mask)
        height = (
            REALTIME_RUST_ROI_BOTTOM - REALTIME_RUST_ROI_TOP
            if sample.detector == "rust"
            else REALTIME_CRACK_ROI_BOTTOM - REALTIME_CRACK_ROI_TOP
        )
        if mask.shape != (height, FRAME_WIDTH):
            raise ValueError("Preview mask does not match its detector ROI")
        if sample.detector == "rust":
            return result_from_class_map(
                mask,
                LetterboxTransform(height, FRAME_WIDTH, height, FRAME_WIDTH, 0, 0),
                sample.detector_method,
            )
        if not np.isin(mask, (0, 255)).all():
            raise ValueError("Preview crack mask must be binary")
        pixels = int(np.count_nonzero(mask))
        return SimpleNamespace(
            mask=mask,
            boxes=[(round(x1), round(y1), round(x2 - x1), round(y2 - y1)) for x1, y1, x2, y2 in boxes],
            crack_pixels=pixels,
            inspected_pixels=mask.size,
            crack_ratio=pixels / mask.size,
            detected=pixels > 0,
            method=sample.detector_method,
        )

    def _render_role(self, role, now):
        # 먼저 만료 원본을 치운다. 남은 원본이 없으면 검은 화면과 waiting/stale 상태를 반환한다.
        # 분석 결과가 대응되는 원본이 없으면 최신 원본만 표시하며 상태는 raw로 기록한다.
        frames = self._frames[role]
        while frames:
            first_key = next(iter(frames))
            age = now - frames[first_key].read_completed_at
            if 0.0 <= age <= REALTIME_RESULT_MAX_AGE_SECONDS:
                break
            expired_key, _ = frames.popitem(last=False)
            self._frame_ids[role].pop(expired_key, None)
        if not frames:
            previous = self._latest_sources.get(role)
            self._source_metadata[role] = self._describe_source(
                role, previous[0] if previous else None, previous[1] if previous else None,
                now, "stale" if previous else "waiting", [])
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        matching = {
            detector: sample
            for detector, sample in self._samples[role].items()
            if self._key(sample) in frames
        }
        if not matching:
            key = next(reversed(frames))
            self._source_metadata[role] = self._describe_source(
                role, key, frames[key].read_completed_at, now, "raw", [])
            return next(reversed(frames.values())).frame
        # 결과가 대응되는 원본 중 가장 최신 것을 골라 그 원본의 결과만 겹친다.
        # 따라서 화면 원본은 카메라의 가장 최근 프레임보다 뒤일 수 있다. 지연은 source_age_ms,
        # 어떤 모델이 겹쳐졌는지는 sources의 detectors와 sequence로 확인한다.
        key = max((self._key(sample) for sample in matching.values()), key=lambda value: value[2])
        results = {
            detector: self._result(sample)
            for detector, sample in matching.items()
            if self._key(sample) == key
        }
        frame = frames[key].frame
        self._source_metadata[role] = self._describe_source(
            role, key, frames[key].read_completed_at, now, "annotated", sorted(results))
        if role == "side":
            annotated = annotate_realtime_control_results(frame, results.get("rust"), results.get("crack"))
        else:
            annotated = annotate_top_realtime_results(frame, results.get("obstacle"), results.get("crack"))
        return draw_realtime_roi_guide(
            frame, annotated, primary_roi_label="RUST" if role == "side" else "OBSTACLE"
        )

    @staticmethod
    def _describe_source(role, key, observed, now, state, detectors):
        return {
            "state": state, "camera_role": role,
            "fresh": state in ("raw", "annotated") and observed is not None
                     and 0 <= now - observed <= REALTIME_RESULT_MAX_AGE_SECONDS,
            "stream_id": key[0] if key else None,
            "sequence": key[1] if key else None,
            "source_stamp_seconds": key[2] if key else None,
            "source_age_ms": (now - observed) * 1000 if observed is not None else None,
            "detectors": detectors,
        }

    def _compose_canvas(self, now, ros_now):
        displays = []
        for role in ("side", "top"):
            try:
                displays.append(self._render_role(role, now))
            except Exception as exc:
                previous = self._latest_sources.get(role)
                self._source_metadata[role] = self._describe_source(
                    role, previous[0] if previous else None, previous[1] if previous else None,
                    now, "error", [])
                self._source_metadata[role]["error"] = str(exc)
                displays.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
        canvas = np.full((720, 1280, 3), (16, 20, 24), dtype=np.uint8)
        # Each 1280x720 original is uniformly reduced to 640x360; no stretching.
        camera_panes = stack_dual_camera_displays(*displays)
        # Remote output uses the already-resized panes, FRONT left and SIDE right.
        # Keep the local diagnostic panel's original SIDE | FRONT layout.
        remote_panes = np.ascontiguousarray(np.hstack((camera_panes[:, 640:], camera_panes[:, :640])))
        mission, control = self._observed_status(now, ros_now)
        # Keep the last observed mission mode even if telemetry becomes stale so
        # a missing update does not remove the capture/return indication.
        mode = mission["mode"] if mission is not None else None
        if mode in (CAPTURE_SCAN_MODE, RESCAN_MODE):
            camera_panes = self._mode_overlay(camera_panes, "capture mode", (0, 255, 255))
            remote_panes = self._mode_overlay(remote_panes, "capture mode", (0, 255, 255))
        elif mode in (RETURN_MODE, RESCAN_RETURN_MODE):
            camera_panes = self._mode_placeholder(camera_panes.shape, "return", (0, 165, 255))
            remote_panes = self._mode_placeholder(remote_panes.shape, "return", (0, 165, 255))
        if mode not in (RETURN_MODE, RESCAN_RETURN_MODE):
            for label, x in (("FRONT", 0), ("SIDE", 640)):
                cv2.rectangle(remote_panes, (x + 4, 4), (x + 88, 32), (16, 20, 24), -1)
                cv2.putText(remote_panes, label, (x + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (230, 240, 245), 1, cv2.LINE_AA)
        canvas[76:436] = camera_panes
        sources = {"side": dict(self._source_metadata["side"]),
                   "front": dict(self._source_metadata["top"])}
        lines = [("RAIL ROBOT | PRESENTATION", (16, 27)),
                 ("SIDE", (16, 62)), ("FRONT", (656, 62))]
        for index, role in enumerate(("side", "front")):
            source = sources[role]
            age = "N/A" if source["source_age_ms"] is None else f"{source['source_age_ms']:.0f} ms"
            detectors = ", ".join(source["detectors"]) or "no overlay"
            lines.append((f"{role.upper()} {source['state'].upper()} | seq {source['sequence']} | age {age} | {detectors}",
                          (16, 466 + index * 26)))
        if mission is None:
            mission_line = "MISSION: no status received"
        else:
            freshness = "CURRENT" if mission["fresh"] else "STALE"
            mission_line = (f"MISSION {mission['mode']} | {freshness} | initial {mission['initial_count']} "
                            f"rescan {mission['rescan_count']} | failed={mission['failed']}")
        lines.append((mission_line, (16, 532)))
        for index, role in enumerate(("side", "front")):
            request = control[role]
            if request is None:
                label = f"{role.upper()} CONTROL REQUEST: no observation"
            else:
                state = "VALID" if request["valid"] else "EXPIRED"
                label = (f"{role.upper()} CONTROL REQUEST {state} | cleaner {request['cleaner_duty_percent']:.0f}% "
                         f"pump {'ON' if request['pump_on'] else 'OFF'} | {request['reason']}")
            lines.append((label, (16, 560 + index * 26)))
        if self._previous_telemetry is not None:
            previous = self._previous_telemetry
            fps = "N/A" if previous["published_fps"] is None else f"{previous['published_fps']:.1f}"
            lines.extend([
                (f"Previous JPEG: {previous['jpeg_bytes'] / 1024:.1f} KiB | output {fps} FPS", (660, 642)),
                (f"Compose {previous['composition_ms']:.1f} ms | encode {previous['encode_ms']:.1f} ms", (660, 670)),
            ])
        for label, position in lines:
            cv2.putText(canvas, label[:155], position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 240, 245), 1, cv2.LINE_AA)
        self._draw_metrics(canvas, now)

        def metric(values, key, max_age):
            value = values.get(key)
            return value[0] if value is not None and 0 <= now - value[1] <= max_age else None
        # 로컬 창은 상태판을 포함한 1280×720 canvas, 원격 JPEG는 FRONT 왼쪽 / SIDE 오른쪽
        # 1280×360 remote_panes를 사용한다. 원격 상태 정보는 별도 상태 토픽에 담는다.
        return canvas, {
            "sources": sources, "mission": mission, "control": control,
            "camera_fps": {"side": metric(self._camera_fps, "side", 1.0),
                           "front": metric(self._camera_fps, "top", 1.0)},
            "model_ms": {f"{'front' if role == 'top' else role}_{model}": metric(self._model_timings, (role, model), 2.0)
                         for role, model in (("side", "rust"), ("side", "crack"), ("top", "obstacle"), ("top", "crack"))},
        }, remote_panes

    @staticmethod
    def _mode_placeholder(shape, label, color):
        image = np.zeros(shape, dtype=np.uint8)
        height, width = image.shape[:2]
        cv2.rectangle(image, (6, 6), (width - 7, height - 7), color, 12)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        cv2.putText(image, label, ((width - text_width) // 2, (height + text_height) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3, cv2.LINE_AA)
        return image

    @staticmethod
    def _mode_overlay(frame, label, color):
        image = frame.copy()
        height, width = image.shape[:2]
        cv2.rectangle(image, (6, 6), (width - 7, height - 7), color, 12)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        left = (width - text_width) // 2
        cv2.rectangle(image, (left - 12, 17), (left + text_width + 12, 55), (16, 20, 24), -1)
        cv2.putText(image, label, (left, 47), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
        return image

    def _presentation_error(self, detail):
        if detail != self._last_presentation_error:
            self.get_logger().error(f"GUI output: {detail}")
            self._last_presentation_error = detail

    def _submit_local(self, image):
        # 로컬 창 작업자에 제출하지 못하면 창을 닫고 JPEG 출력을 계속한다.
        # JPEG 출력도 꺼진 실행에서만 표시 종료를 노드 종료 요청으로 바꾼다.
        if self._display is None:
            return
        try:
            if not self._display.check_status() and self._display.submit(image):
                return
        except Exception as exc:
            self._presentation_error(f"Local preview failed; JPEG output continues: {exc}")
        try:
            self._display.close()
        except Exception as exc:
            self._presentation_error(f"Local preview shutdown failed: {exc}")
        self._display = None
        if not self._gui_output:
            self._exit_requested = True
            self._timer.cancel()

    def _publish_status(self, status):
        if self._status_output is None:
            return
        message = String()
        message.data = json.dumps(status, separators=(",", ":"), allow_nan=False)
        self._status_output.publish(message)

    def _build_frame_trace(self, image_header, observations, output_sequence,
                           compose_start_ns, compose_complete_ns, jpeg_complete_ns):
        """Build one diagnostic row from the sources selected for this JPEG.

        Caller holds _resource_lock. Missing/stale sources use sequence zero; old model results
        are never relabelled as current. The publish-complete boundary is filled immediately after
        the GUI CompressedImage publisher returns.
        """
        from rail_robot_interfaces.msg import FrameTrace
        trace = FrameTrace()
        trace.header = deepcopy(image_header)
        trace.output_sequence = output_sequence
        selected = {}
        missing_camera_count = 0
        for output_role, internal_role in (("front", "top"), ("side", "side")):
            source = observations["sources"][output_role]
            stream_id, camera_sequence = source.get("stream_id"), source.get("sequence")
            source_key = (stream_id, camera_sequence, source.get("source_stamp_seconds"))
            valid_source = (
                source.get("fresh") is True
                and source.get("state") in ("raw", "annotated")
                and isinstance(stream_id, str) and bool(stream_id)
                and type(camera_sequence) is int and camera_sequence > 0
                and source_key in self._frames[internal_role]
                and bool(self._frame_ids[internal_role].get(source_key, ""))
            )
            frame = self._frames[internal_role].get(source_key) if valid_source else None
            if valid_source and frame is not None:
                frame_id = self._frame_ids[internal_role][source_key]
                capture_ns = int(frame.read_completed_monotonic_ns)
                selected[internal_role] = (source_key, source)
            else:
                frame_id, stream_id, camera_sequence, capture_ns = "", "", 0, 0
                selected[internal_role] = (None, source)
                missing_camera_count += 1
            setattr(trace, f"{output_role}_frame_id", frame_id)
            setattr(trace, f"{output_role}_stream_id", stream_id)
            setattr(trace, f"{output_role}_camera_sequence", camera_sequence)
            setattr(trace, f"{output_role}_capture_complete_monotonic_ns", capture_ns)

        # Return modes replace camera panes with placeholders, so no live source identity
        # belongs to the composed remote JPEG even if fresh frames remain cached internally.
        mission = observations.get("mission")
        if mission is not None and mission.get("mode") in (
                RETURN_MODE, RESCAN_RETURN_MODE):
            for output_role, internal_role in (("front", "top"), ("side", "side")):
                setattr(trace, f"{output_role}_frame_id", "")
                setattr(trace, f"{output_role}_stream_id", "")
                setattr(trace, f"{output_role}_camera_sequence", 0)
                setattr(trace, f"{output_role}_capture_complete_monotonic_ns", 0)
                selected[internal_role] = (None, selected[internal_role][1])
            missing_camera_count = 2

        model_roles, model_ids, model_streams = [], [], []
        model_sequences, model_starts, model_ends, model_stale = [], [], [], []
        stale_model_count = 0
        expected_models = (("side", "side", "rust"), ("side", "side", "crack"),
                           ("front", "top", "obstacle"), ("front", "top", "crack"))
        for output_role, internal_role, model in expected_models:
            source_key, source = selected[internal_role]
            sample = self._samples[internal_role].get(model)
            result_used = (
                source_key is not None
                and sample is not None
                and model in source.get("detectors", ())
                and self._key(sample) == source_key
            )
            if result_used:
                model_stream = sample.stream_id
                model_sequence = int(sample.sequence)
                timing_key = (internal_role, model, model_stream, model_sequence)
                timing = self._trace_model_timings.get(timing_key)
                model_start, model_end = timing if timing is not None else (0, 0)
                stale = False
            else:
                model_stream, model_sequence = "", 0
                model_start = model_end = 0
                stale = True
                stale_model_count += 1
            # Keep the ROS source role (`top`/`side`) in model provenance; GUI camera fields use
            # presentation names (`front`/`side`) above.
            model_roles.append(internal_role)
            model_ids.append(model)
            model_streams.append(model_stream)
            model_sequences.append(model_sequence)
            model_starts.append(model_start)
            model_ends.append(model_end)
            model_stale.append(stale)

        trace.model_camera_roles = model_roles
        trace.model_ids = model_ids
        trace.model_stream_ids = model_streams
        trace.model_source_sequences = model_sequences
        trace.model_start_monotonic_ns = model_starts
        trace.model_end_monotonic_ns = model_ends
        trace.model_stale = model_stale
        trace.compose_start_monotonic_ns = compose_start_ns
        trace.compose_complete_monotonic_ns = compose_complete_ns
        trace.jpeg_complete_monotonic_ns = jpeg_complete_ns
        trace.gui_publish_start_monotonic_ns = 0
        trace.gui_publish_complete_monotonic_ns = 0
        trace.missing_camera_count = missing_camera_count
        trace.stale_model_count = stale_model_count
        trace.duplicate_frame_count = 0
        trace.source_sequence_gap_count = 0
        trace.gui_render_drop_count = self._trace_render_drop_count
        source_identities = {}
        for internal_role, (source_key, _) in selected.items():
            identity = None if source_key is None else (source_key[0], int(source_key[1]))
            source_identities[internal_role] = identity
            previous = self._trace_previous_sources.get(internal_role)
            if identity is None or previous is None or previous[0] != identity[0]:
                continue
            if identity[1] == previous[1]:
                trace.duplicate_frame_count += 1
            elif identity[1] > previous[1] + 1:
                # An observed display sequence gap is not proof that the camera sensor dropped frames.
                trace.source_sequence_gap_count += identity[1] - previous[1] - 1
        array_lengths = (len(trace.model_camera_roles), len(trace.model_ids), len(trace.model_stream_ids),
                         len(trace.model_source_sequences), len(trace.model_start_monotonic_ns),
                         len(trace.model_end_monotonic_ns), len(trace.model_stale))
        if len(set(array_lengths)) != 1:
            raise ValueError("FrameTrace model arrays are not aligned")
        return trace, source_identities

    def _publish_frame_trace(self, trace, source_identities, publish_start_ns, publish_complete_ns):
        if self._trace_output is None:
            return
        trace.gui_publish_start_monotonic_ns = publish_start_ns
        # This timestamp ends when the local ROS publisher call returns, not when any subscriber
        # receives the trace or a remote GUI renders the JPEG.
        trace.gui_publish_complete_monotonic_ns = publish_complete_ns
        for role, identity in source_identities.items():
            # The corresponding GUI JPEG has already been published. Remember its exact source,
            # including None, so counters compare against the immediately preceding GUI output.
            self._trace_previous_sources[role] = identity
        try:
            self._trace_output.publish(trace)
        except Exception as exc:
            # Optional instrumentation must not turn an already-published GUI JPEG into a fault.
            self.get_logger().warning(f"Frame trace publication skipped: {type(exc).__name__}")

    def _render(self) -> None:
        # 이전 합성·JPEG 인코딩이 끝나지 않았으면 이번 타이머 호출을 건너뛴다.
        # _render_guard를 기다리는 작업이나 인코딩 대기 프레임을 계속 쌓지 않는다.
        # target_rate_hz는 목표이고 published_fps가 실제 발행 결과이므로 두 값을 구분한다.
        if not self._render_guard.acquire(blocking=False):
            if self._trace_output is not None:
                self._trace_render_drop_count += 1
            return
        status = None
        image = None
        local_image = None
        frame_trace = None
        try:
            with self._resource_lock:
                if self._closed or self._stopping:
                    return
                started = time.perf_counter()
                now = time.monotonic()
                ros_now = self.get_clock().now().nanoseconds / 1e9
                compose_start_ns = time.monotonic_ns() if self._trace_output is not None else 0
                local_image, observations, image = self._compose_canvas(now, ros_now)
                composition_ms = (time.perf_counter() - started) * 1000
                composed_monotonic = time.monotonic()
                compose_complete_ns = time.monotonic_ns() if self._trace_output is not None else 0
                # JPEG의 stamp는 합성 시각이다. 원본별 촬영 시각과 경과시간은
                # status의 sources에서 확인해야 캡처부터의 지연을 구별할 수 있다.
                composed_header_stamp = stamp_from_seconds(self.get_clock().now().nanoseconds / 1e9)
                composed_stamp = stamp_seconds(composed_header_stamp)
                composed_header = None
                if self._trace_output is not None:
                    composed_header = Header()
                    composed_header.stamp = deepcopy(composed_header_stamp)
                    composed_header.frame_id = self._frame_id
                self._output_sequence += 1
                for source in observations["sources"].values():
                    if source["source_age_ms"] is not None:
                        source["source_age_ms"] += (composed_monotonic - now) * 1000
                        source["fresh"] = source["fresh"] and 0 <= source["source_age_ms"] <= REALTIME_RESULT_MAX_AGE_SECONDS * 1000
                mission, control = self._observed_status(composed_monotonic, composed_stamp)
                observations.update(mission=mission, control=control)
                ages = [source["source_age_ms"] for source in observations["sources"].values()
                        if source["source_age_ms"] is not None]
                status = dict(observations, schema_version=1, sequence=self._output_sequence,
                              image_stamp_seconds=composed_stamp, frame_id=self._frame_id,
                              width=image.shape[1], height=image.shape[0], layout="camera_only",
                              target_rate_hz=float(self._publish_rate),
                              state="error", error="", composition_ms=composition_ms,
                              encode_ms=0.0, jpeg_bytes=0, published_fps=self._published_meter.fps(composed_monotonic),
                              source_age_max_ms=max(ages) if ages else None)
                if self._trace_output is not None:
                    try:
                        frame_trace, trace_source_identities = self._build_frame_trace(
                            composed_header,
                            observations, self._output_sequence,
                            compose_start_ns, compose_complete_ns, 0,
                        )
                    except Exception as exc:
                        # Optional provenance diagnostics must not suppress the GUI JPEG.
                        self.get_logger().warning(f"Frame trace build skipped: {type(exc).__name__}")
                if not self._gui_output:
                    self._submit_local(local_image)
                    return
            # JPEG 인코딩 중 자원 잠금을 풀어두고 완료 후 다시 종료 상태를 확인한다.
            # composition_ms와 encode_ms를 나눠 기록하므로 합성 지연과 압축 지연을 구별할 수 있다.
            started = time.perf_counter()
            encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            status["encode_ms"] = (time.perf_counter() - started) * 1000
            jpeg_complete_ns = time.monotonic_ns() if self._trace_output is not None else 0
            if not encoded_ok:
                raise RuntimeError("OpenCV could not encode the composed GUI JPEG.")
            message = CompressedImage()
            message.header.stamp = composed_header_stamp
            message.header.frame_id = self._frame_id
            message.format = "bgr8; jpeg compressed bgr8"
            message.data = array("B", encoded.tobytes())
            if frame_trace is not None:
                frame_trace.jpeg_complete_monotonic_ns = jpeg_complete_ns
                frame_trace.header = deepcopy(message.header)
            with self._resource_lock:
                if self._closed or self._stopping:
                    return
                # 기본 /rail_robot/gui/image/compressed에 최종 JPEG를 발행한다.
                # 이곳에서 네트워크 연결 협상이나 업로드 완료를 기다리지 않는다.
                publish_start_ns = time.monotonic_ns() if frame_trace is not None else 0
                self._image_output.publish(message)
                publish_complete_ns = time.monotonic_ns() if frame_trace is not None else 0
                published = time.monotonic()
                self._published_count += 1
                self._published_meter.record(self._frame_id, self._published_count, published, published)
                if frame_trace is not None:
                    self._publish_frame_trace(frame_trace, trace_source_identities,
                                              publish_start_ns, publish_complete_ns)
                status.update(state="published", jpeg_bytes=len(message.data),
                              published_fps=self._published_meter.fps(published))
                self._publish_status(status)
                self._previous_telemetry = status
                self._last_presentation_error = None
                self._submit_local(local_image)
        except Exception as exc:
            with self._resource_lock:
                if self._closed or self._stopping:
                    return
                self._presentation_error(str(exc))
                if status is not None:
                    status.update(state="error", error=str(exc))
                    try:
                        self._publish_status(status)
                    except Exception as status_exc:
                        self._presentation_error(f"Status publication failed: {status_exc}")
                if local_image is not None:
                    self._submit_local(local_image)
        finally:
            self._render_guard.release()

    def shutdown_requested(self):
        return self._exit_requested

    def prepare_shutdown(self):
        self._stopping = True

    def destroy_node(self):
        with self._resource_lock:
            return self._destroy_locked()

    def _destroy_locked(self):
        if not self._closed:
            self._closed = True
            if self._display is not None:
                try:
                    self._display.close()
                except Exception as exc:
                    self.get_logger().error(f"Preview shutdown failed: {exc}")
        return super().destroy_node()
