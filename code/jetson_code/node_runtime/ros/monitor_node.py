"""Read-only, bounded terminal telemetry; no image, camera or UART ownership."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rosidl_runtime_py.utilities import get_message


TOPICS = {
    "/debug/encoder/diagnostics": "std_msgs/msg/String",
    "/debug/led/state": "std_msgs/msg/String",
    "/debug/ultrasonic/state": "std_msgs/msg/String",
    "/runtime/fault": "std_msgs/msg/String",
    "/mission/state": "rail_robot_interfaces/msg/MissionStatus",
    "/control/decisions": "rail_robot_interfaces/msg/ActuatorDecision",
    "/debug/encoder/state": "rail_robot_interfaces/msg/EncoderState",
    "/debug/encoder/command": "rail_robot_interfaces/msg/EncoderCommand",
    "/debug/front/dc/set": "std_msgs/msg/Bool",
    "/debug/side/dc/set": "std_msgs/msg/Bool",
    "/debug/front/pump/set": "std_msgs/msg/Bool",
    "/debug/side/pump/set": "std_msgs/msg/Bool",
    "/debug/camera/front/fps": "std_msgs/msg/Float64",
    "/debug/camera/side/fps": "std_msgs/msg/Float64",
    "/debug/vision/timing": "rail_robot_interfaces/msg/ModelTiming",
    "/stm/events": "rail_robot_interfaces/msg/StmEvents",
    "/tts/obstacle_state": "std_msgs/msg/Bool",
    "/tts/status": "std_msgs/msg/String",
    "/tts/result": "std_msgs/msg/String",
    "/rail_robot/gui/status": "std_msgs/msg/String",
    "/rail_robot/webrtc/status": "std_msgs/msg/String",
    "/gui/firebase_snapshot/status": "std_msgs/msg/String",
}


def safe_text(value, limit=180):
    """ASCII output prevents escape injection and ambiguous terminal cell widths."""
    value = str(value)[:limit]
    return "".join(char if 32 <= ord(char) < 127 else "?" for char in value)


def number(value):
    return f"{value:.1f}" if isinstance(value, (int, float)) and math.isfinite(value) else "N/A"


def summarize(topic, message):
    """Return a bounded row key/text; never retain ROS messages or image arrays."""
    if topic == "/mission/state":
        text = f"{message.mode} initial={message.initial_count} rescan={message.rescan_count} failed={message.failed} {message.error}"
    elif topic == "/control/decisions":
        topic += "/" + safe_text(message.role, 20)
        text = f"REQUEST cleaner={number(message.cleaner_duty_percent)}% pump={message.pump_on} {message.reason}"
    elif topic == "/debug/encoder/diagnostics":
        data = json.loads(message.data[:8192])
        text = (f"{data.get('reason')} fault={data.get('fault_code')} held={data.get('held')} "
                f"CNT={data.get('left_raw_count')}/{data.get('right_raw_count')} "
                f"PWM={data.get('left_pwm_compare')}/{data.get('right_pwm_compare')} "
                f"historical={data.get('historical')}")
    elif topic == "/debug/encoder/state":
        text = (f"valid={message.feedback_valid} applied={message.applied_direction} {number(message.applied_speed_percent)}% "
                f"counts={message.left_raw_count}/{message.right_raw_count} {message.error}")
    elif topic == "/debug/encoder/command":
        text = f"REQUEST {message.direction} {number(message.speed_percent)}%"
    elif topic == "/debug/ultrasonic/state":
        data = json.loads(message.data[:4096])
        text = (f"valid={data.get('feedback_valid')} {data.get('range_status')} "
                f"distance={data.get('distance_mm')}mm detected={data.get('detected')} held={data.get('held')}")
    elif topic == "/debug/led/state":
        data = json.loads(message.data[:4096])
        text = (f"valid={data.get('feedback_valid')} auto={data.get('auto')} "
                f"requested RGB={data.get('requested_rgb')} obstacle_override={data.get('obstacle_override')}")
    elif topic.endswith("/set"):
        text = "REQUEST " + ("ON" if message.data else "OFF")
    elif topic == "/tts/obstacle_state":
        text = "ULTRASONIC HOLD" if message.data else "ULTRASONIC CLEAR"
    elif topic.endswith("/fps"):
        text = number(message.data) + " camera FPS (reported)"
    elif topic == "/debug/vision/timing":
        topic += "/" + safe_text(message.camera_role, 16) + "/" + safe_text(message.model, 24)
        text = number(message.processing_ms) + " ms"
    elif topic == "/stm/events":
        text = " ".join(safe_text(token, 32) for token in message.tokens[:12])
    elif topic == "/rail_robot/gui/status":
        data = json.loads(message.data[:16384])
        text = (f"{data.get('state')} {number(data.get('published_fps'))} output FPS "
                f"compose={number(data.get('composition_ms'))}ms JPEG={number(data.get('encode_ms'))}ms")
    elif topic == "/rail_robot/webrtc/status":
        data = json.loads(message.data[:16384])
        text = f"{data.get('state')} active={data.get('active')} error={data.get('errorCode', '')} received={data.get('received')}"
    else:
        text = message.data[:180] or "(empty)"
    return topic, safe_text(text)


class MonitorNode(Node):
    def __init__(self, *, clock=time.monotonic, stream=None, node_name="rail_monitor", **kwargs):
        super().__init__(node_name, **kwargs)
        self._monotonic = clock
        self._stream = sys.stdout if stream is None else stream
        self._owns_stream = False
        self._screen_active = False
        # ROS wrappers may pipe stdout even while a controlling terminal exists.
        if stream is None and not self._stream.isatty():
            try:
                self._stream = open('/dev/tty', 'w', encoding='utf-8', buffering=1)
                self._owns_stream = True
            except OSError:
                pass
        self._lock = threading.RLock()
        self._rows = {}
        self._encoder_view = None
        self._encoder_invalid = False
        self._control_view = None
        self._control_invalid = False
        self._listeners = {}
        self._available = set()
        self._nodes = []
        self._graph_error = ""
        self._closed = False
        self._stop_requested = False
        self._hz = self.declare_parameter("refresh_hz", 5.0).value
        self._stale = self.declare_parameter("stale_seconds", 3.0).value
        if not isinstance(self._hz, (int, float)) or not math.isfinite(self._hz) or not 0.2 <= self._hz <= 10:
            raise ValueError("refresh_hz must be between 0.2 and 10")
        if not isinstance(self._stale, (int, float)) or not math.isfinite(self._stale) or not 0.2 <= self._stale <= 3600:
            raise ValueError("stale_seconds must be between 0.2 and 3600")
        self._qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        self.create_timer(2.0, self._discover)
        self.create_timer(1.0 / self._hz, self._render)
        self._discover()

    def _discover(self):
        try:
            graph = dict(self.get_topic_names_and_types())
            nodes = sorted(set(self.get_node_names()))[:64]
            with self._lock:
                self._nodes = nodes
                self._available = set(graph).intersection(TOPICS)
                self._graph_error = ""
            for topic, typename in TOPICS.items():
                if topic not in self._listeners and typename in graph.get(topic, ()):
                    qos = self._qos
                    if topic == "/tts/obstacle_state":
                        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                         reliability=ReliabilityPolicy.RELIABLE,
                                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
                    self._listeners[topic] = self.create_subscription(
                        get_message(typename), topic,
                        lambda message, name=topic: self._receive(name, message), qos)
        except Exception as exc:
            with self._lock:
                self._graph_error = safe_text(exc)

    def _receive(self, topic, message):
        try:
            key, text = summarize(topic, message)
        except (AttributeError, TypeError, ValueError, OverflowError, RecursionError):
            key, text = topic, "INVALID TELEMETRY"
        with self._lock:
            if key not in self._rows and len(self._rows) >= 48:
                del self._rows[min(self._rows, key=lambda item: self._rows[item][1])]
            self._rows[key] = (text, self._monotonic())
            if topic == "/debug/encoder/state" and text == "INVALID TELEMETRY":
                self._encoder_view = None
                self._encoder_invalid = True
                self._rows.pop(topic + "/outputs", None)
            if topic == "/debug/encoder/state" and text != "INVALID TELEMETRY":
                # 최신 샘플 한 개만 복사한다. CNT는 순환값이므로 100ms 토픽 차분으로
                # 속도나 누적 거리를 만들지 않는다(여러 번 순환하면 방향도 잘못 추정됨).
                self._encoder_invalid = False
                self._encoder_view = dict(
                    received=self._monotonic(), valid=message.feedback_valid,
                    direction=safe_text(message.applied_direction, 16), percent=message.applied_speed_percent,
                    left=message.left_raw_count, right=message.right_raw_count,
                    left_modulus=getattr(message, 'left_counter_modulus', 'N/A'),
                    right_modulus=getattr(message, 'right_counter_modulus', 'N/A'),
                    left_pwm=message.left_pwm_compare, right_pwm=message.right_pwm_compare,
                    mcu_millis=getattr(message, 'mcu_millis', 'N/A'), sequence=getattr(message, 'sequence', 'N/A'))
                mask = message.active_output_mask
                outputs = (f"MCU valid={message.feedback_valid} DC(F/S)={int(bool(mask & 1))}/{int(bool(mask & 2))} "
                           f"pump(F/S)={int(bool(mask & 4))}/{int(bool(mask & 8))} "
                           f"PWM={message.left_pwm_compare}/{message.right_pwm_compare}")
                self._rows[topic + "/outputs"] = (safe_text(outputs), self._monotonic())
            if topic == "/debug/encoder/diagnostics":
                if text == "INVALID TELEMETRY":
                    self._control_view = None
                    self._control_invalid = True
                else:
                    payload = json.loads(message.data[:8192])
                    control = payload.get('control_loop')
                    if control is not None:
                        required_numbers = ('sample_sequence', 'sample_period_ms', 'sample_age_ms',
                                            'physical_right_delta', 'physical_left_delta',
                                            'speed_error', 'integral_pwm', 'correction_pwm')
                        if (not isinstance(control, dict)
                                or any(isinstance(control.get(name), bool)
                                       or not isinstance(control.get(name), (int, float))
                                       or not math.isfinite(control[name])
                                       for name in required_numbers)
                                or type(control.get('pi_enabled')) is not bool
                                or type(control.get('valid')) is not bool
                                or type(control.get('applied')) is not bool):
                            self._control_view = None
                            self._control_invalid = True
                        else:
                            self._control_invalid = False
                            control['historical'] = bool(control.get('historical',
                                                                    payload.get('historical', False)))
                            self._control_view = dict(control, received=self._monotonic())

    def lines(self, width=120, height=35):
        width, height = max(1, min(width, 240)), max(1, min(height, 100))
        with self._lock:
            rows, nodes, available, error = dict(self._rows), list(self._nodes), set(self._available), self._graph_error
            encoder, encoder_invalid = self._encoder_view, self._encoder_invalid
            control, control_invalid = self._control_view, self._control_invalid
        now = self._monotonic()
        lines = [f"RAIL MONITOR | read-only | domain {safe_text(os.environ.get('ROS_DOMAIN_ID', '0'), 16)} | Ctrl+C exits"]
        # 80x24 터미널에서도 양쪽 엔코더를 맨 위에 함께 보여준다.
        # 실제 장착 위치는 오른쪽=ENC1/TIM3(left_* 필드), 왼쪽=ENC2/TIM4(right_* 필드)다.
        if encoder is None:
            lines.append("ENCODER INVALID: malformed telemetry" if encoder_invalid else
                         "ENCODER WAIT: start rail_encoder / rail_actuators for live feedback")
        else:
            age = max(0.0, now - encoder['received'])
            state = ('INVALID/HISTORICAL' if not encoder['valid'] else
                     'OLD/HISTORICAL' if age >= min(self._stale, 0.6) else 'LIVE')
            lines.extend([
                f"ENCODER {state} age={age:.1f}s applied={encoder['direction']} {number(encoder['percent'])}%",
                f"RIGHT ENC1/TIM3 CNT={encoder['left']}/{encoder['left_modulus']} PWM={encoder['left_pwm']}",
                f"LEFT  ENC2/TIM4 CNT={encoder['right']}/{encoder['right_modulus']} PWM={encoder['right_pwm']}",
                f"MCU={encoder['mcu_millis']}ms seq={encoder['sequence']} | raw CNT wraps; not RPM/distance",
                "Mounting: RIGHT=ENC1/TIM3, LEFT=ENC2/TIM4.",
            ])
        if control is None:
            if control_invalid:
                lines.append("CONTROL INVALID: malformed controller telemetry")
        else:
            age_ms = max(0.0, float(control['sample_age_ms'])
                         + (now - control['received']) * 1000)
            state = ('INVALID/HISTORICAL' if not control['valid'] else
                     'OLD/HISTORICAL' if control.get('historical') or age_ms >= 600 else
                     'NOT-APPLIED' if not control['applied'] else 'LIVE')
            lines.extend([
                f"CONTROL {state} PI={'ON' if control['pi_enabled'] else 'OFF'} "
                f"dt={control['sample_period_ms']:g}ms sample={control['sample_sequence']} age={age_ms:.0f}ms",
                f"DELTA(dt={control['sample_period_ms']:g}ms) "
                f"RIGHT={control['physical_right_delta']} LEFT={control['physical_left_delta']} "
                f"error={control['speed_error']}",
                f"PI integral={number(control['integral_pwm'])} PWM "
                f"correction={control['correction_pwm']} PWM",
            ])
        lines.extend(["Nodes: " + ", ".join(nodes),
                      "Age = time since received; OLD is historical, WAIT has no sample. REQUEST is not MCU feedback."])
        if error:
            lines.append("Graph: " + error)
        for topic in TOPICS:
            matches = sorted(key for key in rows if key == topic or key.startswith(topic + "/"))
            if not matches:
                lines.append(f"{'WAIT' if topic in available else '---':>4} {topic}: no sample")
            for key in matches:
                text, received = rows[key]
                age = max(0.0, now - received)
                state = "OLD" if age > self._stale else "OK"
                lines.append(f"{state:>4} {age:5.1f}s {key}: {text}")
        if len(lines) > height:
            lines = lines[:height - 1] + [f"... enlarge terminal to show all {len(lines)} rows"]
        return [safe_text(line, width) for line in lines]

    def _render(self):
        if self._closed or self._stop_requested:
            return
        try:
            interactive = self._stream.isatty()
            try:
                size = os.get_terminal_size(self._stream.fileno())
            except (OSError, ValueError, AttributeError):
                size = shutil.get_terminal_size((120, 35))
            # Leave the bottom-right cell untouched to avoid terminal scrolling.
            screen = "\n".join(self.lines(max(1, size.columns - 1), max(1, size.lines - 1)))
            prefix = ""
            if interactive:
                if not self._screen_active:
                    prefix = "\x1b[?1049h\x1b[?25l"
                    self._screen_active = True
                prefix += "\x1b[H\x1b[2J"
            self._stream.write(prefix + screen + ("" if interactive else "\n"))
            self._stream.flush()
        except (BrokenPipeError, OSError):
            self._stop_requested = True

    def shutdown_requested(self):
        return self._stop_requested

    def prepare_shutdown(self):
        self._closed = True
        for timer in self.timers:
            timer.cancel()
        try:
            if self._screen_active:
                self._stream.write("\x1b[?25h\x1b[?1049l")
                self._stream.flush()
                self._screen_active = False
        except (OSError, ValueError):
            pass
        finally:
            if self._owns_stream:
                self._stream.close()
                self._owns_stream = False
