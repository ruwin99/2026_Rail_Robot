"""Bounded DBG1 bench protocol. No ROS imports and no automatic motor start."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import uuid

from .debug_protocol import parse_ready


# bit 4 requests the optional DBG1_CTRL controller snapshot.  Keeping it out of
# DC/pump-only sessions avoids extra UART traffic when encoder data is irrelevant.
CONTROL_TELEMETRY_CAPABILITY = 16
CAPABILITIES = {"motor_encoder": 1 | CONTROL_TELEMETRY_CAPABILITY, "dc_motor": 2,
                "water_pump": 4, "actuators": 7 | CONTROL_TELEMETRY_CAPABILITY}
DIRECTIONS = {"stop": 0, "forward": 1, "reverse": -1}
DIRECTION_NAMES = {value: key for key, value in DIRECTIONS.items()}
CHANNELS = {"front_dc": 0, "side_dc": 1, "front_pump": 2, "side_pump": 3}
DRIVE_FAULT_NAMES = {1: "DRIVE_BAD_CONFIG", 2: "DRIVE_LATE", 3: "DRIVE_ENCODER_RANGE",
                     4: "DRIVE_STALL", 5: "DRIVE_WRONG_DIRECTION"}
DIAGNOSTIC_HINTS = {
    "READY": "구동 요청 대기 또는 정지 상태",
    "RUNNING": "PWM 적용 중. 실제 회전 여부는 두 엔코더 카운터 변화로 확인",
    "ULTRASONIC_HOLD": "초음파 보호정지. 모터 출력이 차단됨",
    "DRIVE_STALL": "PWM 요청 후 엔코더 무변화 감지. 모터 전원/드라이버 활성화/배선과 양쪽 엔코더 입력 확인",
    "DRIVE_WRONG_DIRECTION": "요청 방향과 엔코더 카운트 방향 불일치. A/B 배선 및 방향 설정 확인",
    "DRIVE_LATE": "PID 표본 또는 main 실행 지연. 주기/인터럽트/블로킹 작업 확인",
    "DRIVE_ENCODER_RANGE": "엔코더 카운터 범위 또는 변화량 오류. ARR/분주/신호 확인",
    "DRIVE_BAD_CONFIG": "PID 설정값 오류. 주기/엔코더 모듈러스/게인 설정 확인",
}


@dataclass(frozen=True)
class EncoderFeedback:
    sequence: int
    mcu_millis: int
    left_raw_count: int
    right_raw_count: int
    direction: int
    percent_basis_points: int
    output_mask: int
    left_pwm_compare: int
    right_pwm_compare: int
    received_at: float


def _unsigned(text, maximum=0xFFFFFFFF):
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("Malformed unsigned protocol field.")
    value = int(text)
    if value > maximum:
        raise ValueError("Protocol field exceeds its range.")
    return value


def _signed(text, minimum=-(1 << 31), maximum=(1 << 31) - 1):
    if not text or not text.isascii():
        raise ValueError("Malformed signed protocol field.")
    digits = text[1:] if text.startswith("-") else text
    if not digits or not digits.isdecimal():
        raise ValueError("Malformed signed protocol field.")
    value = int(text)
    if value < minimum or value > maximum:
        raise ValueError("Protocol field exceeds its range.")
    return value


class ActuatorDebugSession:
    """One UART session; each input expires independently of protocol heartbeats."""

    HEARTBEAT_SECONDS = 0.2
    FEEDBACK_TIMEOUT_SECONDS = 0.6
    HANDSHAKE_TIMEOUT_SECONDS = 2.0
    REVERSE_STOP_SECONDS = 0.1

    def __init__(self, uart, capability="actuators", input_timeout_seconds=1.0,
                 *, clock=time.monotonic, session_id=None):
        if capability not in CAPABILITIES:
            raise ValueError("Unknown debug_capability.")
        if not math.isfinite(input_timeout_seconds) or not 0.2 <= input_timeout_seconds <= 5.0:
            raise ValueError("input_timeout_seconds must be finite, between 0.2 and 5 seconds.")
        self.uart, self.clock = uart, clock
        self.capability = CAPABILITIES[capability]
        self.input_timeout_seconds = input_timeout_seconds
        self.session_id = uuid.uuid4().hex if session_id is None else session_id
        if len(self.session_id) != 32 or any(c not in "0123456789abcdef" for c in self.session_id):
            raise ValueError("Debug session must be 32 lowercase hexadecimal digits.")
        self.ready = self.closed = self.failed = False
        self.error = ""
        self.feedback = None
        self.diagnostics = None
        self.control_telemetry = None
        self.last_nonzero_pwm = None
        self.sequence = 0
        self.registered_drive_pwm = 0
        self.right_trim_percent = self.left_trim_percent = 0.0
        self.registered_dc_pwm = self.registered_pump_pwm = 0
        self.left_counter_modulus = self.right_counter_modulus = 0
        self.commanded_direction = self.commanded_percent = 0
        self._drive = (0, 0, float("-inf"))
        self._channels = {name: (False, float("-inf")) for name in CHANNELS}
        self._buffer = bytearray()
        self._started = self.clock()
        self._next_send = self._started
        self._last_drive_direction = 0
        self._stop_sequence = None
        self._stop_sent_at = None
        self._write(f"DBG1_HELLO {self.session_id} {self.capability}")

    def _write(self, line):
        data = (line + "\r\n").encode("ascii")
        if self.uart.write(data) != len(data):
            raise RuntimeError("Incomplete debug UART write.")

    def _require_ready(self):
        if self.failed or self.closed or not self.ready:
            raise ValueError("Debug firmware handshake is not ready.")

    def command_encoder(self, direction, speed_percent):
        self._require_ready()
        if not self.capability & 1:
            raise ValueError("Encoder commands are disabled in this debug mode.")
        if direction not in DIRECTIONS:
            raise ValueError("Encoder direction must be forward, reverse, or stop.")
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, (int, float)):
            raise ValueError("speed_percent must be a number.")
        if not math.isfinite(speed_percent) or not 0 <= speed_percent <= 100:
            raise ValueError("speed_percent must be finite and in [0, 100].")
        if direction == "stop" and speed_percent != 0:
            raise ValueError("A stop command requires speed_percent=0.")
        value = int(round(speed_percent * 100))
        if value and self.diagnostics is not None and self.diagnostics['held']:
            return False
        self._drive = (DIRECTIONS[direction] if value else 0, value, self.clock())
        self._next_send = self.clock()
        return True

    def command_channel(self, channel, enabled):
        self._require_ready()
        if channel not in CHANNELS or type(enabled) is not bool:
            raise ValueError("An actuator input must be a known channel and a Boolean.")
        needed = 2 if channel.endswith("_dc") else 4
        if not self.capability & needed:
            raise ValueError("This actuator channel is disabled in this debug mode.")
        if enabled and self.diagnostics is not None and self.diagnostics['held']:
            return False
        self._channels[channel] = (enabled, self.clock())
        self._next_send = self.clock()
        return True

    def _snapshot(self, now):
        direction, percent, received = self._drive
        if now - received >= self.input_timeout_seconds:
            direction = percent = 0
        mask = sum(1 << CHANNELS[name] for name, (on, received) in self._channels.items()
                   if on and now - received < self.input_timeout_seconds)
        self.commanded_direction, self.commanded_percent = direction, percent
        return direction, percent, mask

    def _send_snapshot(self, direction, percent, mask, now):
        if self.sequence == 0xFFFFFFFF:
            raise RuntimeError("Debug sequence exhausted; restart the debug session.")
        self.sequence += 1
        self._write(f"DBG1_SET {self.session_id} {self.sequence} {direction} {percent} {mask}")
        self._next_send = now + self.HEARTBEAT_SECONDS

    def _receive(self, line, now):
        fields = line.split()
        if len(fields) < 2 or fields[1] != self.session_id:
            return
        if fields[0] in ("DBG1_REJECT", "DBG1_ERROR", "DBG1_TIMEOUT"):
            reason = " ".join(fields[2:] or fields[:1])
            hint = DIAGNOSTIC_HINTS.get(fields[2] if len(fields) > 2 else "", "")
            raise RuntimeError("STM debug rejected/stopped: " + reason +
                               ("; " + hint if hint else "") + self.diagnostic_summary())
        if fields[0] == "DBG1_READY":
            if self.ready:
                raise ValueError("Unexpected DBG1_READY frame.")
            values, self.right_trim_percent = parse_ready(fields, self.capability)
            (_, self.registered_drive_pwm, _,
             self.registered_dc_pwm, self.registered_pump_pwm,
             self.left_counter_modulus, self.right_counter_modulus) = values
            # encoder 1은 실제 오른쪽 모터다. READY trim은 전진 기준이며 후진 값은 포함하지 않는다.
            # left_trim_percent는 실제 왼쪽 보정값이 아닌 기존 소비자 호환 별칭이다.
            self.left_trim_percent = self.right_trim_percent
            self.ready = True
            self._next_send = now
        elif fields[0] == "DBG1_DIAG":
            self._receive_diagnostics(fields, now)
        elif fields[0] == "DBG1_CTRL":
            self._receive_control(fields, now)
        elif fields[0] == "DBG1_STATE":
            if not self.ready or len(fields) != 11 or fields[6] not in ("-1", "0", "1"):
                raise ValueError("Malformed or premature debug feedback.")
            sequence, tick, left, right = [_unsigned(value) for value in fields[2:6]]
            direction = int(fields[6])
            percent = _unsigned(fields[7], 10000)
            mask = _unsigned(fields[8], 15)
            pwm_left, pwm_right = [_unsigned(value, 65535) for value in fields[9:11]]
            if (sequence > self.sequence or left >= self.left_counter_modulus
                    or right >= self.right_counter_modulus or (direction == 0 and percent != 0)
                    or (direction != 0 and not self.capability & 1)
                    or (mask & 3 and not self.capability & 2)
                    or (mask & 12 and not self.capability & 4)):
                raise ValueError("Debug feedback violates negotiated limits.")
            if self.feedback is not None:
                if sequence < self.feedback.sequence:
                    return
                delta = (tick - self.feedback.mcu_millis) % (1 << 32)
                if delta == 0 or delta >= (1 << 31):
                    return
            self.feedback = EncoderFeedback(sequence, tick, left, right, direction,
                                            percent, mask, pwm_left, pwm_right, now)
            if pwm_left or pwm_right:
                self.last_nonzero_pwm = (pwm_left, pwm_right)
        else:
            raise ValueError("Unsupported debug protocol response.")

    def _receive_control(self, fields, now):
        # Optional controller telemetry is negotiated separately so firmware can
        # remain compatible with older debug clients which reject unknown frames.
        if (not self.ready or not self.capability & CONTROL_TELEMETRY_CAPABILITY
                or len(fields) != 15 or fields[6] not in ("-1", "0", "1")):
            raise ValueError("Malformed or premature debug control telemetry.")
        command_sequence, sample_sequence, sample_tick = (
            _unsigned(value) for value in fields[2:5])
        elapsed_ms = _unsigned(fields[5], 1000)
        direction = int(fields[6])
        flags = _unsigned(fields[7], 7)
        values = [_signed(value) for value in fields[8:15]]
        valid, applied = bool(flags & 2), bool(flags & 4)
        if (command_sequence > self.sequence or valid != applied
                or (valid and elapsed_ms == 0)):
            raise ValueError("Debug control telemetry violates negotiated limits.")
        previous = self.control_telemetry
        if previous is not None:
            sequence_delta = (sample_sequence - previous['sample_sequence']) % (1 << 32)
            tick_delta = (sample_tick - previous['sample_millis']) % (1 << 32)
            # STOP/새 방향은 계산을 다시 하지 않고 같은 sample sequence의
            # valid/applied만 지운다. 이 무효화 프레임은 replay로 버리면 안 된다.
            invalidation = (sequence_delta == 0 and not valid and not applied
                            and command_sequence >= previous['command_sequence'])
            if (sequence_delta == 0 and not invalidation) or (sequence_delta != 0 and (
                    sequence_delta >= (1 << 31) or tick_delta >= (1 << 31))):
                return
        (right_raw, left_raw, right_normalized, left_normalized,
         speed_error, integral_q8, correction) = values
        self.control_telemetry = dict(
            command_sequence=command_sequence, sample_sequence=sample_sequence,
            sample_millis=sample_tick, elapsed_ms=elapsed_ms, direction=direction,
            pi_enabled=bool(flags & 1), valid=valid, applied=applied,
            right_raw_delta=right_raw, left_raw_delta=left_raw,
            right_normalized_delta=right_normalized,
            left_normalized_delta=left_normalized, speed_error=speed_error,
            integral_q8=integral_q8, integral_pwm=integral_q8 / 256.0,
            correction_pwm=correction, received_at=now)

    def _receive_diagnostics(self, fields, now):
        # 진단은 STATE와 별도로 검증한다. DIAG만 들어와도 구동 피드백의 만료 기한을
        # 갱신하지 않는다. 잘못된 상태 보고로 통신 deadman을 우회하지 못하게 한다.
        if not self.ready or len(fields) != 12:
            raise ValueError("Malformed or premature debug diagnostics.")
        seq, tick = [_unsigned(value) for value in fields[2:4]]
        reason = fields[4]
        fault = _unsigned(fields[5], 5)
        held, detected = [_unsigned(value, 1) for value in fields[6:8]]
        left, right = [_unsigned(value) for value in fields[8:10]]
        pwm_left, pwm_right = [_unsigned(value, 65535) for value in fields[10:12]]
        # 범위 오류 진단에서는 잘못된 raw CNT 자체를 보존해야 한다. 정상 STATE와 달리
        # negotiated modulus로 진단 값을 버리지 않는다(여전히 uint32만 허용).
        if (seq > self.sequence or reason not in DIAGNOSTIC_HINTS
                or (fault and DRIVE_FAULT_NAMES[fault] != reason)
                or (not fault and reason.startswith("DRIVE_"))
                or (reason == "ULTRASONIC_HOLD" and not held)):
            raise ValueError("Debug diagnostics violate negotiated limits.")
        previous = self.diagnostics
        if previous is not None:
            delta = (tick - previous['mcu_millis']) % (1 << 32)
            if seq < previous['sequence'] or delta >= (1 << 31):
                return
            if delta == 0 and seq == previous['sequence'] and fault == previous['fault_code']:
                return
        self.diagnostics = dict(sequence=seq, mcu_millis=tick, reason=reason, fault_code=fault,
                                held=bool(held), detected=bool(detected), left_raw_count=left,
                                right_raw_count=right, left_pwm_compare=pwm_left,
                                right_pwm_compare=pwm_right, received_at=now)
        if held:
            # HOLD 전/중 요청을 버린다. 해제 뒤 새 ROS 요청 없이 heartbeat가 옛 ON을
            # 재전송하지 않도록 모터와 네 청소 출력의 입력 lease를 함께 지운다.
            self._drive = (0, 0, float('-inf'))
            self._channels = {name: (False, float('-inf')) for name in CHANNELS}
            if previous is None or not previous['held']:
                self._next_send = now

    def diagnostic_summary(self):
        data = self.diagnostics
        if data is None:
            return ""
        summary = (f"; MCU {data['reason']} fault={data['fault_code']}"
                   f" counts={data['left_raw_count']}/{data['right_raw_count']}"
                   f" PWM={data['left_pwm_compare']}/{data['right_pwm_compare']}"
                   f" held={int(data['held'])}"
                   f" diagnostic_age_ms={max(0, (self.clock() - data['received_at']) * 1000):.0f}")
        if self.last_nonzero_pwm is not None:
            summary += f" last_nonzero_PWM={self.last_nonzero_pwm[0]}/{self.last_nonzero_pwm[1]} (historical)"
        return summary

    def tick(self):
        if self.closed or self.failed:
            return
        now = self.clock()
        try:
            # Late queued feedback cannot revive a session whose local deadline
            # elapsed while the executor was unavailable.
            if self.feedback is not None and now - self.feedback.received_at >= self.FEEDBACK_TIMEOUT_SECONDS:
                raise RuntimeError("STM debug feedback timed out.")
            if not self.ready and now - self._started >= self.HANDSHAKE_TIMEOUT_SECONDS:
                raise RuntimeError("STM does not acknowledge DBG1; compatible firmware is required.")
            waiting = self.uart.in_waiting
            if waiting:
                self._buffer.extend(self.uart.read(min(waiting, 4096)))
            if len(self._buffer) > 8192:
                raise ValueError("Debug receive buffer overflow.")
            while b"\n" in self._buffer:
                line, _, rest = self._buffer.partition(b"\n")
                self._buffer = bytearray(rest)
                self._receive(line.decode("ascii").strip(), now)
            if not self.ready:
                if now - self._started >= self.HANDSHAKE_TIMEOUT_SECONDS:
                    raise RuntimeError("STM does not acknowledge DBG1; compatible firmware is required.")
                return
            if self.feedback is None:
                if now - self._started >= self.HANDSHAKE_TIMEOUT_SECONDS:
                    raise RuntimeError("STM debug feedback was not received.")
                return
            if now - self.feedback.received_at >= self.FEEDBACK_TIMEOUT_SECONDS:
                raise RuntimeError("STM debug feedback timed out.")
            if now < self._next_send:
                return
            direction, percent, mask = self._snapshot(now)
            reversing = direction and self._last_drive_direction and direction != self._last_drive_direction
            if reversing:
                if self._stop_sequence is None:
                    self._send_snapshot(0, 0, mask, now)
                    self._stop_sequence, self._stop_sent_at = self.sequence, now
                    return
                if (self.feedback.sequence < self._stop_sequence or self.feedback.direction != 0
                        or now - self._stop_sent_at < self.REVERSE_STOP_SECONDS):
                    self._send_snapshot(0, 0, mask, now)
                    return
            self._send_snapshot(direction, percent, mask, now)
            if direction:
                self._last_drive_direction = direction
                self._stop_sequence = self._stop_sent_at = None
            elif self._stop_sequence is None:
                self._stop_sequence, self._stop_sent_at = self.sequence, now
        except Exception as exc:
            self.fail(str(exc))

    def fail(self, reason):
        self.failed = True
        self.error = reason + (self.diagnostic_summary() if "; MCU " not in reason else "")
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.commanded_direction = self.commanded_percent = 0
        # CLOSE atomically stops drive plus all four outputs in the firmware.
        # A single frame also avoids overrunning its one-command mailbox.
        self._write(f"DBG1_CLOSE {self.session_id}")


RR2_CAPABILITIES = {
    "motor_encoder": "ACTUATOR_DEBUG_ENCODER",
    "dc_motor": "ACTUATOR_DEBUG_DC",
    "water_pump": "ACTUATOR_DEBUG_PUMP",
    "actuators": "ACTUATOR_DEBUG_ALL",
}


class Rr2ActuatorDebugSession:
    """ROS-side debug state; SerialWorker alone owns the UART."""

    HEARTBEAT_SECONDS = 0.2
    FEEDBACK_TIMEOUT_SECONDS = 0.6
    REVERSE_STOP_SECONDS = 0.1

    def __init__(self, worker, capability="actuators", input_timeout_seconds=1.0,
                 *, clock=time.monotonic):
        if capability not in RR2_CAPABILITIES:
            raise ValueError("Unknown debug_capability.")
        if not math.isfinite(input_timeout_seconds) or not 0.2 <= input_timeout_seconds <= 5.0:
            raise ValueError("input_timeout_seconds must be finite, between 0.2 and 5 seconds.")
        self.worker, self.clock = worker, clock
        self.session_id = worker.session_id
        self.capability = CAPABILITIES[capability]
        self.input_timeout_seconds = input_timeout_seconds
        self.closed = self.failed = False
        self.error = ""
        self.feedback = self.diagnostics = self.control_telemetry = None
        self.last_nonzero_pwm = None
        # RR2 READY does not carry DBG1 calibration metadata. Zero is the
        # EncoderState message's unknown modulus sentinel; JSON uses null.
        self.registered_drive_pwm = self.right_trim_percent = self.left_trim_percent = None
        self.registered_dc_pwm = self.registered_pump_pwm = None
        self.left_counter_modulus = self.right_counter_modulus = 0
        self.commanded_direction = self.commanded_percent = 0
        self.sequence = 0
        self._drive = (0, 0, float("-inf"))
        self._channels = {name: (False, float("-inf")) for name in CHANNELS}
        self._started = clock()
        self._next_send = self._started
        self._pending = None
        self._last_drive_direction = 0
        self._reverse_stop_at = None

    @property
    def ready(self):
        return self.worker.ready and not self.closed and not self.failed

    def _require_ready(self):
        if not self.ready:
            raise ValueError("RR2 debug session is not ready.")

    def command_encoder(self, direction, speed_percent):
        self._require_ready()
        if not self.capability & 1:
            raise ValueError("Encoder commands are disabled in this debug mode.")
        if direction not in DIRECTIONS:
            raise ValueError("Encoder direction must be forward, reverse, or stop.")
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, (int, float)):
            raise ValueError("speed_percent must be a number.")
        if not math.isfinite(speed_percent) or not 0 <= speed_percent <= 100:
            raise ValueError("speed_percent must be finite and in [0, 100].")
        if direction == "stop" and speed_percent != 0:
            raise ValueError("A stop command requires speed_percent=0.")
        value = int(round(speed_percent * 100))
        if value and self.diagnostics is not None and self.diagnostics['held']:
            return False
        self._drive = (DIRECTIONS[direction] if value else 0, value, self.clock())
        self._next_send = self.clock()
        return True

    def command_channel(self, channel, enabled):
        self._require_ready()
        if channel not in CHANNELS or type(enabled) is not bool:
            raise ValueError("An actuator input must be a known channel and a Boolean.")
        needed = 2 if channel.endswith("_dc") else 4
        if not self.capability & needed:
            raise ValueError("This actuator channel is disabled in this debug mode.")
        if enabled and self.diagnostics is not None and self.diagnostics['held']:
            return False
        self._channels[channel] = (enabled, self.clock())
        self._next_send = self.clock()
        return True

    def _snapshot(self, now):
        direction, percent, received = self._drive
        if now - received >= self.input_timeout_seconds:
            direction = percent = 0
        mask = sum(1 << CHANNELS[name] for name, (on, received) in self._channels.items()
                   if on and now - received < self.input_timeout_seconds)
        if self.diagnostics is not None and self.diagnostics['held']:
            direction = percent = mask = 0
        self.commanded_direction, self.commanded_percent = direction, percent
        return direction, percent, mask

    def _status(self, result, now):
        fields = result.payload
        if not fields:
            raise ValueError("Empty RR2 debug STATUS")
        key = fields[0]
        if key == "ENCODER":
            if len(fields) != 9 or fields[4] not in ("-1", "0", "1"):
                raise ValueError("Malformed RR2 ENCODER status")
            tick, left, right = (_unsigned(field) for field in fields[1:4])
            direction = int(fields[4])
            percent = _unsigned(fields[5], 10000)
            mask = _unsigned(fields[6], 15)
            pwm_left, pwm_right = (_unsigned(field, 65535) for field in fields[7:9])
            if (direction == 0 and percent or
                    direction != 0 and not self.capability & 1 or
                    mask & 3 and not self.capability & 2 or
                    mask & 12 and not self.capability & 4):
                raise ValueError("RR2 ENCODER status violates capability")
            previous = self.feedback
            if previous is not None:
                delta = (tick - previous.mcu_millis) & 0xffffffff
                if delta == 0 or delta >= 0x80000000:
                    return
            self.feedback = EncoderFeedback(result.sequence, tick, left, right,
                                            direction, percent, mask, pwm_left, pwm_right, now)
            if pwm_left or pwm_right:
                self.last_nonzero_pwm = (pwm_left, pwm_right)
        elif key == "DIAG":
            if len(fields) != 10:
                raise ValueError("Malformed RR2 DIAG status")
            tick = _unsigned(fields[1])
            reason = fields[2]
            fault = _unsigned(fields[3], 5)
            held, detected = (_unsigned(field, 1) for field in fields[4:6])
            left, right = (_unsigned(field) for field in fields[6:8])
            pwm_left, pwm_right = (_unsigned(field, 65535) for field in fields[8:10])
            if (reason not in DIAGNOSTIC_HINTS or
                    fault and DRIVE_FAULT_NAMES[fault] != reason or
                    not fault and reason.startswith("DRIVE_") or
                    reason == "ULTRASONIC_HOLD" and not held):
                raise ValueError("RR2 DIAG status violates limits")
            previous = self.diagnostics
            if previous is not None:
                delta = (tick - previous['mcu_millis']) & 0xffffffff
                if delta == 0 or delta >= 0x80000000:
                    return
            self.diagnostics = dict(sequence=result.sequence, mcu_millis=tick,
                                    reason=reason, fault_code=fault, held=bool(held),
                                    detected=bool(detected), left_raw_count=left,
                                    right_raw_count=right, left_pwm_compare=pwm_left,
                                    right_pwm_compare=pwm_right, received_at=now)
            if held:
                self._drive = (0, 0, float("-inf"))
                self._channels = {name: (False, float("-inf")) for name in CHANNELS}
                self._next_send = now
        elif key == "CTRL":
            if len(fields) != 7 or fields[3] not in ("-1", "0", "1"):
                raise ValueError("Malformed RR2 CTRL status")
            sample = _unsigned(fields[1])
            elapsed = _unsigned(fields[2], 1000)
            flags = _unsigned(fields[4], 7)
            error = _signed(fields[5])
            correction = _signed(fields[6])
            previous = self.control_telemetry
            if previous is not None and sample <= previous['sample_sequence'] and flags & 2:
                return
            self.control_telemetry = dict(
                sample_sequence=sample, elapsed_ms=elapsed, direction=int(fields[3]),
                pi_enabled=bool(flags & 1), valid=bool(flags & 2), applied=bool(flags & 4),
                speed_error=error, correction_pwm=correction, received_at=now)
        # Unknown STATUS keys can be mission/system telemetry on shared firmware.

    def diagnostic_summary(self):
        data = self.diagnostics
        if data is None:
            return ""
        return (f"; MCU {data['reason']} fault={data['fault_code']}"
                f" counts={data['left_raw_count']}/{data['right_raw_count']}"
                f" PWM={data['left_pwm_compare']}/{data['right_pwm_compare']}"
                f" held={int(data['held'])}"
                f" diagnostic_age_ms={max(0, (self.clock() - data['received_at']) * 1000):.0f}")

    def tick(self):
        if self.closed or self.failed:
            return
        now = self.clock()
        try:
            if not self.worker.ready:
                if self.feedback is None and not self.worker.startup_error and now - self._started < 2.0:
                    return
                raise RuntimeError(self.worker.startup_error or "RR2 debug session lost")
            for result in self.worker.drain_results(32):
                if result.kind == "STATUS":
                    self._status(result, now)
                elif result.kind == "EVENT" and result.payload and result.payload[0] in ("DEBUG_FAULT", "DEBUG_TIMEOUT"):
                    raise RuntimeError("STM debug event: " + " ".join(result.payload))
            if self.feedback is None and now - self._started >= 2.0:
                raise RuntimeError("RR2 debug feedback was not received")
            if self.feedback is not None and now - self.feedback.received_at >= self.FEEDBACK_TIMEOUT_SECONDS:
                raise RuntimeError("RR2 debug feedback timed out")
            if self._pending is not None:
                if not self._pending.done():
                    return
                outcome = self._pending.result()
                self._pending = None
                if not outcome.success:
                    raise RuntimeError("RR2 SET failed: " + outcome.error)
                self.sequence = outcome.sequence or self.sequence
            if self.feedback is None or now < self._next_send:
                return
            direction, percent, mask = self._snapshot(now)
            if direction and self._last_drive_direction and direction != self._last_drive_direction:
                if self._reverse_stop_at is None:
                    self._reverse_stop_at = now
                if (now - self._reverse_stop_at < self.REVERSE_STOP_SECONDS or
                        self.feedback.direction != 0 or
                        self.feedback.received_at < self._reverse_stop_at):
                    direction = percent = 0
                else:
                    self._reverse_stop_at = None
            args = (str(direction), str(percent), str(mask))
            self._pending = self.worker.submit("SET", args, coalesce_key="debug_snapshot")
            self._next_send = now + self.HEARTBEAT_SECONDS
            if direction:
                self._last_drive_direction = direction
            if self._pending.done() and not self._pending.result().success:
                raise RuntimeError("RR2 SET rejected: " + self._pending.result().error)
        except (RuntimeError, ValueError, OSError) as exc:
            self.fail(str(exc))

    def fail(self, reason):
        if self.failed:
            return
        self.failed = True
        self.error = reason + (self.diagnostic_summary() if "; MCU " not in reason else "")
        self.commanded_direction = self.commanded_percent = 0
        self.worker.latch_fault()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.commanded_direction = self.commanded_percent = 0
        self.worker.submit("STOP", force=True)
