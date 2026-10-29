"""Single-owner, bounded RR2 serial transport for ROS STM owners.

Only the worker thread opens, reads, writes, and closes the serial port. ROS
callbacks submit immutable commands and consume bounded validated results.
RR2 CMD and PING have independent STM sequence/deduplication lanes, allowing
heartbeats during a bounded CMD acknowledgement window. RX, EVENT_ACK, and
STATUS continue to be serviced throughout that wait.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import threading
import time
import uuid

from inspection_report import DEFAULT_RAIL_SECTION_TARGET

from .rr2_protocol import Rr2Frame, Rr2StreamParser, encode_rr2


_SAFE_OFF = (
    "FRONT_CLEANER_OFF", "FRONT_PUMP_OFF",
    "SIDE_CLEANER_OFF", "SIDE_PUMP_OFF",
)
_MISSION_COMMANDS = frozenset({"PREPARE", "CANCEL_PREPARE", "START", "CAPTURE_OK"})
_ACTUATOR_COMMANDS = frozenset({
    "FRONT_CLEANER_PWM_12_5", "FRONT_CLEANER_PWM_25", "FRONT_CLEANER_OFF",
    "FRONT_PUMP_ON", "FRONT_PUMP_OFF",
    "SIDE_CLEANER_PWM_12_5", "SIDE_CLEANER_PWM_25", "SIDE_CLEANER_OFF",
    "SIDE_PUMP_ON", "SIDE_PUMP_OFF",
})
_ALLOWED_COMMANDS = _MISSION_COMMANDS | _ACTUATOR_COMMANDS
_ON_COMMANDS = frozenset(command for command in _ACTUATOR_COMMANDS
                          if command.endswith("_ON") or "_PWM_" in command)
_DEBUG_MASKS = {
    "ACTUATOR_DEBUG_ENCODER": 0,
    "ACTUATOR_DEBUG_DC": 3,
    "ACTUATOR_DEBUG_PUMP": 12,
    "ACTUATOR_DEBUG_ALL": 15,
}
_CAPABILITIES = frozenset({"MISSION", "SENSOR_DEBUG", *_DEBUG_MASKS})
_LED_MODES = frozenset({"auto", "manual", "breathe", "blink"})


@dataclass(frozen=True)
class CommandResult:
    success: bool
    session_id: str
    sequence: int | None
    error: str = ""
    state: str = ""


@dataclass(frozen=True)
class SerialResult:
    kind: str
    sequence: int
    payload: tuple[str, ...]
    session_id: str


@dataclass
class _CommandGroup:
    opcodes: tuple[str, ...]
    futures: list[Future]
    args: tuple[str, ...] = ()
    coalesce_key: str | None = None
    safety: bool = False
    index: int = 0
    last_sequence: int | None = None


@dataclass
class _Pending:
    group: _CommandGroup | None
    opcode: str
    sequence: int
    wire: bytes
    attempts: int
    deadline: float
    ping: bool = False
    cancelled: bool = False


class SerialWorker:
    """One RR2 capability session with bounded queues and a reserved stop slot."""

    def __init__(
        self,
        serial_factory,
        *,
        serial_kwargs: dict,
        capability: str = "MISSION",
        session: str | None = None,
        queue_capacity: int = 32,
        result_capacity: int = 64,
        ack_timeout: float = 0.12,
        retry_limit: int = 2,
        heartbeat_interval: float = 0.1,
        heartbeat_timeout: float = 0.6,
        poll_interval: float = 0.002,
        hello_timeout: float = 0.15,
        hello_attempts: int = 3,
    ) -> None:
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        if type(result_capacity) is not int or result_capacity < 1:
            raise ValueError("result_capacity must be a positive integer")
        if ack_timeout <= 0 or retry_limit < 1 or poll_interval <= 0:
            raise ValueError("RR2 timing and retry limits must be positive")
        # The RR2 MCU watchdog is 600 ms. Leave margin for scheduling and UART
        # writes even when every command retry is used.
        if ack_timeout * retry_limit >= 0.35:
            raise ValueError("CMD acknowledgement window must remain below 350 ms")
        if heartbeat_interval <= 0 or heartbeat_timeout <= heartbeat_interval:
            raise ValueError("Invalid heartbeat timing")
        if hello_timeout <= 0 or hello_attempts < 1:
            raise ValueError("Invalid HELLO retry limits")
        if capability not in _CAPABILITIES:
            raise ValueError("Unsupported RR2 capability")
        self.capability = capability
        self._stop_opcode = "SAFE_OFF" if capability == "MISSION" else "STOP"
        self._stop_commands = _SAFE_OFF if capability == "MISSION" else ("STOP",)
        self.session_id = session or uuid.uuid4().hex
        if (len(self.session_id) != 32 or
                any(char not in "0123456789abcdef" for char in self.session_id)):
            raise ValueError("RR2 session must be 32 lowercase hex characters")
        self._factory = serial_factory
        self._serial_kwargs = dict(serial_kwargs)
        self._queue_capacity = queue_capacity
        self._result_capacity = result_capacity
        self._ack_timeout = ack_timeout
        self._retry_limit = retry_limit
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._poll_interval = poll_interval
        self._hello_timeout = hello_timeout
        self._hello_attempts = hello_attempts

        self._condition = threading.Condition()
        self._normal: deque[_CommandGroup] = deque()
        self._safety: _CommandGroup | None = None
        self._active: _Pending | None = None
        self._emergency_stop_future: Future[CommandResult] | None = None
        self._results: deque[SerialResult] = deque()
        self._ready_event = threading.Event()
        self._stopped_event = threading.Event()
        self._stop_requested = False
        self._closing = False
        self._off_barrier = False
        self._safe_off_acknowledged = False
        self._fault_latched = False
        self._heartbeats_suspended = False
        self._heartbeat_tx_lock = threading.Lock()
        self._command_tx_lock = threading.Lock()
        self._failed = False
        self._failure = ""
        self._startup_error = ""
        self._parser = Rr2StreamParser()
        self._cmd_sequence = 0
        self._ping_sequence_counter = 0
        self._ping_sequence: int | None = None
        self._ping_wire: bytes | None = None
        self._ping_attempts = 0
        self._ping_deadline = 0.0
        self._last_ack = 0.0
        self._last_ping = 0.0
        self._last_event_sequence = 0
        self._last_status_sequence = 0
        self._seen_events: set[int] = set()
        self._thread = threading.Thread(
            target=self._run, name="rail-rr2-serial-owner", daemon=True
        )
        self._thread.start()

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set() and not self._failed and not self._closing

    @property
    def startup_error(self) -> str:
        return self._startup_error

    def wait_ready(self, timeout_seconds: float) -> bool:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        self._ready_event.wait(timeout_seconds)
        return self.ready

    def submit(
        self,
        opcode: str,
        args: tuple[str, ...] = (),
        *,
        coalesce_key: str | None = None,
        force: bool = False,
    ) -> Future[CommandResult]:
        """Queue one RR2 action; force reissues an acknowledged safety stop."""
        future: Future = Future()
        if type(force) is not bool:
            return self._completed(future, "force must be a boolean")
        if force and opcode != self._stop_opcode:
            return self._completed(future, f"force is supported only for {self._stop_opcode}")
        if not isinstance(args, tuple) or any(type(arg) is not str for arg in args):
            return self._completed(future, "RR2 command arguments must be a tuple of strings")
        error = self._validate_command(opcode, args)
        if error:
            return self._completed(future, error)
        with self._condition:
            if self._closing or self._stop_requested:
                return self._completed(future, "Serial worker is stopping")
            if opcode == self._stop_opcode and self._emergency_stop_future is not None:
                return self._emergency_stop_future
            if self._fault_latched and opcode != self._stop_opcode:
                return self._completed(future, f"RR2 worker fault is latched; only {self._stop_opcode} is allowed")
            if self._failed or not self._ready_event.is_set():
                reason = self._failure or self._startup_error or "RR2 session is not READY"
                return self._completed(future, reason)
            if opcode == self._stop_opcode:
                if self._safety is not None:
                    # Repeated mission-state OFF requests share the one
                    # reserved transaction; waiter memory stays constant.
                    return self._safety.futures[0]
                if self._safe_off_acknowledged and not force:
                    future.set_result(CommandResult(
                        True, self.session_id, None, state="OFF"
                    ))
                    return future
                self._off_barrier = True
                self._safe_off_acknowledged = False
                self._reject_queued_on_locked(f"Superseded by {self._stop_opcode}")
                self._safety = _CommandGroup(self._stop_commands, [future], safety=True)
                self._condition.notify()
                return future
            if self._is_on(opcode, args) and self._off_barrier:
                return self._completed(future, f"ON rejected until {self._stop_opcode} is acknowledged")
            if coalesce_key:
                for group in self._normal:
                    if group.coalesce_key == coalesce_key:
                        self._complete_group_locked(
                            group, False, "Superseded by a newer actuator state",
                            group.last_sequence,
                        )
                        group.opcodes = (opcode,)
                        group.futures = [future]
                        group.args = args
                        if self._is_on(opcode, args):
                            self._safe_off_acknowledged = False
                        self._condition.notify()
                        return future
            if len(self._normal) >= self._queue_capacity:
                return self._completed(future, "RR2 command queue is full")
            self._normal.append(_CommandGroup((opcode,), [future], args, coalesce_key))
            if self._is_on(opcode, args):
                self._safe_off_acknowledged = False
            self._condition.notify()
        return future

    def _validate_command(self, opcode: str, args: tuple[str, ...]) -> str:
        if self.capability == "MISSION":
            # PREPARE만 다음 임무의 단계별 구간 수(한 자리 1..기본 구간 수) 인자 하나를 받는다.
            if opcode == "PREPARE" and len(args) == 1:
                allowed = tuple(str(n) for n in range(1, DEFAULT_RAIL_SECTION_TARGET + 1))
                return ("" if args[0] in allowed
                        else f"RR2 PREPARE section count must be 1..{DEFAULT_RAIL_SECTION_TARGET}")
            if args:
                return "RR2 mission commands do not accept arguments"
            if opcode not in _ALLOWED_COMMANDS and opcode != "SAFE_OFF":
                return f"Unsupported RR2 mission command: {opcode}"
            return ""
        if opcode == "STOP":
            return "" if not args else "RR2 STOP does not accept arguments"
        if self.capability == "SENSOR_DEBUG":
            if opcode != "LED":
                return f"RR2 SENSOR_DEBUG cannot send {opcode}"
            if len(args) != 4 or args[0] not in _LED_MODES:
                return "RR2 LED requires mode and three RGB bytes"
            if any(not self._decimal(arg, 255) for arg in args[1:]):
                return "RR2 LED RGB must be 0..255"
            return ""
        if opcode != "SET":
            return f"RR2 {self.capability} cannot send {opcode}"
        if len(args) != 3 or args[0] not in ("-1", "0", "1") or not self._decimal(args[1], 10000) or not self._decimal(args[2], 15):
            return "RR2 SET requires direction, basis points, and mask"
        direction, percent, mask = int(args[0]), int(args[1]), int(args[2])
        if direction == 0 and percent != 0:
            return "RR2 SET stop direction requires zero speed"
        if (direction != 0 or percent != 0) and self.capability != "ACTUATOR_DEBUG_ENCODER" and self.capability != "ACTUATOR_DEBUG_ALL":
            return f"RR2 {self.capability} cannot command drive"
        if mask & ~_DEBUG_MASKS[self.capability]:
            return f"RR2 {self.capability} cannot command output mask {mask}"
        return ""

    @staticmethod
    def _decimal(value: str, maximum: int) -> bool:
        return bool(value) and value.isascii() and value.isdecimal() and int(value) <= maximum

    def _is_on(self, opcode: str, args: tuple[str, ...]) -> bool:
        return opcode in _ON_COMMANDS or (opcode == "SET" and any(int(arg) for arg in args))

    @staticmethod
    def _completed(
        future: Future, error: str
    ) -> Future[CommandResult]:
        future.set_result(CommandResult(False, "", None, error))
        return future

    def drain_results(self, max_items: int) -> list[SerialResult]:
        if type(max_items) is not int or max_items < 0:
            raise ValueError("max_items must be a nonnegative integer")
        with self._condition:
            count = min(max_items, len(self._results))
            return [self._results.popleft() for _ in range(count)]

    def suspend_heartbeats(self) -> None:
        """Stop new/retry PING traffic without stopping command/RX/close work."""
        # Serialize with PING writes so return means no heartbeat can still be
        # emitted by an operation that began before the suspension request.
        with self._heartbeat_tx_lock:
            self._heartbeats_suspended = True
            self._ping_sequence = None
            self._ping_wire = None
            self._ping_attempts = 0

    def latch_fault(self, *, force: bool = True) -> Future[CommandResult]:
        """Reject mission work, retire active work, and queue the reserved OFF group."""
        if type(force) is not bool:
            return self._completed(Future(), "force must be a boolean")
        # Keep the heartbeat lock before the condition lock: retry service may
        # hold it while reporting a link failure through the condition.
        self.suspend_heartbeats()
        future: Future = Future()
        # Serialize the latch with initial and retry CMD writes. The TX lock is
        # always acquired before _condition so the latch cannot return while a
        # normal frame is still about to start on the wire.
        with self._command_tx_lock:
            with self._condition:
                if self._emergency_stop_future is not None:
                    return self._emergency_stop_future
                if self._closing or self._stop_requested:
                    return self._completed(future, "Serial worker is stopping")
                if self._failed or not self._ready_event.is_set():
                    reason = self._failure or self._startup_error or "RR2 session is not READY"
                    return self._completed(future, reason)
                self._fault_latched = True
                self._reject_all_locked("RR2 worker fault is latched")
                pending = self._active
                if (pending is not None and pending.group is not None
                        and not pending.group.safety):
                    pending.cancelled = True
                    self._complete_group_locked(
                        pending.group, False, "Cancelled by latched RR2 fault",
                        pending.sequence,
                    )
                if self._safety is not None:
                    return self._safety.futures[0]
                if self._safe_off_acknowledged and not force:
                    future.set_result(CommandResult(
                        True, self.session_id, None, state="OFF"
                    ))
                    return future
                self._off_barrier = True
                self._safe_off_acknowledged = False
                self._safety = _CommandGroup(self._stop_commands, [future], safety=True)
                self._condition.notify_all()
        return future

    def emergency_stop(self) -> Future[CommandResult]:
        """Prioritize one mission ESTOP, cancel queued work, and prevent restart."""
        future: Future[CommandResult] = Future()
        if self.capability != "MISSION":
            return self._completed(future, "ESTOP requires the MISSION capability")
        # Even if the link has failed, stop refreshing the MCU watchdog.
        self.suspend_heartbeats()
        with self._command_tx_lock:
            with self._condition:
                if self._emergency_stop_future is not None:
                    return self._emergency_stop_future
                if self._closing or self._stop_requested:
                    return self._completed(future, "Serial worker is stopping")
                if self._failed or not self._ready_event.is_set():
                    return self._completed(
                        future, self._failure or self._startup_error or
                        "RR2 session is not READY"
                    )
                self._fault_latched = True
                self._reject_all_locked("Cancelled by ESTOP")
                if self._active is not None and self._active.group is not None:
                    self._active.cancelled = True
                    self._complete_group_locked(
                        self._active.group, False, "Cancelled by ESTOP",
                        self._active.sequence,
                    )
                if self._safety is not None:
                    self._complete_group_locked(
                        self._safety, False, "Superseded by ESTOP",
                        self._safety.last_sequence,
                    )
                self._off_barrier = True
                self._safe_off_acknowledged = False
                self._safety = _CommandGroup(("ESTOP",), [future], safety=True)
                self._emergency_stop_future = future
                self._condition.notify_all()
        return future

    def stop(self, timeout_seconds: float, *, safe_off: bool = True) -> bool:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        deadline = time.monotonic() + timeout_seconds
        # Do this before taking _condition to preserve lock order with retry
        # service and to stop watchdog refresh while shutdown waits for OFF ACK.
        self.suspend_heartbeats()
        safe_future = None
        owner = False
        with self._command_tx_lock:
            with self._condition:
                if self._stopped_event.is_set():
                    return True
                if not self._closing:
                    owner = True
                    self._closing = True
                    self._reject_all_locked("Serial worker is shutting down")
                    pending = self._active
                    if (pending is not None and pending.group is not None
                            and not pending.group.safety):
                        pending.cancelled = True
                        self._complete_group_locked(
                            pending.group, False, "Cancelled by serial worker shutdown",
                            pending.sequence,
                        )
                    if safe_off and self._ready_event.is_set() and not self._failed:
                        if self._safety is not None:
                            safe_future = self._safety.futures[0]
                        elif self._safe_off_acknowledged:
                            safe_future = Future()
                            safe_future.set_result(CommandResult(
                                True, self.session_id, None, state="OFF"
                            ))
                        else:
                            safe_future = Future()
                            self._off_barrier = True
                            self._safe_off_acknowledged = False
                            self._safety = _CommandGroup(self._stop_commands, [safe_future], safety=True)
                    else:
                        self._stop_requested = True
                else:
                    # Concurrent/repeated callers join the existing shutdown
                    # OFF transaction instead of canceling it prematurely.
                    if self._safety is not None:
                        safe_future = self._safety.futures[0]
                    elif self._safe_off_acknowledged:
                        safe_future = Future()
                        safe_future.set_result(CommandResult(
                            True, self.session_id, None, state="OFF"
                        ))
                    elif self._stop_requested:
                        safe_future = None
                    else:
                        self._stop_requested = True
                self._condition.notify_all()
        safe_ok = True
        if safe_future is not None:
            try:
                safe_ok = safe_future.result(timeout=max(0.0, deadline - time.monotonic())).success
            except (TimeoutError, FutureTimeoutError):
                # Python 3.10(Jetson)에서는 Future.result의 timeout 예외가 내장 TimeoutError와 다르다.
                # 놓치면 stop()이 예외로 끝나 아래의 종료 정리(락 해제, 스레드 join)를 건너뛴다.
                safe_ok = False
        if not owner:
            # A concurrent caller observes shutdown within its own deadline,
            # but only the first caller may cancel transactions or close the
            # worker. In particular, a short joiner must not abort the owner's
            # still-pending SAFE_OFF ACK wait.
            stopped = self._stopped_event.wait(max(0.0, deadline - time.monotonic()))
            return safe_ok and stopped
        with self._condition:
            self._stop_requested = True
            if self._active is not None and self._active.group is not None:
                self._complete_group_locked(self._active.group, False, "Worker stopped", self._active.sequence)
                self._active = None
            if self._safety is not None:
                self._complete_group_locked(self._safety, False, "Worker stopped", self._safety.last_sequence)
                self._safety = None
            self._condition.notify_all()
        self._thread.join(max(0.0, deadline - time.monotonic()))
        return safe_ok and self._stopped_event.is_set()

    def _reject_queued_on_locked(self, reason: str) -> None:
        retained = deque()
        while self._normal:
            group = self._normal.popleft()
            if any(self._is_on(opcode, group.args) for opcode in group.opcodes):
                self._complete_group_locked(group, False, reason, group.last_sequence)
            else:
                retained.append(group)
        self._normal = retained

    def _reject_all_locked(self, reason: str) -> None:
        while self._normal:
            self._complete_group_locked(self._normal.popleft(), False, reason, None)

    @staticmethod
    def _next_lane_sequence(current: int, label: str) -> int:
        if current >= 0xFFFFFFFF:
            raise RuntimeError(f"RR2 {label} sequence exhausted")
        return current + 1

    def _write(self, frame: bytes) -> None:
        if self._serial.write(frame) != len(frame):
            raise OSError("Incomplete RR2 UART write")

    def _read_frames(self) -> list[Rr2Frame]:
        waiting = min(int(self._serial.in_waiting), 1024)
        if waiting <= 0:
            return []
        frames = []
        chunk = self._serial.read(waiting)
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        # One corrupted frame is discarded; parsing resumes only at the next
        # LF boundary inside the shared bounded stream parser.
        for part in chunk.splitlines(keepends=True):
            try:
                frames.extend(self._parser.feed(part))
            except ValueError:
                continue
        return frames

    def _handshake(self) -> None:
        hello = encode_rr2(Rr2Frame("HELLO", self.session_id, None, (self.capability,)))
        for _ in range(self._hello_attempts):
            self._write(hello)
            deadline = time.monotonic() + self._hello_timeout
            while time.monotonic() < deadline:
                frames = self._read_frames()
                for index, frame in enumerate(frames):
                    if (frame.kind == "READY" and frame.session == self.session_id
                            and frame.payload == ("2", self.capability)):
                        now = time.monotonic()
                        self._last_ack = self._last_ping = now
                        # A single UART read can contain READY followed by the first
                        # STATUS/EVENT. Retain those frames before waking callers.
                        for pending in frames[index + 1:]:
                            self._handle_frame(pending)
                        self._ready_event.set()
                        return
                time.sleep(self._poll_interval)
        raise TimeoutError(f"STM RR2 {self.capability} READY timed out")

    def _complete_group_locked(
        self, group: _CommandGroup, success: bool, error: str = "",
        sequence: int | None = None, state: str = "",
    ) -> None:
        result = CommandResult(success, self.session_id, sequence, error, state)
        for future in group.futures:
            if not future.done():
                future.set_result(result)
        if group.safety and self._safety is group:
            self._safety = None
            self._off_barrier = False

    def _queue_result(self, result: SerialResult) -> bool:
        overflow = False
        with self._condition:
            if len(self._results) >= self._result_capacity:
                status_index = next((index for index, item in enumerate(self._results)
                                     if item.kind == "STATUS"), None)
                if status_index is not None:
                    del self._results[status_index]
                elif result.kind == "STATUS":
                    return False
                else:
                    overflow = True
            if not overflow:
                self._results.append(result)
                return True
        # A dropped EVENT is a link fault, not merely a failed enqueue: the
        # active command and heartbeat must stop before another worker tick.
        self._fail_link("RR2 result queue overflowed on a discrete event")
        return False

    def _handle_frame(self, frame: Rr2Frame) -> None:
        if frame.session != self.session_id:
            return
        now = time.monotonic()
        if frame.kind == "ACK":
            opcode, result, state = frame.payload
            if opcode == "PING" and self._ping_sequence == frame.sequence:
                if result == "ACCEPTED":
                    self._last_ack = now
                    self._ping_sequence = None
                    self._ping_wire = None
                    self._ping_attempts = 0
                return
            pending = self._active
            if (pending is None or pending.cancelled or pending.ping or pending.sequence != frame.sequence
                    or pending.opcode != opcode):
                return
            if result != "ACCEPTED":
                self._finish_pending(False, f"STM rejected {opcode}: {result}", state)
                return
            self._last_ack = now
            self._ping_sequence = None
            self._ping_wire = None
            self._ping_attempts = 0
            self._finish_pending(True, "", state)
            return
        if frame.kind == "NACK":
            opcode = frame.payload[0]
            pending = self._active
            if (pending and not pending.cancelled and not pending.ping and pending.sequence == frame.sequence
                    and pending.opcode == opcode):
                self._finish_pending(False, "STM rejected RR2 command: " + " ".join(frame.payload[1:]))
            elif opcode == "PING" and self._ping_sequence == frame.sequence:
                self._fail_link("STM rejected RR2 heartbeat")
            return
        if frame.kind == "EVENT":
            duplicate = frame.sequence in self._seen_events
            if not duplicate and frame.sequence > self._last_event_sequence:
                # Never tell the MCU to discard an event that cannot be
                # delivered to the bounded ROS result queue.
                if not self._queue_result(SerialResult(
                    "EVENT", frame.sequence, frame.payload, frame.session
                )):
                    return
                self._seen_events.add(frame.sequence)
                self._last_event_sequence = frame.sequence
                if len(self._seen_events) > 256:
                    self._seen_events = {max(self._seen_events)}
            elif not duplicate:
                self._fail_link("RR2 event sequence is stale without a delivered record")
                return
            # Duplicates were already queued; only those and newly queued
            # events may be acknowledged.
            event_ack = encode_rr2(Rr2Frame(
                "EVENT_ACK", self.session_id, frame.sequence, ("ACCEPTED",)
            ))
            self._write(event_ack)
            return
        if frame.kind == "STATUS":
            if frame.sequence > self._last_status_sequence:
                self._last_status_sequence = frame.sequence
                self._queue_result(SerialResult(
                    "STATUS", frame.sequence, frame.payload, frame.session
                ))

    def _finish_pending(self, success: bool, error: str = "", state: str = "") -> None:
        pending, self._active = self._active, None
        if pending is None or pending.group is None:
            return
        group = pending.group
        group.last_sequence = pending.sequence
        if not success:
            with self._condition:
                self._complete_group_locked(group, False, error, pending.sequence, state)
                if group.safety:
                    self._safe_off_acknowledged = False
                    self._failed = True
                    self._failure = f"SAFE_OFF failed: {error}"
                    self._reject_all_locked(self._failure)
            return
        if group.safety:
            group.index += 1
            if group.index >= len(group.opcodes):
                with self._condition:
                    self._safe_off_acknowledged = True
                    self._complete_group_locked(group, True, sequence=pending.sequence, state=state)
            return
        with self._condition:
            self._complete_group_locked(group, True, sequence=pending.sequence, state=state)

    def _dispatch(self) -> None:
        with self._command_tx_lock:
            with self._condition:
                if self._active is not None:
                    return
                if self._safety is not None:
                    group = self._safety
                    opcode = group.opcodes[group.index]
                elif self._normal and not self._off_barrier and not self._fault_latched:
                    group = self._normal.popleft()
                    opcode = group.opcodes[group.index]
                else:
                    return
                self._cmd_sequence = self._next_lane_sequence(self._cmd_sequence, "CMD")
                sequence = self._cmd_sequence
                wire = encode_rr2(Rr2Frame(
                    "CMD", self.session_id, sequence,
                    (opcode, *(() if group.safety else group.args))
                ))
                pending = _Pending(group, opcode, sequence, wire, 1, 0.0)
                self._active = pending
            # Record ownership before writing. If serial.write raises or reports
            # a partial frame, _fail_link can resolve the dequeued group Future.
            self._write(wire)
            # A successful write may consume part of the ACK budget. Start the
            # deadline only after the complete frame leaves the UART.
            with self._condition:
                if self._active is pending and not pending.cancelled:
                    pending.deadline = time.monotonic() + self._ack_timeout

    def _service_retries(self, now: float) -> None:
        with self._command_tx_lock:
            pending = self._active
            if pending is not None and pending.cancelled:
                self._active = None
                pending = None
            if pending is not None and now >= pending.deadline:
                if pending.attempts >= self._retry_limit:
                    self._finish_pending(False, f"ACK timed out for {pending.opcode}")
                else:
                    # Same session, sequence, opcode and bytes: retries are
                    # idempotent at the MCU request cache.
                    self._write(pending.wire)
                    if self._active is pending and not pending.cancelled:
                        pending.attempts += 1
                        pending.deadline = time.monotonic() + self._ack_timeout
        if self._failed:
            return
        now = time.monotonic()
        with self._heartbeat_tx_lock:
            if self._heartbeats_suspended:
                return
            if now - self._last_ack >= self._heartbeat_timeout:
                self._fail_link("STM RR2 heartbeat acknowledgement timed out")
                return
            if self._ping_sequence is None and now - self._last_ping >= self._heartbeat_interval:
                self._ping_sequence_counter = self._next_lane_sequence(
                    self._ping_sequence_counter, "PING"
                )
                sequence = self._ping_sequence_counter
                wire = encode_rr2(Rr2Frame("PING", self.session_id, sequence, ()))
                self._write(wire)
                completed = time.monotonic()
                self._ping_sequence, self._ping_wire = sequence, wire
                self._ping_attempts = 1
                self._ping_deadline = completed + self._ack_timeout
                self._last_ping = completed
            elif self._ping_sequence is not None and now >= self._ping_deadline:
                if self._ping_attempts >= self._retry_limit:
                    self._ping_sequence = self._ping_wire = None
                else:
                    self._write(self._ping_wire)
                    completed = time.monotonic()
                    self._ping_attempts += 1
                    self._ping_deadline = completed + self._ack_timeout
                    self._last_ping = completed

    def _fail_link(self, reason: str) -> None:
        with self._condition:
            self._failed = True
            self._safe_off_acknowledged = False
            self._failure = reason
            if self._active is not None:
                pending, self._active = self._active, None
                if pending.group is not None:
                    self._complete_group_locked(
                        pending.group, False, reason, pending.sequence
                    )
            self._reject_all_locked(reason)
            if self._safety is not None:
                self._complete_group_locked(self._safety, False, reason, None)
            self._ping_sequence = self._ping_wire = None

    def _run(self) -> None:
        self._serial = None
        try:
            self._serial = self._factory(**self._serial_kwargs)
            self._handshake()
            while True:
                with self._condition:
                    if self._stop_requested:
                        break
                try:
                    for frame in self._read_frames():
                        self._handle_frame(frame)
                    now = time.monotonic()
                    self._service_retries(now)
                    if not self._failed:
                        self._dispatch()
                except (OSError, RuntimeError, ValueError) as exc:
                    self._fail_link(str(exc))
                with self._condition:
                    self._condition.wait(self._poll_interval)
        except BaseException as exc:
            self._startup_error = str(exc)
            self._fail_link(self._startup_error)
        finally:
            with self._condition:
                self._reject_all_locked("Serial worker stopped")
                if self._safety is not None:
                    self._complete_group_locked(self._safety, False, "Serial worker stopped")
                    self._safety = None
            try:
                if self._serial is not None:
                    self._serial.close()
            finally:
                self._serial = None
                self._stopped_event.set()
