"""Temporal safety decisions and independent FRONT/SIDE UART scheduling.

Detector execution, cameras, reports and UI are outside this responsibility.
Existing validated-result checks and heartbeat/hold timing are preserved.
"""
from __future__ import annotations

# 모델이 낸 마스크/등급을 시간 이력과 운용 규칙으로 청소 명령에 바꾸는 경계다. 여기서 모델 가중치를 학습하지 않는다.
# 모델 가중치(.plan 등), detector의 confidence/마스크 후처리 설정, settings.py의 제어 임계값은 서로 다른 조정 대상이다.
# 읽는 순서: 결과 검증 함수 → FrontControlHistory/RealtimeControlHistory → immediate_off/hold → ROS 또는 UART 출력.

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from crack_detector import REALTIME_DETECTOR_PREFIX as REALTIME_CRACK_DETECTOR_PREFIX
from hrsegnet_crack_detector import DETECTOR_PREFIX as HRSEGNET_CRACK_DETECTOR_PREFIX
from multitask_detector import CRACK_METHOD_PREFIX as MULTITASK_CRACK_METHOD_PREFIX
from obstacle_detector import (
    DETECTOR_PREFIX as OBSTACLE_DETECTOR_PREFIX,
    ObstacleDetectionResult,
)

from .settings import *
from .uart import CleanerController, PumpController

if TYPE_CHECKING:
    from .data import CameraFrame, RealtimeInferenceOutcome, TopRealtimeInferenceOutcome
    from .vision import AlternatingRealtimeInference, AlternatingTopRealtimeInference

@dataclass(frozen=True)
class _ScheduledActuatorDecision:
    decision: RealtimeControlDecision | FrontControlDecision
    valid_until: float


@dataclass(frozen=True)
class RealtimeActuatorHoldSnapshot:
    """Read-only hold state captured before a waiting-path update."""

    captured_at: float
    has_decision: bool
    loss_deadline_was_set: bool
    valid_until: float | None
    watchdog_expired: bool
    loss_started_at: float | None


class RealtimeActuatorDecisionHold:
    """Hold the last validated command after control first loses readiness."""

    def __init__(
        self,
        hold_seconds: float = REALTIME_LAST_VALID_COMMAND_HOLD_SECONDS,
    ) -> None:
        if hold_seconds <= 0.0:
            raise ValueError("Realtime actuator hold must be positive.")
        self.hold_seconds = float(hold_seconds)
        self.clear()

    def clear(self) -> None:
        self._decision = None
        self._watchdog_deadline: float | None = None
        self._loss_deadline: float | None = None

    def accept(
        self,
        decision: RealtimeControlDecision | FrontControlDecision,
        *,
        now: float | None = None,
    ) -> None:
        if now is None:
            now = time.monotonic()
        self._decision = decision
        self._watchdog_deadline = float(now) + self.hold_seconds
        self._loss_deadline = None

    @property
    def valid_until(self) -> float | None:
        if self._decision is None:
            return None
        return (
            self._loss_deadline
            if self._loss_deadline is not None
            else self._watchdog_deadline
        )

    @property
    def has_decision(self) -> bool:
        return self._decision is not None

    @property
    def loss_started_at(self) -> float | None:
        if self._loss_deadline is None:
            return None
        return self._loss_deadline - self.hold_seconds

    def diagnostic_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> RealtimeActuatorHoldSnapshot:
        if now is None:
            now = time.monotonic()
        captured_at = float(now)
        loss_deadline_was_set = self._loss_deadline is not None
        valid_until = self.valid_until
        watchdog_expired = bool(
            self._decision is not None
            and not loss_deadline_was_set
            and self._watchdog_deadline is not None
            and captured_at > self._watchdog_deadline
        )
        return RealtimeActuatorHoldSnapshot(
            captured_at=captured_at,
            has_decision=self._decision is not None,
            loss_deadline_was_set=loss_deadline_was_set,
            valid_until=valid_until,
            watchdog_expired=watchdog_expired,
            loss_started_at=self.loss_started_at,
        )

    def begin_loss(
        self,
        *,
        now: float | None = None,
    ) -> RealtimeControlDecision | FrontControlDecision | None:
        # 정상 accept는 마지막 유효 결정의 watchdog을 갱신한다. 준비 손실을 처음 발견하면 별도 손실 기한을 한 번 설정한다.
        # 기본 hold_seconds=1.2초이며 이미 watchdog이 만료됐다면 손실 기한을 새로 열지 않는다.
        # 손실 감지가 늦어지면 accept 시각과 loss 기한의 기준이 달라진다. 새 위험은 호출자가 clear하여 이 유지를 끊는다.
        if self._decision is None:
            return None
        if now is None:
            now = time.monotonic()
        if self._loss_deadline is None:
            if (
                self._watchdog_deadline is None
                or float(now) > self._watchdog_deadline
            ):
                self.clear()
                return None
            self._loss_deadline = float(now) + self.hold_seconds
        return self.current(now=now)

    def current(
        self,
        *,
        now: float | None = None,
    ) -> RealtimeControlDecision | FrontControlDecision | None:
        if self._decision is None:
            return None
        if now is None:
            now = time.monotonic()
        valid_until = self.valid_until
        if valid_until is None or float(now) > valid_until:
            self.clear()
            return None
        return self._decision


class RealtimeActuatorArbiter:
    """Own all realtime UART writes and heartbeat FRONT/SIDE independently."""

    _ROLES = ("front", "side")

    def __init__(
        self,
        uart,
        *,
        heartbeat_seconds: float = REALTIME_ACTUATOR_HEARTBEAT_SECONDS,
    ) -> None:
        if uart is None:
            raise ValueError("Realtime actuator arbiter requires an open UART.")
        if heartbeat_seconds <= 0.0:
            raise ValueError("Realtime actuator heartbeat must be positive.")
        self.heartbeat_seconds = float(heartbeat_seconds)
        self._controllers = {
            "front": (
                CleanerController(uart, FRONT_ACTUATOR_COMMANDS),
                PumpController(uart, FRONT_ACTUATOR_COMMANDS),
            ),
            "side": (
                CleanerController(uart, SIDE_ACTUATOR_COMMANDS),
                PumpController(uart, SIDE_ACTUATOR_COMMANDS),
            ),
        }
        self._targets: dict[str, _ScheduledActuatorDecision | None] = {
            role: None for role in self._ROLES
        }
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Realtime actuator arbiter was already started.")
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-uart-arbiter",
            daemon=True,
        )
        self._thread.start()

    def publish(
        self,
        role: str,
        decision: RealtimeControlDecision | FrontControlDecision,
        *,
        valid_until: float,
    ) -> None:
        if role not in self._targets:
            raise ValueError(f"Unknown realtime actuator role: {role}")
        if not isinstance(valid_until, (int, float)):
            raise ValueError("Realtime actuator valid_until must be numeric.")
        with self._lock:
            self._targets[role] = _ScheduledActuatorDecision(
                decision,
                float(valid_until),
            )
        self._wake.set()

    def clear(self, role: str) -> None:
        if role not in self._targets:
            raise ValueError(f"Unknown realtime actuator role: {role}")
        with self._lock:
            self._targets[role] = None
        self._wake.set()

    def clear_all(self) -> None:
        with self._lock:
            for role in self._ROLES:
                self._targets[role] = None
        self._wake.set()

    def desired_state(self) -> tuple[float, bool, float, bool]:
        with self._lock:
            front = self._targets["front"]
            side = self._targets["side"]
        now = time.monotonic()

        def state(target: _ScheduledActuatorDecision | None) -> tuple[float, bool]:
            if target is None or now > target.valid_until:
                return 0.0, False
            return (
                target.decision.cleaner_duty_percent,
                target.decision.pump_on,
            )

        front_state = state(front)
        side_state = state(side)
        return (*front_state, *side_state)

    def failure_message(self) -> str | None:
        with self._lock:
            return self._failure

    def raise_if_failed(self) -> None:
        failure = self.failure_message()
        if failure is not None:
            raise RuntimeError(failure)

    def worker_is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        if self._thread is None:
            return
        self.clear_all()
        self._stop.set()
        self._wake.set()
        self._thread.join(REALTIME_ACTUATOR_WORKER_CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Realtime actuator arbiter did not stop safely.")
        self._thread = None
        self.raise_if_failed()

    def _safe_off_all(self) -> None:
        force_cleaning_pairs_safe_off(
            self._controllers["front"],
            self._controllers["side"],
        )

    def _run(self) -> None:
        try:
            self._safe_off_all()
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    targets = dict(self._targets)
                for role in self._ROLES:
                    cleaner, pump = self._controllers[role]
                    target = targets[role]
                    if target is None or now > target.valid_until:
                        force_cleaning_safe_off(cleaner, pump)
                    else:
                        update_cleaning_actuators(
                            cleaner,
                            pump,
                            decision=target.decision,
                        )
                self._wake.wait(self.heartbeat_seconds)
                self._wake.clear()
        except Exception as exc:
            with self._lock:
                self._failure = f"Realtime UART arbiter failed: {exc}"
            self._stop.set()
        finally:
            try:
                self._safe_off_all()
            except Exception as exc:
                with self._lock:
                    if self._failure is None:
                        self._failure = (
                            f"Realtime UART arbiter safe-OFF failed: {exc}"
                        )


def highest_rust_grade(class_map) -> int:
    """Return the highest valid rust class present in the realtime ROI."""

    # SIDE 제어 ROI의 class_map은 기본 (240, 1280), 클래스는 0~3이어야 한다. confidence나 녹 면적 비율이 아니다.
    # 평균/다수결 대신 가장 높은 픽셀 등급을 반환하므로 작은 고등급 영역도 최종 등급을 올린다.
    # 과도한 OFF를 분석할 때는 이 최댓값 규칙과 detector가 만든 실제 class_map을 함께 확인한다.
    if not isinstance(class_map, np.ndarray) or class_map.ndim != 2:
        raise ValueError("Realtime rust class map must be a two-dimensional array.")
    expected_shape = (
        REALTIME_RUST_ROI_BOTTOM - REALTIME_RUST_ROI_TOP,
        FRAME_WIDTH,
    )
    if class_map.shape != expected_shape:
        raise ValueError("Realtime rust class map shape does not match the control ROI.")
    if not np.isin(class_map, RUST_CLASS_IDS).all():
        raise ValueError("Realtime rust class map contains an invalid class ID.")
    return int(np.max(class_map))


def water_control_blocked_by_rust(class_map) -> bool:
    """Fail closed; every non-zero rust grade blocks the water pump."""

    try:
        return highest_rust_grade(class_map) >= 1
    except ValueError:
        return True


def validated_crack_ratio(crack_result) -> float:
    """Validate a realtime crack result and return its filtered ROI ratio."""

    # 승인된 detector의 ready 결과인지, 기본 균열 ROI (180, 1280)의 0/255 이진 마스크인지 먼저 확인한다.
    # crack_ratio는 후처리된 흰 픽셀 수/전체 ROI 픽셀 수(0~1)이며 모델 confidence와 다르다.
    # 전달된 비율을 마스크로 다시 계산해 비교하므로 다른 ROI 비율이나 표시용 백분율을 넣으면 오류가 된다.
    if crack_result is None or getattr(crack_result, "status", None) != "ready":
        raise ValueError("Realtime crack result is missing or not ready.")
    crack_method = str(getattr(crack_result, "method", ""))
    if not crack_method.startswith(
        (
            REALTIME_CRACK_DETECTOR_PREFIX,
            MULTITASK_CRACK_METHOD_PREFIX,
            HRSEGNET_CRACK_DETECTOR_PREFIX,
        )
    ):
        raise ValueError("Realtime crack result method is not approved.")
    mask = getattr(crack_result, "mask", None)
    expected_shape = (
        REALTIME_CRACK_ROI_BOTTOM - REALTIME_CRACK_ROI_TOP,
        FRAME_WIDTH,
    )
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.shape != expected_shape
    ):
        raise ValueError("Realtime crack mask shape does not match the control ROI.")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("Realtime crack mask must contain only zero and 255.")
    ratio = float(getattr(crack_result, "crack_ratio", float("nan")))
    measured_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if not np.isfinite(ratio) or abs(ratio - measured_ratio) > 1e-9:
        raise ValueError("Realtime crack ratio does not match its filtered mask.")
    return ratio


def cleaning_blocked_by_crack(crack_result) -> bool:
    """Return whether the validated crack ROI exceeds the 0.05-percent limit."""

    # 기본 CRACK_CONTROL_STOP_RATIO=0.0005는 0.05%다. 0.05를 넣으면 5%가 되어 기준이 100배 달라진다.
    # 비교는 엄격한 > 이므로 정확히 임계값인 경우는 이 조건만으로 차단하지 않는다.
    # 검증 실패는 균열 없음으로 보지 않고 차단한다. detector의 마스크 생성 임계값은 이 제어 기준과 별개다.
    try:
        return validated_crack_ratio(crack_result) > CRACK_CONTROL_STOP_RATIO
    except ValueError:
        return True


def obstacle_detected_in_control_roi(result: ObstacleDetectionResult) -> bool:
    """Return the validated front-camera bottom-third obstacle decision."""

    # TOP 전체 화면의 아무 박스나 다시 검색하지 않고 y=480:720 제어 ROI의 bool을 사용한다.
    # 클래스/confidence/박스와 ROI의 판정은 obstacle_detector.py 쪽 책임이다. 이 함수는 타입·상태·method를 검증한다.
    # 화면에는 박스가 있는데 제어가 바뀌지 않으면 control_roi_detected와 detector의 ROI 판정부터 확인한다.
    if not isinstance(result, ObstacleDetectionResult) or result.status != "ready":
        raise ValueError("Realtime obstacle result is missing or not ready.")
    if not str(result.method).startswith(OBSTACLE_DETECTOR_PREFIX):
        raise ValueError("Realtime obstacle result method is not approved.")
    if type(result.control_roi_detected) is not bool:
        raise ValueError("Realtime obstacle control ROI flag must be boolean.")
    return result.control_roi_detected


def update_cleaning_actuators(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    decision: RealtimeControlDecision | FrontControlDecision,
) -> None:
    """Apply an already validated four-result temporal control decision."""

    cleaner_controller.set_duty_percent(decision.cleaner_duty_percent)
    pump_controller.update(decision.pump_on)


def publish_realtime_actuator_decision(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    role: str,
    history: RealtimeControlHistory | FrontControlHistory,
    decision: RealtimeControlDecision | FrontControlDecision,
    hold: RealtimeActuatorDecisionHold | None = None,
    arbiter: RealtimeActuatorArbiter | None = None,
) -> None:
    """Publish one role atomically, or preserve simulated/direct behavior."""

    if hold is not None:
        hold.accept(decision)
    if arbiter is None:
        update_cleaning_actuators(
            cleaner_controller,
            pump_controller,
            decision=decision,
        )
        return
    valid_until = hold.valid_until if hold is not None else history.valid_until
    if valid_until is None:
        arbiter.clear(role)
        return
    arbiter.publish(role, decision, valid_until=valid_until)


def clear_realtime_actuator_role(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    role: str,
    arbiter: RealtimeActuatorArbiter | None = None,
) -> None:
    if arbiter is None:
        force_cleaning_safe_off(cleaner_controller, pump_controller)
    else:
        arbiter.clear(role)


def maintain_waiting_role_control(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    history: RealtimeControlHistory | FrontControlHistory,
    inference: AlternatingRealtimeInference | AlternatingTopRealtimeInference,
    role: str = "side",
    hold: RealtimeActuatorDecisionHold | None = None,
    arbiter: RealtimeActuatorArbiter | None = None,
) -> bool:
    """Heartbeat a still-fresh cached decision without mutating its history."""

    if hold is not None:
        history.prune()
        decision = hold.current() if history.ready else hold.begin_loss()
        valid_until = hold.valid_until
        if decision is not None and valid_until is not None:
            if arbiter is None:
                update_cleaning_actuators(
                    cleaner_controller,
                    pump_controller,
                    decision=decision,
                )
            else:
                arbiter.publish(role, decision, valid_until=valid_until)
            return True
        clear_realtime_actuator_role(
            cleaner_controller,
            pump_controller,
            role=role,
            arbiter=arbiter,
        )
        return False

    cache_ttl = inference.remaining_fresh_seconds()
    history.prune()
    if history.ready and cache_ttl is not None and cache_ttl > 0.0:
        publish_realtime_actuator_decision(
            cleaner_controller,
            pump_controller,
            role=role,
            history=history,
            decision=history.decision(),
            arbiter=arbiter,
        )
        return True
    clear_realtime_actuator_role(
        cleaner_controller,
        pump_controller,
        role=role,
        arbiter=arbiter,
    )
    return False


@dataclass(frozen=True)
class ControlOffDiagnostic:
    reason: str
    monotonic_seconds: float
    history_ready: bool
    detail: str | None = None
    frame_sequence: int | None = None
    frame_age_ms: float | None = None
    new_crack_pct: float | None = None
    new_crack_blocked: bool = False
    new_rust_grade: int | None = None
    history_crack_pct: float | None = None
    history_rust_grade: int | None = None
    loss_age_ms: float | None = None
    hold_remaining_ms: float | None = None
    pre_loss_deadline_set: bool | None = None
    pre_valid_until: float | None = None
    watchdog_expired_at_off: bool | None = None


def select_control_off_reason(
    *,
    camera_error: bool,
    new_crack_blocked: bool,
    new_rust_grade: int | None,
    new_rust_forced_off: bool,
    rolling_history_hazard: bool,
    had_valid_decision: bool | None,
    loss_deadline_was_set: bool = False,
    watchdog_expired: bool = False,
) -> str:
    """Select one stable OFF reason using the documented safety priority."""

    if camera_error:
        return "camera_error"
    if new_crack_blocked:
        return "new_crack"
    if new_rust_forced_off and new_rust_grade is not None:
        return "new_rust_grade"
    if rolling_history_hazard:
        return "rolling_hazard"
    if watchdog_expired:
        return "decision_watchdog_expired"
    if had_valid_decision is True and loss_deadline_was_set:
        return "history_hold_expired"
    if had_valid_decision is False:
        return "startup_or_no_valid_decision"
    return "unknown"


class DualControlOffTransitionLogger:
    """Emit one diagnostic for each independent role ON-to-OFF transition."""

    _ROLE_STATE_INDICES = {"front": (0, 1), "side": (2, 3)}

    def __init__(self) -> None:
        self._previous_active = {"front": None, "side": None}

    @classmethod
    def _role_active(
        cls,
        role: str,
        actuator_state: tuple[float, bool, float, bool],
    ) -> bool:
        cleaner_index, pump_index = cls._ROLE_STATE_INDICES[role]
        return bool(
            actuator_state[cleaner_index] > 0.0
            or actuator_state[pump_index]
        )

    def is_turning_off(
        self,
        role: str,
        actuator_state: tuple[float, bool, float, bool],
    ) -> bool:
        return bool(
            self._previous_active[role] is True
            and not self._role_active(role, actuator_state)
        )

    @staticmethod
    def _format_optional(value, *, decimals: int = 1) -> str:
        if value is None:
            return "na"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)

    @staticmethod
    def _format_detail(value: str | None) -> str:
        if value is None:
            return "na"
        return repr(" ".join(str(value).split()))

    def observe(
        self,
        actuator_state: tuple[float, bool, float, bool],
        *,
        front: ControlOffDiagnostic | None = None,
        side: ControlOffDiagnostic | None = None,
    ) -> tuple[str, ...]:
        diagnostics = {"front": front, "side": side}
        lines: list[str] = []
        for role in self._ROLE_STATE_INDICES:
            active = self._role_active(role, actuator_state)
            if self._previous_active[role] is True and not active:
                diagnostic = diagnostics[role]
                if diagnostic is None:
                    raise ValueError(f"Missing {role} CONTROL_OFF diagnostic.")
                lines.append(
                    "[CONTROL_OFF] "
                    f"mono={diagnostic.monotonic_seconds:.6f} "
                    f"role={role} reason={diagnostic.reason} "
                    f"detail={self._format_detail(diagnostic.detail)} "
                    f"frame_seq={self._format_optional(diagnostic.frame_sequence)} "
                    "frame_age_ms="
                    f"{self._format_optional(diagnostic.frame_age_ms)} "
                    "new_crack_pct="
                    f"{self._format_optional(diagnostic.new_crack_pct, decimals=4)} "
                    f"new_crack_blocked={str(diagnostic.new_crack_blocked).lower()} "
                    "new_rust_grade="
                    f"{self._format_optional(diagnostic.new_rust_grade)} "
                    f"hist_ready={str(diagnostic.history_ready).lower()} "
                    "hist_crack_pct="
                    f"{self._format_optional(diagnostic.history_crack_pct, decimals=4)} "
                    "hist_rust_grade="
                    f"{self._format_optional(diagnostic.history_rust_grade)} "
                    "loss_age_ms="
                    f"{self._format_optional(diagnostic.loss_age_ms)} "
                    "hold_remaining_ms="
                    f"{self._format_optional(diagnostic.hold_remaining_ms)} "
                    "pre_loss_deadline_set="
                    f"{self._format_optional(diagnostic.pre_loss_deadline_set)} "
                    "pre_valid_until="
                    f"{self._format_optional(diagnostic.pre_valid_until, decimals=6)} "
                    "watchdog_expired_at_off="
                    f"{self._format_optional(diagnostic.watchdog_expired_at_off)}"
                )
            self._previous_active[role] = active
        return tuple(lines)

    def observe_forced_off(
        self,
        actuator_state_before_off: tuple[float, bool, float, bool],
        *,
        reason: str,
        detail: str,
        now: float,
    ) -> tuple[str, ...]:
        diagnostics = {}
        for role in self._ROLE_STATE_INDICES:
            active = self._role_active(role, actuator_state_before_off)
            self._previous_active[role] = active
            if active:
                diagnostics[role] = ControlOffDiagnostic(
                    reason=reason,
                    monotonic_seconds=now,
                    history_ready=False,
                    detail=detail,
                )
        return self.observe(
            (0.0, False, 0.0, False),
            front=diagnostics.get("front"),
            side=diagnostics.get("side"),
        )


def classify_runtime_control_off(exc: Exception) -> tuple[str, str]:
    """Return a stable safety-OFF reason and single-line exception detail."""

    detail = " ".join(str(exc).split()) or type(exc).__name__
    lower_detail = detail.lower()
    reason = (
        "uart_error"
        if any(token in lower_detail for token in ("uart", "serial", " stm"))
        else "runtime_error"
    )
    return reason, detail


def current_dual_actuator_state(
    front_cleaner: CleanerController,
    front_pump: PumpController,
    side_cleaner: CleanerController,
    side_pump: PumpController,
    *,
    arbiter: RealtimeActuatorArbiter | None,
) -> tuple[float, bool, float, bool]:
    if arbiter is not None:
        return arbiter.desired_state()
    return (
        front_cleaner.duty_percent,
        front_pump.is_on,
        side_cleaner.duty_percent,
        side_pump.is_on,
    )


@dataclass(frozen=True)
class RealtimeControlDecision:
    """Worst-case command derived from each detector's last four inferences."""

    cleaner_duty_percent: float
    pump_on: bool
    rust_grade: int
    crack_ratio: float


@dataclass(frozen=True)
class FrontControlDecision:
    """Front-pair command derived from top-camera four-result windows."""

    cleaner_duty_percent: float
    pump_on: bool
    foreign_seen_recently: bool
    crack_ratio: float


class FrontControlHistory:
    """Hold independent top obstacle and crack four-result windows."""

    # 기본값은 장애물 4개 + 균열 4개의 독립 이력이다. 교대 추론의 카메라 프레임 4장과 같지 않다.
    # 각 결과에는 원본 읽기 완료 시각이 붙고 기본 TTL=0.8초를 지난 항목은 prune에서 제거한다.
    # 한 모델이 느려서 4개의 신선한 표본을 못 채우면 다른 모델 결과가 충분해도 ready=False다.
    def __init__(
        self,
        *,
        window_size: int = REALTIME_CONTROL_WINDOW_SIZE,
        result_ttl_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Front control window size must be positive.")
        if result_ttl_seconds <= 0.0:
            raise ValueError("Front control result TTL must be positive.")
        self.window_size = int(window_size)
        self.result_ttl_seconds = float(result_ttl_seconds)
        self.reset()

    def reset(self) -> None:
        self._crack_ratios: deque[tuple[float, float]] = deque(
            maxlen=self.window_size
        )
        self._foreign_detected: deque[tuple[float, bool]] = deque(
            maxlen=self.window_size
        )

    def prune(self, *, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.result_ttl_seconds
        while self._crack_ratios and self._crack_ratios[0][0] < cutoff:
            self._crack_ratios.popleft()
        while self._foreign_detected and self._foreign_detected[0][0] < cutoff:
            self._foreign_detected.popleft()

    @property
    def ready(self) -> bool:
        self.prune()
        return self.ready_without_prune

    @property
    def ready_without_prune(self) -> bool:
        return (
            len(self._crack_ratios) == self.window_size
            and len(self._foreign_detected) == self.window_size
        )

    @property
    def valid_until(self) -> float | None:
        self.prune()
        if not self.ready:
            return None
        oldest = min(self._crack_ratios[0][0], self._foreign_detected[0][0])
        return oldest + self.result_ttl_seconds

    def update_crack(
        self,
        crack_result,
        *,
        observed_at: float | None = None,
    ) -> None:
        try:
            ratio = validated_crack_ratio(crack_result)
        except (TypeError, ValueError) as exc:
            self.reset()
            if isinstance(exc, ValueError):
                raise
            raise ValueError("Realtime crack ratio is not numeric.") from exc
        if observed_at is None:
            observed_at = time.monotonic()
        self._crack_ratios.append((float(observed_at), ratio))
        self.prune(now=float(observed_at))

    def update_foreign(
        self,
        detected: bool,
        *,
        observed_at: float | None = None,
    ) -> None:
        if type(detected) is not bool:
            self.reset()
            raise ValueError("Foreign-object control result must be a boolean.")
        if observed_at is None:
            observed_at = time.monotonic()
        self._foreign_detected.append((float(observed_at), detected))
        self.prune(now=float(observed_at))

    def update(self, outcome: TopRealtimeInferenceOutcome) -> None:
        """Append only the top-camera result produced on this frame."""

        # 이번 outcome에 실제로 들어 있는 모델 결과만 해당 이력에 추가한다. 다른 모델의 이전 결과를 복제하지 않는다.
        # 두 결과가 모두 없고 ready도 아니면 이력을 초기화한다. 일시적 대기와 모델별 신선도는 호출자 경로까지 함께 본다.
        if (
            not outcome.ready
            and outcome.display_crack_result is None
            and outcome.display_obstacle_result is None
        ):
            self.reset()
            return
        observed_at = outcome.frame_read_completed_at
        if outcome.display_crack_result is not None:
            self.update_crack(
                outcome.display_crack_result,
                observed_at=observed_at,
            )
        if outcome.display_obstacle_result is not None:
            self.update_foreign(
                obstacle_detected_in_control_roi(
                    outcome.display_obstacle_result
                ),
                observed_at=observed_at,
            )

    def decision(self) -> FrontControlDecision:
        # 신선한 이력이 모두 찬 뒤 균열은 4개 중 최대 비율, 장애물은 4개 중 하나라도 검출됐는지로 결정한다.
        # 균열 초과는 앞 브러시/펌프 모두 OFF. 균열이 없으면 장애물 유무에 따라 브러시 단계 25/12.5, 펌프 ON이다.
        # 이 단계 숫자는 UART/STM의 등록 PWM 단계명이며 모델 점수나 실제 회전속도 백분율이 아니다.
        if not self.ready:
            raise RuntimeError(
                "Four fresh top-crack and foreign-object results are required."
            )

        crack_ratio = max(value for _, value in self._crack_ratios)
        foreign_seen_recently = any(value for _, value in self._foreign_detected)
        crack_blocked = crack_ratio > CRACK_CONTROL_STOP_RATIO
        return FrontControlDecision(
            cleaner_duty_percent=(
                0.0
                if crack_blocked
                else (
                    CLEANER_STAGE_ONE_DUTY_PERCENT
                    if foreign_seen_recently
                    else CLEANER_BASE_DUTY_PERCENT
                )
            ),
            pump_on=not crack_blocked,
            foreign_seen_recently=foreign_seen_recently,
            crack_ratio=crack_ratio,
        )


class RealtimeControlHistory:
    """Hold independent four-inference windows for alternating detectors."""

    def __init__(
        self,
        *,
        no_crack: bool = False,
        window_size: int = REALTIME_CONTROL_WINDOW_SIZE,
        result_ttl_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Realtime control window size must be positive.")
        if result_ttl_seconds <= 0.0:
            raise ValueError("Realtime control result TTL must be positive.")
        self.no_crack = bool(no_crack)
        self.window_size = int(window_size)
        self.result_ttl_seconds = float(result_ttl_seconds)
        self.reset()

    def reset(self) -> None:
        self._rust_grades: deque[tuple[float, int]] = deque(
            maxlen=self.window_size
        )
        self._crack_ratios: deque[tuple[float, float]] = deque(
            maxlen=self.window_size
        )

    def prune(self, *, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.result_ttl_seconds
        while self._rust_grades and self._rust_grades[0][0] < cutoff:
            self._rust_grades.popleft()
        while self._crack_ratios and self._crack_ratios[0][0] < cutoff:
            self._crack_ratios.popleft()

    @property
    def ready(self) -> bool:
        self.prune()
        return self.ready_without_prune

    @property
    def ready_without_prune(self) -> bool:
        rust_ready = len(self._rust_grades) == self.window_size
        # SIDE도 녹/균열 각각 기본 4개가 필요하며 앞의 prune이 0.8초 지난 원본 표본을 지운다.
        # no_crack=True인 호출 경로만 균열 이력 조건을 생략한다. 실제 ROS 운용에서 사용되는 옵션인지는 호출자를 확인한다.
        crack_ready = self.no_crack or len(self._crack_ratios) == self.window_size
        return rust_ready and crack_ready

    @property
    def valid_until(self) -> float | None:
        self.prune()
        if not self.ready:
            return None
        oldest = self._rust_grades[0][0]
        if not self.no_crack:
            oldest = min(oldest, self._crack_ratios[0][0])
        return oldest + self.result_ttl_seconds

    def update(self, outcome: RealtimeInferenceOutcome) -> None:
        """Append only the detector result produced on this camera frame."""

        if (
            not outcome.ready
            and outcome.display_rust_result is None
            and outcome.display_crack_result is None
        ):
            self.reset()
            return
        observed_at = outcome.frame_read_completed_at
        if observed_at is None:
            observed_at = time.monotonic()
        if outcome.display_rust_result is not None:
            rust_result = outcome.display_rust_result
            if getattr(rust_result, "status", None) != "ready":
                raise ValueError("Realtime rust result is not ready.")
            self._rust_grades.append(
                (
                    float(observed_at),
                    highest_rust_grade(getattr(rust_result, "class_map", None)),
                )
            )
        if not self.no_crack and outcome.display_crack_result is not None:
            self._crack_ratios.append(
                (
                    float(observed_at),
                    validated_crack_ratio(outcome.display_crack_result),
                )
            )
        self.prune(now=float(observed_at))

    def decision(self) -> RealtimeControlDecision:
        # SIDE는 최근 녹 등급의 최댓값과 균열 비율의 최댓값을 사용한다. 학습 모델을 추가로 실행하는 판정이 아니다.
        # 균열 초과 또는 녹 2/3이면 브러시 OFF, 녹 1이면 단계 25, 녹 0이면 단계 12.5다.
        # 펌프는 균열 차단이 없고 녹 0일 때만 ON이므로 녹 1에서는 브러시 ON/펌프 OFF가 정상 결과다.
        if not self.ready:
            raise RuntimeError(
                "Four fresh inference results from each enabled model are required."
            )

        rust_grade = max(value for _, value in self._rust_grades)
        crack_ratio = (
            0.0
            if self.no_crack
            else max(value for _, value in self._crack_ratios)
        )
        crack_blocked = crack_ratio > CRACK_CONTROL_STOP_RATIO

        if crack_blocked or rust_grade >= 2:
            cleaner_duty_percent = 0.0
        elif rust_grade == 1:
            cleaner_duty_percent = CLEANER_STAGE_ONE_DUTY_PERCENT
        else:
            cleaner_duty_percent = CLEANER_BASE_DUTY_PERCENT

        pump_on = not crack_blocked and rust_grade == 0
        return RealtimeControlDecision(
            cleaner_duty_percent=cleaner_duty_percent,
            pump_on=pump_on,
            rust_grade=rust_grade,
            crack_ratio=crack_ratio,
        )


def side_outcome_requires_immediate_off(
    outcome: RealtimeInferenceOutcome | None,
    *,
    history_ready: bool,
) -> bool:
    """Reject newly observed SIDE hazards before a four-result window rebuilds."""

    # 새 위험을 받았을 때 4개 표본이 다시 찰 때까지 이전 ON을 유지하지 않도록 호출자에게 즉시 해제를 요청한다.
    # 균열 차단/녹 2 이상은 바로 True. 녹 1도 이력이 미완성인 동안 SIDE 전체를 끄고 준비 후 제한된 명령을 만든다.
    # FRONT의 아래 함수는 새 균열만 이 경로로 차단한다. 장애물은 위 FrontControlHistory의 단계 변경에 반영된다.
    if outcome is None:
        return False
    if (
        outcome.display_crack_result is not None
        and cleaning_blocked_by_crack(outcome.display_crack_result)
    ):
        return True
    if outcome.display_rust_result is None:
        return False
    rust_result = outcome.display_rust_result
    if getattr(rust_result, "status", None) != "ready":
        return True
    rust_grade = highest_rust_grade(getattr(rust_result, "class_map", None))
    # Grade 1 must at least stop water. Until a complete window can produce
    # that restricted command, conservatively stop the whole SIDE pair.
    return rust_grade >= 2 or (rust_grade == 1 and not history_ready)


def front_outcome_requires_immediate_off(
    outcome: TopRealtimeInferenceOutcome | None,
) -> bool:
    """Reject a newly observed FRONT crack without waiting for history."""

    return bool(
        outcome is not None
        and outcome.display_crack_result is not None
        and cleaning_blocked_by_crack(outcome.display_crack_result)
    )


def build_control_off_diagnostic(
    *,
    role: str,
    outcome: RealtimeInferenceOutcome | TopRealtimeInferenceOutcome | None,
    history_decision: RealtimeControlDecision | FrontControlDecision | None,
    history_ready: bool,
    camera_error: bool,
    camera_error_detail: str | None,
    hold_snapshot: RealtimeActuatorHoldSnapshot,
    hold: RealtimeActuatorDecisionHold,
    frame: CameraFrame | None,
    now: float,
) -> ControlOffDiagnostic:
    """Build diagnostics without consulting display caches or changing control."""

    new_crack_result = (
        None if outcome is None else outcome.display_crack_result
    )
    new_crack_blocked = False
    new_crack_pct = None
    if new_crack_result is not None:
        try:
            new_crack_ratio = validated_crack_ratio(new_crack_result)
            new_crack_pct = new_crack_ratio * 100.0
            new_crack_blocked = new_crack_ratio > CRACK_CONTROL_STOP_RATIO
        except (TypeError, ValueError):
            pass

    new_rust_grade = None
    if role == "side" and outcome is not None:
        new_rust_result = outcome.display_rust_result
        if new_rust_result is not None:
            try:
                if getattr(new_rust_result, "status", None) == "ready":
                    new_rust_grade = highest_rust_grade(
                        getattr(new_rust_result, "class_map", None)
                    )
            except (TypeError, ValueError):
                pass
    new_rust_forced_off = bool(
        new_rust_grade is not None
        and (
            new_rust_grade >= 2
            or (new_rust_grade == 1 and not history_ready)
        )
    )

    rolling_history_hazard = bool(
        history_decision is not None
        and history_decision.cleaner_duty_percent == 0.0
        and not history_decision.pump_on
    )
    history_crack_pct = (
        None
        if history_decision is None
        else history_decision.crack_ratio * 100.0
    )
    history_rust_grade = (
        history_decision.rust_grade
        if isinstance(history_decision, RealtimeControlDecision)
        else None
    )

    loss_started_at = hold.loss_started_at
    if loss_started_at is None:
        loss_started_at = hold_snapshot.loss_started_at
    valid_until = hold.valid_until
    if valid_until is None:
        valid_until = hold_snapshot.valid_until
    loss_age_ms = (
        None
        if loss_started_at is None
        else max(0.0, now - loss_started_at) * 1000.0
    )
    hold_remaining_ms = (
        None
        if valid_until is None
        else max(0.0, valid_until - now) * 1000.0
    )
    frame_age_ms = (
        None
        if frame is None
        else max(0.0, now - frame.read_completed_at) * 1000.0
    )
    watchdog_expired_at_off = bool(
        hold_snapshot.watchdog_expired
        or (
            hold_snapshot.has_decision
            and not hold_snapshot.loss_deadline_was_set
            and hold_snapshot.valid_until is not None
            and now > hold_snapshot.valid_until
        )
    )
    reason = select_control_off_reason(
        camera_error=camera_error,
        new_crack_blocked=new_crack_blocked,
        new_rust_grade=new_rust_grade,
        new_rust_forced_off=new_rust_forced_off,
        rolling_history_hazard=rolling_history_hazard,
        had_valid_decision=hold_snapshot.has_decision,
        loss_deadline_was_set=hold_snapshot.loss_deadline_was_set,
        watchdog_expired=watchdog_expired_at_off,
    )
    return ControlOffDiagnostic(
        reason=reason,
        monotonic_seconds=now,
        history_ready=history_ready,
        detail=(
            " ".join(str(camera_error_detail).split())
            if camera_error_detail is not None
            else None
        ),
        frame_sequence=None if frame is None else frame.sequence,
        frame_age_ms=frame_age_ms,
        new_crack_pct=new_crack_pct,
        new_crack_blocked=new_crack_blocked,
        new_rust_grade=new_rust_grade,
        history_crack_pct=history_crack_pct,
        history_rust_grade=history_rust_grade,
        loss_age_ms=loss_age_ms,
        hold_remaining_ms=hold_remaining_ms,
        pre_loss_deadline_set=hold_snapshot.loss_deadline_was_set,
        pre_valid_until=hold_snapshot.valid_until,
        watchdog_expired_at_off=watchdog_expired_at_off,
    )


def force_cleaning_safe_off(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
) -> None:
    """Fail closed once without spamming repeated OFF commands."""

    if cleaner_controller.is_on:
        cleaner_controller.force_off()
    if pump_controller.is_on:
        pump_controller.force_off()


def force_cleaning_pairs_safe_off(
    *pairs: tuple[CleanerController, PumpController],
) -> None:
    """Fail every supplied pair closed even if one shared-UART write fails."""

    first_error = None
    for cleaner_controller, pump_controller in pairs:
        for controller in (cleaner_controller, pump_controller):
            try:
                controller.force_off()
            except RuntimeError as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error
