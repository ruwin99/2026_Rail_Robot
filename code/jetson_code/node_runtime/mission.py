"""Inspection protocol state, independent of UART, camera, inference and ROS.

The coordinator performs I/O between prepare/complete calls. Capture acceptance
means every required raw image has been saved and its analysis job accepted; it
does not mean inference is finished. The section counter advances only after a
complete CAPTURE_OK write. A transition to REALTIME or COMPLETE is committed
only after the coordinator finishes the existing OFF and worker-drain checks.
"""
from __future__ import annotations

# 임무 순서만 관리하는 순수 FSM. UART·촬영·모델 실행은 ros/mission_node.py가 연결한다.
# 읽는 순서: 시작 승인 → 촬영 요청/승인 → prepare_transition → complete_transition.
# 여기의 mode는 Jetson의 상위 임무 상태이며, STM 내부의 모터 구동 상태와 구분한다.
# 트리거 문자열·목표 구간 수는 settings.py, initial/rescan 이름은 inspection_report.py에서 가져온다.
# 저장 실패나 UART 응답 누락의 원인은 이 FSM만으로 알 수 없으므로 호출자인 ros/mission_node.py도 확인한다.

from dataclasses import dataclass

from inspection_report import INITIAL_PHASE, RESCAN_PHASE

from .settings import *


@dataclass(frozen=True)
class CaptureRequest:
    phase: str
    phase_sequence: int


@dataclass(frozen=True)
class MissionTransition:
    trigger: str
    previous_mode: str
    next_mode: str


class MissionStateMachine:
    """Preserve the STM-led initial 4 / realtime / rescan 4 inspection order."""

    def __init__(self) -> None:
        # 초기 상태가 CAPTURE_SCAN이어도 시작된 것은 아니다. START 전송과 STARTED 수신을 따로 추적한다.
        self.mode = CAPTURE_SCAN_MODE
        self.initial_rail_section_count = 0
        self.rescan_rail_section_count = 0
        self.start_command_sent = False
        self.start_acknowledged = False
        self.pending_capture: CaptureRequest | None = None
        self.capture_accepted = False
        self.failure: str | None = None
        self._pending_transition: MissionTransition | None = None

    @staticmethod
    def validate_batch(messages: list[str]) -> None:
        # 같은 UART 수신 묶음 안의 모순을 먼저 검사한다. 실시간 시작/종료가 함께 오면 처리 기회가 없었던 것이다.
        # 촬영 요청도 한 묶음에 둘 이상이면 구간을 구분할 수 없으므로 하나로 합치지 않고 예외를 올린다.
        if (
            REALTIME_START_TRIGGER in messages
            and RESCAN_RETURN_START_TRIGGER in messages
        ):
            raise RuntimeError(
                f"Timing/protocol error: STM {REALTIME_START_TRIGGER} and "
                f"{RESCAN_RETURN_START_TRIGGER} arrived in the same UART read "
                "batch; no realtime frame could be inspected."
            )
        if messages.count(CAPTURE_TRIGGER) > 1:
            raise RuntimeError(
                f"Multiple {CAPTURE_TRIGGER} messages arrived before one "
                "frame could be saved. Stopping to avoid duplicate captures."
            )

    def _require_healthy(self) -> None:
        # failure는 자동으로 지우지 않는 오류 기록이다. 후속 이벤트로 촬영 카운터가 다시 진행하지 못하게 한다.
        # 이 예외가 반복되면 최초 실패 사유와 mission 노드의 /runtime/fault를 먼저 확인한다.
        if self.failure is not None:
            raise RuntimeError(f"Mission stopped after capture failure: {self.failure}")

    def mark_start_sent(self) -> None:
        """Called after the entire START command has been written to UART."""
        self._require_healthy()
        if (
            self.start_command_sent
            or self.mode != CAPTURE_SCAN_MODE
            or self.initial_rail_section_count != 0
        ):
            raise RuntimeError("START is only allowed once before inspection.")
        self.start_command_sent = True

    def acknowledge_started(self) -> bool:
        """Unsolicited and duplicate STARTED messages leave state unchanged."""
        # START를 보낸 적이 없거나 이미 승인된 경우 False로 끝내 중복 시작을 막는다.
        # 시작이 멈춘 경우 start_command_sent와 start_acknowledged를 비교해 전송 단계/응답 단계를 구별한다.
        self._require_healthy()
        if not self.start_command_sent or self.start_acknowledged:
            return False
        self.start_acknowledged = True
        return True

    def require_started(self, message: str) -> None:
        self._require_healthy()
        if message in MOTION_TRIGGERS and not self.start_acknowledged:
            raise RuntimeError(
                f"Protocol error: STM {message} arrived before STARTED "
                "acknowledgement; possible old firmware or automatic start."
            )

    def begin_capture(self) -> CaptureRequest:
        # 같은 구간의 중복 저장/ACK를 막도록 촬영 또는 상태 전이를 한 번에 하나만 허용한다.
        self.require_started(CAPTURE_TRIGGER)
        if self.pending_capture is not None or self._pending_transition is not None:
            raise RuntimeError("A capture or mode transition is already pending.")
        if self.mode == CAPTURE_SCAN_MODE:
            # 현재 모드로 저장 단계와 다음 구간 번호를 정한다. count + 1은 아직 확정 카운터가 아닌 요청 번호다.
            # RETURN/REALTIME에서 촬영 이벤트가 오면 저장을 강행하지 않고 아래 프로토콜 오류로 처리한다.
            phase, count = INITIAL_PHASE, self.initial_rail_section_count
        elif self.mode == RESCAN_MODE:
            phase, count = RESCAN_PHASE, self.rescan_rail_section_count
        else:
            raise RuntimeError(
                f"Protocol error: STM {CAPTURE_TRIGGER} arrived in "
                f"{self.mode} mode; no capture was acknowledged."
            )
        if count >= AUTOMATIC_RAIL_SECTION_TARGET:
            raise RuntimeError(
                f"Protocol error: extra STM {CAPTURE_TRIGGER} after "
                f"{AUTOMATIC_RAIL_SECTION_TARGET} {phase} Rail Sections."
            )
        request = CaptureRequest(phase, count + 1)
        self.pending_capture = request
        self.capture_accepted = False
        return request

    def _require_pending_capture(self, request: CaptureRequest) -> None:
        # 값이 같은 새 객체도 허용하지 않고 현재 보류 중인 요청 객체인지 확인한다.
        # 이전 비동기 응답이 늦게 도착해 다음 촬영의 상태를 바꾸는 것을 막는 경계다.
        self._require_healthy()
        if request is not self.pending_capture:
            raise RuntimeError("Capture response does not match the pending request.")

    def accept_capture(self, request: CaptureRequest) -> None:
        """Call only after all required cameras are saved and queued."""
        # 저장/접수 성공만 표시하고 pending_capture는 남긴다. 이어지는 UART 전송이 실패할 수도 있기 때문이다.
        # capture_accepted=True인데 구간 수가 그대로라면 mission 노드의 CAPTURE_OK 완료 콜백을 확인한다.
        self._require_pending_capture(request)
        if self.capture_accepted:
            raise RuntimeError("The pending capture was already accepted.")
        self.capture_accepted = True

    def acknowledge_capture(self, request: CaptureRequest) -> None:
        """Commit one section after CAPTURE_OK was completely written."""
        # 원본 저장·분석 큐 접수 후 UART ACK까지 성공해야 구간 수가 증가한다. 추론 완료 수가 아니다.
        self._require_pending_capture(request)
        if not self.capture_accepted:
            raise RuntimeError("CAPTURE_OK requires an accepted raw capture.")
        if request.phase == INITIAL_PHASE:
            self.initial_rail_section_count += 1
        else:
            self.rescan_rail_section_count += 1
        self.pending_capture = None
        self.capture_accepted = False

    def fail_capture(self, request: CaptureRequest, reason: str) -> None:
        """Latch a save/queue/write failure without advancing or retrying ACK."""
        self._require_pending_capture(request)
        self.capture_accepted = False
        self.failure = str(reason) or "capture failed"

    def prepare_transition(self, trigger: str) -> MissionTransition | None:
        """Validate STM phase order; return None for legacy ignored triggers."""
        # STM 이벤트 순서를 검증해 전이 후보만 만든다. 외부 OFF/drain 작업 중에는 기존 mode를 유지한다.
        self.require_started(trigger)
        if self.pending_capture is not None or self._pending_transition is not None:
            raise RuntimeError("A capture or mode transition is already pending.")
        if trigger == RETURN_START_TRIGGER:
            # 정상 순서: initial 촬영 목표 달성 → RETURN → REALTIME → RESCAN_RETURN → RESCAN → 완료.
            # 초기 RETURN/REALTIME의 부적절한 순서는 기존 호환 동작대로 None을 반환해 무시한다.
            # 아래 재검사 단계의 순서 위반은 예외다. 모든 잘못된 이벤트가 같은 방식으로 처리되지는 않는다.
            if (
                self.mode != CAPTURE_SCAN_MODE
                or self.initial_rail_section_count != AUTOMATIC_RAIL_SECTION_TARGET
            ):
                return None
            next_mode = RETURN_MODE
        elif trigger == REALTIME_START_TRIGGER:
            if self.mode != RETURN_MODE:
                return None
            next_mode = REALTIME_MODE
        elif trigger == RESCAN_RETURN_START_TRIGGER:
            if self.mode != REALTIME_MODE:
                raise RuntimeError(
                    f"Protocol error: STM {trigger} arrived in {self.mode} mode."
                )
            next_mode = RESCAN_RETURN_MODE
        elif trigger == RESCAN_START_TRIGGER:
            if self.mode != RESCAN_RETURN_MODE or self.rescan_rail_section_count != 0:
                raise RuntimeError(
                    f"Protocol error: STM {trigger} arrived in {self.mode} mode "
                    f"with {self.rescan_rail_section_count} rescan Rail Sections."
                )
            next_mode = RESCAN_MODE
        elif trigger == RESCAN_DONE_TRIGGER:
            if (
                self.mode != RESCAN_MODE
                or self.rescan_rail_section_count != AUTOMATIC_RAIL_SECTION_TARGET
            ):
                raise RuntimeError(
                    f"Protocol error: STM {trigger} arrived in {self.mode} mode "
                    f"with {self.rescan_rail_section_count}/"
                    f"{AUTOMATIC_RAIL_SECTION_TARGET} rescan Rail Sections."
                )
            next_mode = RESCAN_DONE_MODE
        elif trigger == DONE_TRIGGER:
            if self.mode != RESCAN_DONE_MODE:
                raise RuntimeError(
                    f"Protocol error: STM {trigger} arrived in {self.mode} mode "
                    f"before {RESCAN_DONE_TRIGGER}."
                )
            next_mode = COMPLETE_MODE
        else:
            return None
        transition = MissionTransition(trigger, self.mode, next_mode)
        self._pending_transition = transition
        return transition

    def complete_transition(self, transition: MissionTransition) -> None:
        """Commit only after required safe-OFF/drain work has succeeded."""
        # 준비했던 전이 객체와 이전 모드가 모두 일치해야 mode를 바꾼다. 늦은 콜백의 잘못된 확정을 방지한다.
        # mode가 바뀌지 않으면 _pending_transition과 호출자의 OFF/drain 응답을 함께 확인한다.
        self._require_healthy()
        if (
            transition is not self._pending_transition
            or self.mode != transition.previous_mode
        ):
            raise RuntimeError("Mode transition does not match the pending request.")
        self.mode = transition.next_mode
        self._pending_transition = None
