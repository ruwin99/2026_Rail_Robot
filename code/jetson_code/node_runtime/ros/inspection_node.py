"""Capture acceptance and inspection persistence, with no STM or GUI ownership."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import String
from rail_robot_interfaces.srv import CapturePair, CaptureInspection, DrainInspection

from dashboard_export import finalize_dashboard_run
from inspection_report import INITIAL_PHASE, RESCAN_PHASE
from ..inspection import (
    CaptureAnalysisWorker, queue_dual_capture_for_analysis,
    validate_db_test_dashboard_manifest,
)
from ..settings import (
    CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS,
    AUTOMATIC_RAIL_SECTION_TARGET, CAPTURE_TRIGGER, REALTIME_INPUT_MAX_AGE_SECONDS,
)
from .common import qos_events, qos_state, stamp_seconds
from .images import message_to_frame
from .deployment import (create_inspection_resources, create_inspection_obstacle_factory,
                         read_parameters, close_detectors)


# mission의 구간 촬영 요청을 받아 원본 저장과 분석 작업 등록을 책임진다.
# 실시간 제어용 vision 노드와 별도로 촬영 검사 모델·분석 작업자·검사 보고서를 소유한다.
# 저장·분석 큐·사진별 처리는 node_runtime/inspection.py, 모델 생성은 ros/deployment.py를 본다.
# 최종 보고서 확정은 dashboard_export.py, 선택적 사진 전송은 node_runtime/live_capture.py로 이어진다.
class InspectionNode(Node):
    """Own capture engines and workbook; accept only durable, paired queue jobs.

    Default construction reads validated deployment parameters; tests may inject
    engines/workbooks. A service acceptance is not inference completion.
    """
    def __init__(self, detector=None, crack_detector=None, workbook=None,
                 output_directory: Path | None = None, *,
                 obstacle_detector_factory=None,
                 node_name: str = "inspection", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self._lock = threading.RLock()
        self._busy = False
        self._closed = False
        self._stopping = False
        self._pending_pair = None
        self._pending_pair_deadline = None
        self._failure = None
        self._fault_sent = False
        self._accepted = {}
        self._counts = {INITIAL_PHASE: 0, RESCAN_PHASE: 0}
        self._manifest = None
        self._failure_finalized = False
        self._detectors = ()
        self._worker = None
        self._workbook = None
        self._live_captures = None
        try:
            # 기본 실행에서는 검사 모델·보고서·출력 디렉터리를 함께 준비한다.
            # readiness가 실패하면 _worker 생성 여부와 deployment의 모델 초기화 예외를 먼저 본다.
            if detector is None and crack_detector is None and workbook is None:
                detector, crack_detector, workbook, generated_output = create_inspection_resources(self)
                self._detectors = (detector, crack_detector)
                obstacle_detector_factory = create_inspection_obstacle_factory(self)
                if output_directory is None:
                    output_directory = generated_output
            elif detector is None or workbook is None:
                raise ValueError("Supply capture detector and workbook together, or omit all resources.")
            self._detectors = (detector, crack_detector)
            if output_directory is None:
                output_directory = read_parameters(self, ("output_directory",))["output_directory"]
            self._output = Path(output_directory).expanduser().resolve()
            self._workbook = workbook
            dashboard_live = (self.get_parameter("dashboard_live").value
                              if self.has_parameter("dashboard_live") else
                              self.declare_parameter("dashboard_live", False).value)
            if not isinstance(dashboard_live, bool):
                raise ValueError("dashboard_live must be a boolean.")
            # 사진 업로드는 선택 기능이다. 설정 실패 시에도 로컬 검사 경로는 유지한다.
            # _live_captures가 None이면 검사 자체는 계속하지만 이 실행에서는 업로드가 비활성이다.
            # 원본/분석 사진 업로드의 관찰자는 아래 분석 작업자에 전달되며 ROS 응답과 분리된다.
            if dashboard_live:
                from ..live_dashboard import LiveDashboardClient
                from ..live_capture import LiveCaptureOutbox

                try:
                    live_client = LiveDashboardClient.from_environment()
                    live_client.require_capture_storage()
                    run_key = hashlib.sha256(str(Path(workbook.path).resolve()).encode("utf-8")).hexdigest()[:24]
                    self._live_captures = LiveCaptureOutbox(
                        live_client, "run_" + run_key,
                        "~/.local/share/rail_robot/live-capture-outbox",
                        report_error=self.get_logger().warning)
                    self._live_captures.start()
                except Exception as exc:
                    self._live_captures = None
                    self.get_logger().warning(
                        f"Live capture output unavailable ({type(exc).__name__}); "
                        "local inspection continues. Check private dashboard config and restart to enable uploads.")
            self._worker = CaptureAnalysisWorker(
                detector, workbook, self._output, crack_detector=crack_detector,
                obstacle_detector_factory=obstacle_detector_factory,
                live_capture_observer=self._live_captures)
            self._worker.start()
            self._callbacks = ReentrantCallbackGroup()
            self._pair_client = self.create_client(
                CapturePair, "camera/capture_pair", callback_group=self._callbacks)
            self._fault_pub = self.create_publisher(String, "runtime/fault", qos_events)
            self._manifest_pub = self.create_publisher(
                String, "inspection/finalized", qos_state)
            self.create_subscription(String, "runtime/fault", self._on_fault,
                                     qos_events, callback_group=self._callbacks)
            self.create_service(CaptureInspection, "inspection/capture",
                                self._capture, callback_group=self._callbacks)
            self.create_service(DrainInspection, "inspection/drain",
                                self._drain, callback_group=self._callbacks)
            self.create_timer(0.05, self._check_worker, callback_group=self._callbacks)
        except Exception:
            self.destroy_node()
            raise

    def _fail(self, error):
        # 첫 실패 원인을 _failure에 보존하고 _fault_sent로 동일 노드의 반복 발행을 막는다.
        # 다른 노드의 runtime/fault도 _on_fault를 거쳐 여기로 오므로 원인 노드 접두사를 확인한다.
        with self._lock:
            if self._failure is None:
                self._failure = str(error)
            if not self._fault_sent and not self._closed:
                message = String()
                message.data = "inspection: " + self._failure
                self._fault_pub.publish(message)
                self._fault_sent = True

    def _check_worker(self):
        if self._closed or self._stopping:
            return
        self._expire_pending_pair()
        if self._worker is None:
            return
        failure = self._worker.failure_message()
        if failure is not None:
            self._fail(failure)
        self._finalize_failure_if_idle()

    def _expire_pending_pair(self):
        # 카메라 쌍 요청의 2초 시간 초과는 상시 50ms 타이머가 확인한다. 요청마다 타이머를 만들고
        # 지우면 MultiThreadedExecutor가 대기 목록에 올린 타이머를 다른 스레드가 파괴해
        # spin_once가 InvalidHandle로 노드를 종료시킬 수 있다(2026-10-05 실기 임무 중단 원인).
        with self._lock:
            future, deadline = self._pending_pair, self._pending_pair_deadline
        if future is not None and deadline is not None and not future.done() and time.monotonic() >= deadline:
            future.cancel()

    def _on_fault(self, message):
        if message.data:
            self._fail(message.data)

    def _finalize_failure_if_idle(self):
        # 분석 작업자나 drain이 보고서를 쓰는 동안에는 실패 보고서를 덮어쓰지 않는다.
        # _busy와 worker의 유휴 상태를 함께 확인하고 한 번만 실패 상태로 확정한다.
        with self._lock:
            if (self._worker is None or self._failure is None or self._failure_finalized or self._busy or
                    self._manifest is not None or not self._worker.wait_until_idle(0)):
                return
            try:
                finalize_dashboard_run(
                    workbook=self._workbook, output_directory=self._output,
                    status="failed", failure_reason=self._failure)
            except Exception as exc:
                self.get_logger().error("Could not finalize failed inspection: " + str(exc))
            self._failure_finalized = True

    def _validate_request(self, request):
        # 요청은 request_id, CAMERA_CAPTURE 트리거, initial/rescan 단계와 구간 번호가 필요하다.
        # 여기서는 형식·전체 상태를 검사하고, 중복 ID 및 구간 순서 검사는 _capture에서 이어간다.
        if self._closed or self._stopping:
            raise RuntimeError("Inspection node is closed.")
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if not request.request_id or request.trigger != CAPTURE_TRIGGER:
            raise ValueError("Capture requires an ID and CAMERA_CAPTURE trigger.")
        if request.phase not in self._counts:
            raise ValueError("Capture phase must be initial or rescan.")
        if not 1 <= request.phase_sequence <= AUTOMATIC_RAIL_SECTION_TARGET:
            raise ValueError(
                f"Capture section is outside the {AUTOMATIC_RAIL_SECTION_TARGET}+{AUTOMATIC_RAIL_SECTION_TARGET} mission.")
        if self._manifest is not None:
            raise RuntimeError("Inspection is already finalized.")

    async def _capture(self, request, response):
        claimed = False
        future = None
        try:
            with self._lock:
                self._validate_request(request)
                signature = (request.phase, request.phase_sequence, request.trigger)
                # 같은 요청의 재전송은 이미 수락한 결과를 돌려주어 구간을 중복 촬영하지 않는다.
                # 단, ID가 같아도 phase/phase_sequence/trigger가 달라지면 재사용 오류로 거절한다.
                if request.request_id in self._accepted:
                    if self._accepted[request.request_id] != signature:
                        raise ValueError("Capture ID was reused for a different section.")
                    response.accepted = True
                    return response
                # _busy는 촬영 수락 또는 drain 한 건이 진행 중이라는 표시다.
                # _counts는 분석 완료 수가 아닌 단계별 수락 수이며 다음 구간만 받도록 제한한다.
                if self._busy:
                    raise RuntimeError("Another inspection operation is pending.")
                if request.phase_sequence != self._counts[request.phase] + 1:
                    raise ValueError("Capture sections must be accepted in order.")
                if (request.phase == RESCAN_PHASE and
                        self._counts[INITIAL_PHASE] != AUTOMATIC_RAIL_SECTION_TARGET):
                    raise ValueError("Rescan requires all initial sections.")
                self._busy = True
                claimed = True
            if not self._pair_client.service_is_ready():
                raise RuntimeError("Camera capture service is unavailable.")
            pair_request = CapturePair.Request()
            pair_request.request_id = request.request_id
            requested_ros = self.get_clock().now().nanoseconds / 1e9
            future = self._pair_client.call_async(pair_request)
            with self._lock:
                if self._closed or self._stopping:
                    future.cancel()
                    raise RuntimeError("Inspection node is closed.")
                self._pending_pair = future
                self._pending_pair_deadline = time.monotonic() + 2.0
            # await 동안 다른 콜백이 카메라 응답과 시간 초과를 처리할 수 있도록 구성되어 있다.
            # 2초 안에 응답이 없으면 _check_worker 타이머가 future를 취소한다. _pending_pair는 종료 시 취소할 대상이며,
            # 이 취소만으로 카메라 쪽에서 이미 시작한 읽기 작업까지 중단되었다고 보지는 않는다.
            pair = await future
            if future.cancelled() or pair is None:
                raise TimeoutError("Stopped camera pair request timed out.")
            if not pair.success:
                raise RuntimeError(pair.error or "Stopped camera pair failed.")
            if pair.side.camera_role != "side" or pair.top.camera_role != "top":
                raise ValueError("Capture response must contain SIDE and TOP.")
            now_mono = time.monotonic()
            now_ros = self.get_clock().now().nanoseconds / 1e9
            # 정지 촬영 원본은 카메라가 이 요청을 받은 뒤 안정화 대기 후 새로 읽은 프레임이다.
            # 실시간 입력 기준(0.2초)을 응답 도착 시각에 그대로 대면, 분석 작업자가 엔진을 올리는 등
            # CPU가 바쁠 때 2.7MB 두 장의 전달·역직렬화 지연만으로 정상 원본이 stale fault가 된다
            # (2026-10-05 실기 2구간째 중단). 요청 시작 전 0.2초보다 오래된 원본만 거절한다.
            pair_max_age = max(REALTIME_INPUT_MAX_AGE_SECONDS,
                               now_ros - requested_ros + REALTIME_INPUT_MAX_AGE_SECONDS)
            side = message_to_frame(pair.side, now_ros, now_mono, pair_max_age)
            top = message_to_frame(pair.top, now_ros, now_mono, pair_max_age)
            with self._lock:
                if self._closed or self._stopping:
                    raise RuntimeError("Inspection node is stopping.")
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                # 두 원본을 저장하고 분석 큐에 함께 넣은 뒤에만 accepted를 반환한다.
                # 분석·보고서 작성·클라우드 업로드 완료를 의미하는 응답은 아니다.
                # 큐 등록 실패 시 아래 _accepted/_counts 갱신 전 예외가 나므로 수락 성공으로 세지 않는다.
                queue_dual_capture_for_analysis(
                    side.frame, top.frame, request.phase, request.phase_sequence,
                    request.trigger, self._worker,
                    captured_at=datetime.fromtimestamp(
                        stamp_seconds(pair.side.header.stamp), tz=timezone.utc),
                    side_frame_read_completed_at=side.read_completed_at,
                    top_frame_read_completed_at=top.read_completed_at,
                    output_directory=self._output,
                )
                self._accepted[request.request_id] = signature
                self._counts[request.phase] += 1
                response.accepted = True
        except Exception as exc:
            response.accepted, response.error = False, str(exc)
            self._fail(exc)
        finally:
            # 자신이 _busy를 잡았을 때만 해제하므로 거절된 중복 호출이 다른 진행 중 작업의 상태를 풀어버리지 않는다.
            if claimed:
                with self._lock:
                    self._busy = False
                    self._pending_pair = None
                    self._pending_pair_deadline = None
                if self._closed:
                    self._finalize_failure_if_idle()
        return response

    def _drain(self, request, response):
        claimed = False
        try:
            with self._lock:
                if self._closed or self._stopping:
                    raise RuntimeError("Inspection node is stopping.")
                if self._busy:
                    raise RuntimeError("Cannot drain during capture acceptance.")
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                self._busy = True
                claimed = True
            # 상태 전환 시 밀린 분석의 완료를 확인하는 별도 단계다.
            # finalize 요청이면 초기 검사와 재검사 구간 수만큼의 쌍을 확인한 후 최종 보고서를 확정한다.
            # 대기가 끝나도 worker.failure_message가 있으면 성공 처리하지 않는다.
            # 성공 시 inspection/finalized는 보고서 묶음의 manifest 경로이며 업로드 완료 알림은 아니다.
            if not self._worker.wait_until_idle(CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS):
                raise TimeoutError("Inspection analysis did not drain within 5 seconds.")
            with self._lock:
                if self._closed or self._stopping:
                    raise RuntimeError("Inspection node is stopping.")
                failure = self._failure or self._worker.failure_message()
                if failure is not None:
                    raise RuntimeError(failure)
                if request.finalize and self._manifest is None:
                    if any(n != AUTOMATIC_RAIL_SECTION_TARGET for n in self._counts.values()):
                        raise RuntimeError(
                            f"Finalization requires all {AUTOMATIC_RAIL_SECTION_TARGET}+"
                            f"{AUTOMATIC_RAIL_SECTION_TARGET} paired captures.")
                    manifest = finalize_dashboard_run(
                        workbook=self._workbook, output_directory=self._output,
                        status="complete")
                    if manifest is None:
                        raise RuntimeError("Inspection has no finalized manifest.")
                    validate_db_test_dashboard_manifest(manifest)
                    self._manifest = manifest
                    message = String()
                    message.data = str(manifest.resolve())
                    self._manifest_pub.publish(message)
                response.success = True
                response.manifest_path = str(self._manifest or "")
        except Exception as exc:
            response.success, response.error = False, str(exc)
            self._fail(exc)
        finally:
            if claimed:
                with self._lock:
                    self._busy = False
                if self._closed:
                    self._finalize_failure_if_idle()
        return response

    def prepare_shutdown(self):
        with self._lock:
            self._stopping = True
            if self._pending_pair is not None:
                self._pending_pair.cancel()

    def destroy_node(self):
        # 미완료 임무는 실패로 기록하고 대기 촬영을 취소한 뒤 작업자 종료를 기다린다.
        # 작업자 종료가 실패한 경우 아직 사용 중일 수 있는 모델을 바로 close하지 않고 오류를 남긴다.
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            if self._manifest is None:
                self._fail("Inspection stopped before mission completion.")
            if self._pending_pair is not None:
                self._pending_pair.cancel()
        if self._worker is None:
            if self._live_captures is not None:
                self._live_captures.close()
            close_detectors(self._detectors)
            return super().destroy_node()
        failure = self._worker.shutdown()
        if self._live_captures is not None:
            self._live_captures.close()
        self._finalize_failure_if_idle()
        if failure is None:
            for detector in self._detectors:
                if detector is not None:
                    try:
                        detector.close()
                    except Exception as exc:
                        self.get_logger().error("Could not close inspection detector: " + str(exc))
        else:
            self.get_logger().error(failure)
        return super().destroy_node()

    def readiness(self):
        with self._lock:
            if self._closed or self._stopping or self._failure:
                return False, self._failure or "Inspection node is closed."
            if self._worker is None:
                return False, "Inspection resources are not initialized."
            failure = self._worker.failure_message()
            return (False, failure) if failure else (True, "Capture engines and analysis worker initialized.")
