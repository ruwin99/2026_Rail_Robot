import { useEffect, useState } from "react";
import { Timestamp } from "firebase/firestore";
import { isFresh, type Telemetry } from "./live-data";
import { listenLauncherStatus, listenMissionLaunch, requestMissionLaunch,
  type LauncherStatus, type LaunchRequest } from "./remote-launch";
import { listenRemoteCommand, requestRemoteCommand, REMOTE_REQUEST_TTL_MS,
  type RemoteAction, type RemoteCommand } from "./remote-commands";
import { selectVisibleCommand, type VisibleCommand } from "./remote-command-selection";
import type { Robot } from "./webrtc-signaling";

const resultMessages: Record<string, string> = {
  expired: "요청이 만료되어 실행하지 않았습니다.",
  invalid_command: "명령 형식이 맞지 않아 실행하지 않았습니다.",
  unauthorized: "운영 권한 또는 로봇의 원격 제어 설정을 확인하세요.",
  mission_not_ready: "시작 준비 상태가 아니어서 실행하지 않았습니다.",
  remote_stop_requested: "정지 요청이 우선되어 시작하지 않았습니다.",
  start_rejected: "임무 시작 서비스가 요청을 거절했습니다.",
  clock_not_synced: "Jetson 시계가 아직 동기화되지 않아 시작하지 않았습니다. 시간 동기화(NTP) 연결을 확인하세요.",
  start_unavailable: "임무 시작 서비스를 사용할 수 없습니다.",
  start_timeout: "시작 서비스 응답이 지연되었습니다. 현장 상태를 확인하세요.",
  start_outcome_unknown: "시작 결과를 확인할 수 없습니다. 현장 상태를 확인하세요.",
  estop_rejected: "STM이 정지 요청을 거절했습니다. 현장 상태를 즉시 확인하세요.",
  estop_unavailable: "STM 정지 서비스를 사용할 수 없습니다. 현장 상태를 즉시 확인하세요.",
  estop_unconfirmed: "정지 응답을 확인할 수 없습니다. 현장 상태를 즉시 확인하세요.",
};
const launchResultMessages: Record<string, string> = {
  already_running: "rail_mission이 이미 실행 중입니다.",
  launch_failed: "rail_mission 실행에 실패했습니다.",
  expired: "실행 요청이 만료되었습니다.",
};

export function RemoteMissionControls({ robot, uid, isAdmin, telemetry, now }: {
  robot?: Robot; uid: string | null; isAdmin: boolean; telemetry: Telemetry | null; now: number;
}) {
  const [submissions, setSubmissions] = useState<{ robotId: string; action: RemoteAction; sentAt: number }[]>([]);
  const [selected, setSelected] = useState<VisibleCommand | null>(null);
  const [received, setReceived] = useState<{ robotId: string; id: string; data: RemoteCommand | null } | null>(null);
  const [requestError, setRequestError] = useState<{ robotId: string; sentAt: number; message: string } | null>(null);
  const [launcherState, setLauncherState] = useState<{ robotId: string; status: LauncherStatus | null } | null>(null);
  const [launchSelected, setLaunchSelected] = useState<{ robotId: string; id: string; sentAt: number } | null>(null);
  const [launchReceived, setLaunchReceived] = useState<{ robotId: string; id: string; data: LaunchRequest | null } | null>(null);
  const [launchSubmitting, setLaunchSubmitting] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const robotId = robot?.id;
  const current = selected?.robotId === robotId ? selected : null;
  const command = current && received && received.robotId === robotId && received.id === current.id ? received.data : null;
  const submittingStart = submissions.some((item) => item.robotId === robotId && item.action === "START");
  const submittingStop = submissions.some((item) => item.robotId === robotId && item.action === "E_STOP");
  const allowed = !!(isAdmin && uid && robot?.remoteControlEnabled && robot.operatorUids?.includes(uid));
  const launcher = launcherState && launcherState.robotId === robotId ? launcherState.status : null;
  const currentLaunch = launchSelected && launchSelected.robotId === robotId ? launchSelected : null;
  const launchResult = currentLaunch && launchReceived && launchReceived.robotId === robotId
    && launchReceived.id === currentLaunch.id ? launchReceived.data : null;
  const launcherUpdatedAt = launcher?.updatedAt.toMillis() ?? 0;
  const launchAllowed = !!(isAdmin && uid && robot?.remoteLaunchEnabled
    && robot.operatorUids?.includes(uid));
  // 만료 판정은 같은 시계끼리 비교한다: 화면 시각은 요청을 보낸 브라우저 sentAt 기준,
  // launcher 상태 갱신 시각은 서버 createdAt 기준.
  const launchServerDeadline = launchResult ? launchResult.createdAt.toMillis() + REMOTE_REQUEST_TTL_MS : 0;
  const launchExpiredLocally = !!currentLaunch && now >= currentLaunch.sentAt + REMOTE_REQUEST_TTL_MS;
  const previousLaunchSettled = !currentLaunch
    || (launchResult?.state === "requested" && launchExpiredLocally
      && launcherUpdatedAt > launchServerDeadline)
    || ((launchResult?.state === "succeeded" || launchResult?.state === "failed")
      && launchResult.finishedAt instanceof Timestamp
      && launcherUpdatedAt > launchResult.finishedAt.toMillis());
  const launchReady = !!(launchAllowed && launcher
    && launcher.senderId === robot.senderId && launcher.ready && !launcher.missionRunning
    && isFresh(launcher.updatedAt.toDate().toISOString(), now, 15_000)
    && !launchSubmitting && previousLaunchSettled);
  const mission = telemetry?.mission;
  const missionFresh = !!(telemetry && isFresh(telemetry.updatedAt.toDate().toISOString(), now)
    && isFresh(mission?.observedAt, now));
  const startReady = allowed && missionFresh && mission?.phase === "CAPTURE_SCAN"
    && mission.completed === 0 && (!current || current.action !== "START"
      || command?.state === "failed");
  // E_STOP is idempotent at the STM bridge. Allow another request if an ACK is lost
  // or the operator still sees motion; throttle rapid repeat clicks in the browser.
  const stopReady = allowed && !submittingStop
    && (!current || current.action !== "E_STOP" || now - current.sentAt >= 2_000);

  useEffect(() => {
    if (!current) return;
    return listenRemoteCommand(current.robotId, current.id,
      (data) => setReceived({ robotId: current.robotId, id: current.id, data }),
      () => setRequestError({ robotId: current.robotId, sentAt: current.sentAt, message: "명령 결과를 읽지 못했습니다. 권한과 연결을 확인하세요." }));
  }, [current]);

  useEffect(() => {
    if (!robotId || !isAdmin || !uid) return;
    return listenLauncherStatus(robotId,
      (status) => setLauncherState({ robotId, status }),
      () => setLauncherState({ robotId, status: null }));
  }, [robotId, isAdmin, uid]);

  useEffect(() => {
    if (!currentLaunch) return;
    return listenMissionLaunch(currentLaunch.robotId, currentLaunch.id,
      (data) => setLaunchReceived({ robotId: currentLaunch.robotId, id: currentLaunch.id, data }),
      () => setLaunchError("실행 결과를 읽지 못했습니다. 연결 상태를 확인하세요."));
  }, [currentLaunch]);

  const launchMission = async () => {
    if (!robotId || !launchReady) return;
    if (!window.confirm("Jetson에서 rail_mission을 실행합니다. 이 요청은 로봇 주행 START를 보내지 않습니다. 계속하시겠습니까?")) return;
    setLaunchSubmitting(true);
    setLaunchError(null);
    try {
      const id = await requestMissionLaunch(robotId);
      setLaunchSelected({ robotId, id, sentAt: Date.now() });
    } catch {
      setLaunchError("실행 요청을 저장하지 못했습니다. 운영 권한과 연결 상태를 확인하세요.");
    } finally {
      setLaunchSubmitting(false);
    }
  };

  const send = async (action: RemoteAction) => {
    if (!robotId || !allowed || (action === "START" && !startReady) || (action === "E_STOP" && !stopReady)) return;
    if (action === "START" && !window.confirm("로봇이 실제로 움직일 수 있습니다. 현장 안전 상태와 물리 비상정지를 확인했습니까?")) return;
    const pending = { robotId, action, sentAt: Date.now() };
    setSubmissions((items) => [...items, pending]);
    setRequestError(null);
    try {
      const id = await requestRemoteCommand(robotId, action);
      setSelected((previous) => selectVisibleCommand(previous, { ...pending, id }));
    } catch {
      setRequestError({ robotId, sentAt: pending.sentAt, message: "명령 요청을 저장하지 못했습니다. 운영 권한과 네트워크를 확인하세요." });
    } finally {
      setSubmissions((items) => items.filter((item) => item !== pending));
    }
  };

  let status = "명령 기록 없음";
  if (current) {
    if (!command) status = "명령 결과 수신 대기";
    else if (command.state === "requested") status = now >= current.sentAt + REMOTE_REQUEST_TTL_MS
      ? "수신 시간 초과 · 실행 여부를 현장에서 확인하세요." : "요청 저장됨 · Jetson 수신 대기";
    else if (command.state === "accepted") status = now - current.sentAt > 70_000
      ? "처리 시간 초과 · 실행 결과를 현장에서 확인하세요." : "Jetson 수락 · ROS 2 처리 중";
    else if (command.state === "succeeded") status = current.action === "START"
      ? "UART START 전송 완료 · 실제 출발은 임무 상태로 확인하세요."
      : "STM이 정지 명령을 승인했습니다. 실제 정지 상태를 현장에서 확인하세요.";
    else status = resultMessages[command.resultCode || ""] || "명령을 완료하지 못했습니다. 현장 상태를 확인하세요.";
  }

  let launchMessage = launchError;
  if (!launchMessage && currentLaunch) {
    if (!launchResult) launchMessage = "rail_mission 실행 결과 수신 대기";
    else if (launchResult.state === "requested") launchMessage = launchExpiredLocally
      ? "실행 요청 만료 · Jetson 상태를 확인하세요." : "rail_mission 실행 요청 저장됨";
    else if (launchResult.state === "accepted") launchMessage = "Jetson이 rail_mission 실행 중";
    else if (launchResult.state === "succeeded") launchMessage = "rail_mission 프로세스 실행 · 임무 상태 확인 후 별도 시작 요청";
    else launchMessage = launchResultMessages[launchResult.resultCode || ""]
      || "rail_mission 실행 실패 · Jetson 로그를 확인하세요.";
  }

  return <div className="live-remote-controls" aria-label="로봇 원격 명령">
    <div className="live-remote-buttons">
      <button type="button" className="remote-launch" disabled={!launchReady}
        title={!launchReady ? "Jetson 실행 서비스 또는 원격 실행 권한 확인 필요" : undefined}
        onClick={() => void launchMission()}>rail_mission 실행</button>
      <button type="button" disabled={!startReady || submittingStart} onClick={() => void send("START")}>임무 시작 요청</button>
      <button type="button" className="remote-stop" disabled={!stopReady} onClick={() => void send("E_STOP")}>원격 정지 요청</button>
    </div>
    {launchMessage && <p role="status">{launchMessage}</p>}
    {allowed && !startReady && <p className="live-note">시작은 신선한 초기 검사 상태에서만 요청할 수 있습니다.</p>}
    <p role="status">{submittingStop || (submittingStart && current?.action !== "E_STOP") ? "명령 요청 전송 중"
      : requestError && requestError.robotId === robotId && requestError.sentAt >= (current?.sentAt ?? 0)
        ? requestError.message : status}</p>
  </div>;
}
