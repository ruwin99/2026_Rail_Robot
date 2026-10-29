import { useEffect, useState, type ReactNode } from "react";
import { getBlob, ref } from "firebase/storage";
import { storage } from "./firebase";
import { isFresh, listenCaptures, listenTelemetry, metric, type LiveCapture, type Telemetry } from "./live-data";
import { missionStages, missionStageState } from "./mission-stages";
import { resolveNodeHealth, type NodeHealthStatus } from "./node-health";

// [데이터 소비 부분] useLiveTelemetry는 Firestore 구독 수명, CaptureImage는 Storage 다운로드를 맡는다.
// 경로/스키마는 live-data.ts에서, 로봇 선택/로그인 조건은 LiveView.tsx에서 넘어온다.
function CaptureImage({ objectKey, label, robotId, captureId }: { objectKey: string; label: string; robotId: string; captureId: string }) {
  const [result, setResult] = useState<{ key: string; url?: string; error?: boolean } | null>(null);
  useEffect(() => {
    let active = true;
    let url: string | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const load = async () => {
      // [사진 범위 검사] 선택한 robot/capture 하위 object key만 허용하고 15MiB 한도로 받는다.
      // 이 검사는 서버 권한 검사를 대체하지 않는다. 오류 시 objectKey 범위와 Storage rules를 구별해 확인한다.
      try {
        if (!objectKey.startsWith(`robots/${robotId}/captures/${captureId}/`) || objectKey.includes("..")) throw new Error("scope");
        const blob = await getBlob(ref(storage, objectKey), 15 * 1024 * 1024);
        if (!active) return;
        url = URL.createObjectURL(blob);
        setResult({ key: objectKey, url });
      } catch {
        // [실패/재시도] 최초 시도 포함 최대 세 번이며 다음 대기는 3초, 6초다.
        // 화면/사진이 교체된 뒤 도착한 응답은 active로 무시하고 정리 시 Blob URL을 해제한다.
        if (!active) return;
        setResult({ key: objectKey, error: true });
        if (++attempts < 3) timer = setTimeout(load, 3_000 * attempts);
      }
    };
    void load();
    return () => { active = false; clearTimeout(timer); if (url) URL.revokeObjectURL(url); };
  }, [objectKey, robotId, captureId]);
  return <figure>{result?.key === objectKey && result.url
    ? <a href={result.url} target="_blank" rel="noreferrer"><img src={result.url} alt={label} /></a>
    : <div className="live-image-placeholder">{result?.key === objectKey && result.error ? "이미지를 불러오지 못했습니다" : "이미지 수신 중"}</div>}
    <figcaption>{label}</figcaption></figure>;
}

export function useLiveTelemetry(robotId?: string) {
  // [구독 묶음] 텔레메트리와 촬영 목록을 독립 구독/오류로 관리한다. 한쪽 실패가 다른쪽 성공을 지우지 않는다.
  const [telemetryState, setTelemetryState] = useState<{ robotId: string; data: Telemetry | null; error: string } | null>(null);
  const [captureState, setCaptureState] = useState<{ robotId: string; data: LiveCapture[]; error: string } | null>(null);
  const [now, setNow] = useState(0);
  useEffect(() => {
    let active = true;
    // 로봇이 바뀌면 각 수신값의 robotId 대조로 이전 데이터를 숨긴다. cleanup은 구독과 타이머를 해제한다.
    // now의 1초 갱신은 값의 나이 표시용이며 Firestore를 1초마다 새로 조회하는 코드가 아니다.
    if (!robotId) return;
    const stopTelemetry = listenTelemetry(robotId, (data) => { if (active) setTelemetryState({ robotId, data, error: "" }); },
      () => { if (active) setTelemetryState({ robotId, data: null, error: "상태 데이터 권한 또는 연결을 확인하세요." }); });
    const stopCaptures = listenCaptures(robotId, (data) => { if (active) setCaptureState({ robotId, data, error: "" }); },
      () => { if (active) setCaptureState({ robotId, data: [], error: "촬영 데이터 권한 또는 연결을 확인하세요." }); });
    const tick = () => setNow(Date.now());
    const firstTick = setTimeout(tick, 0);
    const timer = setInterval(tick, 1_000);
    return () => { active = false; stopTelemetry(); stopCaptures(); clearTimeout(firstTick); clearInterval(timer); };
  }, [robotId]);
  const telemetryCurrent = !!robotId && telemetryState?.robotId === robotId;
  const captureCurrent = !!robotId && captureState?.robotId === robotId;
  return { telemetry: telemetryCurrent ? telemetryState.data : null, captures: captureCurrent ? captureState.data : [], now,
    error: [telemetryCurrent ? telemetryState.error : "", captureCurrent ? captureState.error : ""].filter(Boolean).join(" ") };
}

export function MissionPanel({ telemetry, now, error, controls }: { telemetry: Telemetry | null; now: number; error: string; controls?: ReactNode }) {
  const fresh = telemetry && isFresh(telemetry.updatedAt.toDate().toISOString(), now);
  const mission = telemetry?.mission;
  const missionFresh = !!fresh && isFresh(mission?.observedAt, now);
  const ratio = mission && mission.total > 0 ? Math.max(0, Math.min(100, mission.completed / mission.total * 100)) : 0;
  const stage = missionStages.find((item) => item.phase === mission?.phase);
  const missionLabel = mission?.phase === "FAILED" ? "임무 오류" : stage?.label || "임무 상태 미수신";
  return <section className="live-quadrant live-mission-panel" aria-label="임무 상태와 작업 흐름">
    <header className="live-card-heading"><h3>임무 현황</h3><span className={missionFresh ? "live-badge" : "live-badge stale"}>{missionFresh ? "상태 수신 중" : "상태 수신 대기"}</span></header>
    {error && <p role="alert">{error}</p>}
    <div className="live-mission-summary"><strong>{missionLabel}</strong>
      {mission && <span>촬영 구간 진행률 · {mission.completed} / {mission.total}{!missionFresh ? " · 마지막 수신값" : ""}</span>}</div>
    <div className="live-progress" role="progressbar" aria-label="촬영 구간 진행률" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={mission ? Math.round(ratio) : undefined}><span style={{ width: `${ratio}%` }} /></div>
    <p className="live-note">촬영 승인 수입니다. 전체 임무 완료율은 아래 작업 흐름을 확인하세요.</p>
    <div className="live-status-columns"><div>{controls}
      <h4>남은 작업 <small>{mission && !missionFresh ? "마지막 수신값" : "현재 단계 기준"}</small></h4>{mission?.remainingTasks?.length
      ? <ul>{mission.remainingTasks.map((task, i) => <li key={i}>{task}</li>)}</ul>
      : <p>{mission && mission.completed >= mission.total && mission.total > 0 ? "남은 검사 구간 없음" : "임무 데이터 수신 후 표시"}</p>}</div>
      <div><h4>작업 흐름 <small>{missionFresh ? (mission?.phase === "FAILED" ? "임무 중단" : stage ? "현재 FSM 단계" : "상태 확인 불가") : "상태 수신 대기"}</small></h4>
        {mission?.phase === "FAILED" && missionFresh && <p role="alert">임무가 중단되었습니다. 마지막 진행 단계를 확인할 수 없습니다.</p>}
        <ol className="live-mission-route" aria-label="임무 작업 단계">{missionStages.map((item, index) => {
          const state = missionStageState(mission?.phase, missionFresh, index);
          const stateLabel = state === "done" ? "완료" : state === "active"
            ? (item.phase === "CAPTURE_SCAN" ? "현재 단계" : "진행 중") : state === "unknown" ? "확인 불가" : "대기";
          return <li key={item.phase} className={`live-mission-stop ${state}`} aria-current={state === "active" ? "step" : undefined}
            aria-label={`${item.label}: ${stateLabel}`}><span className="live-mission-dot" aria-hidden="true" />
            <span>{item.label}</span><small>{stateLabel}</small></li>;
        })}</ol>{missionFresh && mission?.phase === "CAPTURE_SCAN" &&
          <p className="live-note">초기 검사 단계는 로봇 출발 전에도 표시될 수 있습니다.</p>}</div></div>
  </section>;
}

export function CameraMetricsPanel({ telemetry, now }: { telemetry: Telemetry | null; now: number }) {
  return <section className="live-quadrant live-metrics-panel" aria-label="카메라별 처리 성능">
    <header className="live-card-heading"><h3>카메라 처리 성능</h3><span>FPS · 추론 시간</span></header>
    <div className="live-metrics-table"><table><thead><tr><th>카메라</th><th>프레임</th><th>녹</th><th>크랙</th><th>이물질</th></tr></thead>
      <tbody>{(["side", "front"] as const).map((role) => {
        const camera = telemetry?.cameras?.[role];
        const fresh = isFresh(camera?.observedAt, now);
        return <tr key={role}><th>{role.toUpperCase()}<small>{fresh ? "수신 중" : "수신 대기 / 지연"}</small></th>
          <td>{metric(camera?.fps, "FPS")}</td><td>{metric(camera?.rustMs, "ms")}</td>
          <td>{metric(camera?.crackMs, "ms")}</td><td>{metric(camera?.obstacleMs, "ms")}</td></tr>;
      })}</tbody></table></div>
    <p className="live-note">카메라 수집 FPS와 모델 추론 시간입니다. 미수신 값은 —로 표시합니다.</p>
  </section>;
}

const nodeStatusLabel: Record<NodeHealthStatus, string> = {
  ok: "정상 신호", observed: "피드백 수신", fault: "오류 감지", stale: "수신 지연", unknown: "확인 불가",
};

export function NodeHealthPanel({ telemetry, now }: { telemetry: Telemetry | null; now: number }) {
  const updatedAt = telemetry?.updatedAt?.toDate().toISOString() ?? null;
  const items = resolveNodeHealth(telemetry?.nodeHealth, updatedAt, now);
  const reporting = items.filter((item) => item.status === "ok").length;
  const faults = items.filter((item) => item.status === "fault").length;
  return <section className="live-quadrant live-node-health" aria-label="주요 신호와 장치 상태">
    <header className="live-card-heading"><h3>주요 신호 · 장치 상태</h3>
      <span>{telemetry?.nodeHealth ? `정상 신호 ${reporting} · 오류 ${faults}` : "상태 수신 대기"}</span></header>
    <ul className="live-node-health-list">{items.map((item) => <li key={item.id} className={`node-${item.status}`}>
      <span className="live-node-health-dot" aria-hidden="true" />
      <span className="live-node-health-name">{item.label}</span>
      <span className="live-node-health-state">{nodeStatusLabel[item.status]}</span>
    </li>)}</ul>
    <p className="live-note">엔코더 피드백은 양쪽 공통 응답으로, 개별 회전을 검증하지 않습니다. 클리너·펌프도 작동 피드백이 없습니다. 이 목록은 전체 ROS 노드 검사가 아닙니다.</p>
  </section>;
}

export function CapturePanel({ captures, robotId }: { captures: LiveCapture[]; robotId?: string }) {
  const [selectedId, setSelectedId] = useState("");
  const selected = captures.find((capture) => capture.captureId === selectedId) ?? captures[0];
  return <section className="live-quadrant live-captures" aria-label="실시간 촬영 및 분석 이미지">
    <header className="live-card-heading"><h3>촬영 · 분석 결과</h3><span>최근 {captures.length}건</span></header>
    {selected && robotId ? <>
      <label className="live-capture-select">촬영 선택 <select value={selectedId && captures.some((c) => c.captureId === selectedId) ? selectedId : ""}
        onChange={(event) => setSelectedId(event.target.value)}><option value="">최신 촬영 자동 표시</option>
        {captures.map((capture) => <option key={capture.captureId} value={capture.captureId}>
          {capture.cameraRole.toUpperCase()} · {capture.phase} #{capture.zone} · {new Date(capture.capturedAt).toLocaleTimeString("ko-KR")}
        </option>)}</select></label>
      <p className="live-note">{selected.runId} · {new Date(selected.capturedAt).toLocaleString("ko-KR")} · {selected.status === "processed" ? "분석 완료" : "촬영 완료 · 분석 대기"}</p>
      <div className="live-capture-images"><CaptureImage key={`${selected.captureId}-raw`} robotId={robotId} captureId={selected.captureId} objectKey={selected.rawObjectKey} label="촬영 원본" />
        {selected.processedObjectKeys.map((item) => <CaptureImage key={item.objectKey} robotId={robotId} captureId={selected.captureId} objectKey={item.objectKey} label={item.label} />)}
        {!selected.processedObjectKeys.length && <div className="live-image-placeholder">분석 이미지 수신 대기</div>}</div>
    </> : <div className="live-preview-empty">촬영 결과를 기다리고 있습니다.</div>}
  </section>;
}
