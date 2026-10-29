import { useEffect, useRef, useState } from "react";
import { onAuthStateChanged } from "firebase/auth";
import { getBlob, getMetadata, ref } from "firebase/storage";
import { auth, storage } from "./firebase";
import { createViewerPeer } from "./webrtc-peer";
import { firebaseSignalChannel, listenRobots, type Robot } from "./webrtc-signaling";
import { CameraMetricsPanel, CapturePanel, MissionPanel, NodeHealthPanel, useLiveTelemetry } from "./LiveTelemetry";
import { RemoteMissionControls } from "./RemoteMissionControls";
import { viewerIceServers } from "./turn-servers";

// [연결 수명 조정] 이 파일에서는 인증 변화 → 로봇 선택 → WebRTC 시도/재시도 → Storage 폴백을 추적한다.
// SDP·ICE 순서는 webrtc-peer.ts, Firestore 문서/권한은 webrtc-signaling.ts가 맡는다.
// 아래 주석은 데이터/연결 처리에 한정하며 JSX 배치와 스타일은 연결 프로토콜에 관여하지 않는다.
export function LiveView({ isAdmin }: { isAdmin: boolean }) {
  const [uid, setUid] = useState<string | null>(auth.currentUser?.uid ?? null);
  const [robotListState, setRobotListState] = useState<{ uid: string; robots: Robot[]; error: boolean; revision: number } | null>(null);
  const [robotId, setRobotId] = useState("");
  const [statusState, setStatusState] = useState<{ key: string; value: string } | null>(null);
  const [streamState, setStreamState] = useState<{ key: string; value: MediaStream | null } | null>(null);
  const [retry, setRetry] = useState(0);
  const [playingState, setPlayingState] = useState<{ key: string; value: boolean } | null>(null);
  const [playbackEpoch, setPlaybackEpoch] = useState(0);
  const [snapshotState, setSnapshotState] = useState<{ key: string; url: string; time: string; captured: boolean } | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const reconnectBarrier = useRef<Promise<void>>(Promise.resolve());
  const robots = isAdmin && uid && robotListState?.uid === uid ? robotListState.robots : [];
  const configError = !!(isAdmin && uid && robotListState?.uid === uid && robotListState.error);
  const robot = robots.find((item) => item.id === robotId) ?? robots[0];
  const connectionKey = `${uid ?? ""}:${robot?.id ?? ""}:${robot?.senderId ?? ""}:${robotListState?.revision ?? 0}:${retry}`;
  const snapshotKey = `${connectionKey}:${robot?.previewObjectKey ?? ""}:${playbackEpoch}`;
  const status = !globalThis.RTCPeerConnection ? "이 브라우저는 영상 연결을 지원하지 않습니다."
    : statusState?.key === connectionKey ? statusState.value : "연결 준비 중";
  const stream = streamState?.key === connectionKey ? streamState.value : null;
  const playing = playingState?.key === connectionKey && playingState.value;
  const snapshot = snapshotState?.key === snapshotKey ? snapshotState : null;
  const previewObjectKey = robot?.previewObjectKey;
  const liveRobotId = isAdmin && uid ? robot?.id : undefined;
  const live = useLiveTelemetry(liveRobotId);

  useEffect(() => onAuthStateChanged(auth, (user) => {
    setUid(user?.uid ?? null);
    setRobotListState(null);
    setStreamState(null);
    setPlayingState(null);
    setStatusState(null);
    setSnapshotState(null);
  }), []);
  // [인증/로봇 목록] 로그인 사용자 또는 관리자 상태가 바뀌면 목록 구독을 다시 만든다.
  // UI의 isAdmin 검사만으로 읽기 권한이 생기지 않는다. listenRobots의 viewerUids와 rules도 통과해야 한다.
  useEffect(() => {
    if (!isAdmin || !uid) return;
    return listenRobots((items) => setRobotListState((previous) => ({
      uid, robots: items, error: false, revision: (previous?.revision ?? 0) + 1,
    })), () => setRobotListState((previous) => ({
      uid, robots: [], error: true, revision: (previous?.revision ?? 0) + 1,
    })));
  }, [isAdmin, uid]);

  useEffect(() => {
    // [세션 시작 조건] 사용자·로봇·브라우저 지원 여부가 맞을 때만 연결한다.
    // robot ID/senderId/로그인/수동 retry가 바뀌면 이전 effect를 정리하고 새 연결 수명을 만든다.
    if (!isAdmin || !uid || !robot) return;
    if (!globalThis.RTCPeerConnection) return;
    let active = true;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let viewer: ReturnType<typeof createViewerPeer> | undefined;
    const connect = async () => {
      // effect 교체에서 등록한 이전 stop Promise를 기다린 뒤 로그인 UID를 다시 확인한다.
      // 이 barrier만으로 모든 자동 재시도의 원격 정리가 끝났다고 보장하지는 않는다.
      // 최종 동시 시청 제한은 Firestore 세션 생성 트랜잭션의 active 임대 검사가 담당한다.
      await reconnectBarrier.current;
      if (!active) return;
      if (!auth.currentUser || auth.currentUser.uid !== uid) return;
      // TURN 임시 계정(Cloud Function)을 받아 다른 네트워크에서도 중계로 연결한다.
      // 받지 못하면 STUN만 쓰며, 이때 같은 네트워크 밖의 시청은 실패할 수 있다.
      const ice = await viewerIceServers(robot.id);
      if (!active) return;
      viewer = createViewerPeer(firebaseSignalChannel(robot), {
        peer: new RTCPeerConnection({ iceServers: ice.servers }),
        stream: (next) => { if (active) { setStreamState({ key: connectionKey, value: next }); if (!next) setPlayingState({ key: connectionKey, value: false }); } },
        status: (next) => { if (active) setStatusState({ key: connectionKey, value: next }); },
        failed: (error) => {
          if (!active) return;
          setStreamState({ key: connectionKey, value: null });
          setPlayingState({ key: connectionKey, value: false });
          failures += 1;
          // [자동 재시도] 같은 effect 안에서 실패 1/2/3회에 5/10/20초 후 새 viewer를 만든다.
          // 4회째에는 자동 재시도를 멈춘다. 연결 성공만으로 failures를 0으로 되돌리는 코드는 없다.
          const detail = error instanceof Error ? error.message : "시청 권한 또는 네트워크를 확인하세요.";
          setStatusState({ key: connectionKey, value: `${detail} · ${failures < 4 ? "잠시 후 재연결합니다." : "다시 연결을 눌러주세요."}` });
          if (failures < 4) timer = setTimeout(() => void connect(), Math.min(30_000, 5_000 * 2 ** (failures - 1)));
        },
      });
      await viewer.start();
    };
    const leave = () => {
      // [화면 이탈/교체] 예약 재시도와 후속 상태 반영을 막고 viewer.stop을 다음 effect의 barrier로 남긴다.
      // pagehide의 비동기 Firestore 정리가 브라우저 종료 전에 완료된다고 보장하지는 않는다.
      active = false;
      clearTimeout(timer);
      if (viewer) reconnectBarrier.current = viewer.stop();
    };
    const restore = (event: PageTransitionEvent) => { if (event.persisted) setRetry((value) => value + 1); };
    void connect();
    window.addEventListener("pagehide", leave);
    window.addEventListener("pageshow", restore);
    return () => { leave(); window.removeEventListener("pagehide", leave); window.removeEventListener("pageshow", restore); };
  }, [isAdmin, uid, robot, retry, connectionKey]);

  useEffect(() => {
    const element = video.current;
    // [미디어 전달] 스트림을 video에 연결하는 단계다. play() 거절은 자동재생 정책 문제일 수 있어
    // WebRTC 협상 실패와 구별한다. playing은 아래 실제 재생 이벤트에서 갱신한다.
    if (!element) return;
    element.srcObject = stream;
    element.dataset.playbackTime = "0";
    element.dataset.videoFrames = "0";
    element.dataset.videoSize = "0x0";
    if (stream) void element.play().catch(() => setStatusState({ key: connectionKey, value: "영상 재생 버튼을 눌러주세요." }));
    return () => { element.srcObject = null; };
  }, [stream, connectionKey]);

  useEffect(() => {
    // [폴백 시작 조건] WebRTC 연결 여부 자체가 아니라 playing이 false일 때 5초 JPEG 조회를 사용한다.
    // 로봇별 previewObjectKey가 없거나 인증 조건을 만족하지 않으면 Storage 요청도 시작하지 않는다.
    if (!isAdmin || !uid || !previewObjectKey || playing) return;
    let active = true;
    let objectUrl: string | null = null;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const object = ref(storage, previewObjectKey);
        const metadata = await getMetadata(object);
        // [동일 버전 확인] 메타데이터 → 파일 → 메타데이터 순으로 읽어 업로드 교체 중인지 검사한다.
        // generation이 바뀌면 서로 다른 시각/파일 조합을 표시하지 않고 이번 갱신을 건너뛴다.
        const blob = await getBlob(object);
        const after = await getMetadata(object);
        if (!active) return;
        if (metadata.generation !== after.generation) return;
        const previous = objectUrl;
        objectUrl = URL.createObjectURL(blob);
        const captured = metadata.customMetadata?.capturedAt;
        // capturedAt이 있으면 촬영 시각, 없으면 Storage 갱신 시각이다. 최신 재생 시각으로 바꾸지 않는다.
        // 이전 Blob URL은 새 이미지 교체 후 해제해 반복 다운로드로 메모리가 계속 남지 않게 한다.
        setSnapshotState({ key: snapshotKey, url: objectUrl, time: captured || metadata.updated, captured: !!captured });
        if (previous) URL.revokeObjectURL(previous);
      } catch { /* 일시 실패 시 마지막 이미지와 원래 시각을 유지한다. 실패를 새 촬영으로 표시하지 않는다. */ }
      // 요청이 끝난 뒤 5초를 기다리므로 네트워크가 느려도 refresh 요청을 주기마다 중첩하지 않는다.
      finally { if (active) timer = setTimeout(refresh, 5_000); }
    };
    void refresh();
    return () => { active = false; clearTimeout(timer); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [isAdmin, uid, playing, previewObjectKey, snapshotKey]);

  const message = !isAdmin || !uid ? "관리자 로그인 후 시청할 수 있습니다."
    : configError ? "로봇 설정을 불러오지 못했습니다. 권한 또는 네트워크를 확인하세요."
    : !robot ? "연결할 로봇 등록 대기 · 등록된 시청 권한이 없습니다." : status;
  return <section className="live-operations" aria-label="Jetson 실시간 운영 화면">
    <section className="live-preview-panel live-quadrant" aria-label="Jetson 로봇 카메라 화면">
    <div className="section-heading">
      <div><p className="section-kicker">LIVE CAMERA</p><h3>로봇 GUI 카메라 화면</h3></div>
      {robots.length > 1 ? <label>로봇 <select value={robot?.id} onChange={(event) => setRobotId(event.target.value)}>
        {robots.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label> : null}
      <button type="button" disabled={!isAdmin || !uid || !robot} onClick={() => setRetry((value) => value + 1)}>다시 연결</button>
    </div>
    <p role="status">{playing ? "실시간 영상 재생 중" : message}</p>
    <video ref={video} autoPlay playsInline muted controls hidden={!stream}
      aria-label="로봇 실시간 영상" onPlaying={() => { setPlayingState({ key: connectionKey, value: true }); setPlaybackEpoch((value) => value + 1); }}
      onTimeUpdate={({ currentTarget: element }) => {
        // Expose media progress to browser diagnostics without exposing the stream or credentials.
        element.dataset.playbackTime = element.currentTime.toFixed(3);
        element.dataset.videoFrames = String(element.getVideoPlaybackQuality?.().totalVideoFrames ?? 0);
        element.dataset.videoSize = `${element.videoWidth}x${element.videoHeight}`;
      }}
      onWaiting={() => setPlayingState({ key: connectionKey, value: false })}
      onStalled={() => setPlayingState({ key: connectionKey, value: false })}
      onPause={() => setPlayingState({ key: connectionKey, value: false })} />
    {!playing && snapshot ? <div>
      <p>대체 스냅샷 · 실시간 영상이 아닙니다.<br />
        {snapshot.captured ? "촬영 시각" : "업로드 시각 (촬영 시각 미제공)"}: {new Date(snapshot.time).toLocaleString("ko-KR")}</p>
      <img src={snapshot.url} alt="마지막으로 업로드된 로봇 화면" />
    </div> : !stream ? <div className="live-preview-empty">수신된 카메라 화면이 없습니다.</div> : null}
    </section>
    <MissionPanel telemetry={live.telemetry} now={live.now} error={live.error}
      controls={<RemoteMissionControls robot={robot} uid={uid} isAdmin={isAdmin} telemetry={live.telemetry} now={live.now} />} />
    <NodeHealthPanel telemetry={live.telemetry} now={live.now} />
    <CapturePanel key={liveRobotId ?? "none"} captures={live.captures} robotId={liveRobotId} />
    <CameraMetricsPanel telemetry={live.telemetry} now={live.now} />
  </section>;
}
