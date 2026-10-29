// Browser media lifecycle, independent of Firebase for deterministic tests.
// [브라우저 WebRTC 처리] start → offer 생성/세션 등록 → answer 적용 → ICE 적용 → 영상 수신 순서다.
// Firestore 경로/권한/문서 저장은 webrtc-signaling.ts, 재시도와 스냅샷 선택은 LiveView.tsx를 확인한다.
// 여기의 channel은 연결 정보만 교환하며 영상 바이트는 RTCPeerConnection의 미디어 경로로 들어온다.
export type Candidate = RTCIceCandidateInit & { endOfCandidates?: boolean };
export interface SignalChannel {
  create(offer: RTCSessionDescriptionInit): Promise<void>;
  candidate(value: Candidate): Promise<void>;
  heartbeat(): Promise<void>;
  close(): Promise<void>;
  listen(answer: (sdp: RTCSessionDescriptionInit) => void,
    candidate: (value: Candidate) => void, ended: (reason?: string) => void,
    failed: (error: unknown) => void): () => void;
}

function sessionEndMessage(reason?: string): string {
  // 종료 사유를 합치지 않는다. 협상 제한시간, heartbeat 누락, 잠금 만료, 세션 만료는 각각 다른 원인이다.
  // http_403도 항상 만료를 뜻하지 않으므로 권한/문서 상태/늦은 쓰기를 시그널링 쪽에서 구별해야 한다.
  switch (reason) {
    case "negotiation_timeout": return "영상 연결 협상 시간이 초과되었습니다.";
    case "heartbeat_timeout": return "시청 연결 확인이 지연되어 영상을 종료했습니다.";
    case "lock_lease_expired": return "시청 연결 유지 시간이 만료되었습니다.";
    case "session_expired": return "시청 세션의 유효 시간이 만료되었습니다.";
    case "http_403": return "영상 시청 권한 또는 세션 유효성을 확인하세요.";
    default: return "송신자가 연결을 종료했습니다.";
  }
}

export function createViewerPeer(channel: SignalChannel, options: {
  peer: RTCPeerConnection;
  stream: (stream: MediaStream | null) => void;
  status: (status: string) => void;
  failed: (error: unknown) => void;
}) {
  const pc = options.peer;
  let disposed = false;
  let created = false;
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let disconnectTimeout: ReturnType<typeof setTimeout> | undefined;
  let mediaWatch: ReturnType<typeof setInterval> | undefined;
  let lastFrameAt = Date.now();
  let decodedFrames = 0;
  let unlisten = () => {};
  let remoteQueue = Promise.resolve();
  let outgoing = Promise.resolve();
  let startWork: Promise<void> | undefined;
  let closeWork: Promise<void> | undefined;
  const pendingLocal: Candidate[] = [];
  // [대기 후보] local은 세션 문서 생성 전, remote는 answer 적용 전에 생긴 ICE를 임시 보관한다.
  // created는 Firestore 세션 생성 완료이고 connected/영상 재생 완료 플래그가 아니다.
  const pendingRemote: Candidate[] = [];
  let currentStream: MediaStream | null = null;

  const stop = () => {
    // [정리] 같은 stop 호출에는 같은 Promise를 돌려준다. 먼저 타이머·구독·미디어를 즉시 닫고,
    // 진행 중인 create가 끝난 뒤 세션 close를 시도해 늦게 생성된 세션도 정리한다.
    // 네트워크 오류로 close가 실패해도 브라우저 미디어 종료는 이미 수행된 상태다.
    if (closeWork) return closeWork;
    disposed = true;
    clearInterval(heartbeat);
    clearTimeout(timeout);
    clearTimeout(disconnectTimeout);
    clearInterval(mediaWatch);
    unlisten();
    pc.onicecandidate = null;
    pc.ontrack = null;
    pc.onconnectionstatechange = null;
    pc.getReceivers().forEach((receiver) => receiver.track?.stop());
    pc.close();
    options.stream(null);
    closeWork = (async () => {
      try { await startWork; } catch { /* Initial signaling failure is reported separately. */ }
      if (created) await channel.close().catch(() => {});
    })();
    return closeWork;
  };
  const fail = (error: unknown) => {
    if (disposed) return;
    // Close media immediately; an offline Firestore cleanup must not hide the
    // failure. New session creation still has to acquire the exclusive lease.
    void stop();
    options.failed(error);
  };
  const send = (candidate: Candidate) => {
    // 로컬 ICE 쓰기는 Promise 체인으로 순서를 유지한다. 이 브라우저 구현은 후보별로 전송하며
    // Jetson의 후보 묶음 전송과 구별한다. 종료 후 대기 중인 새 쓰기는 disposed로 건너뛴다.
    outgoing = outgoing.then(async () => {
      if (!disposed) await channel.candidate(candidate);
    }).catch(fail);
  };
  const enqueue = (action: () => Promise<void>) => {
    // answer 적용과 원격 후보 적용이 서로 추월하지 않도록 직렬화한다.
    // ICE가 먼저 도착해도 pendingRemote에 보관하고 remoteDescription이 설정된 뒤 적용한다.
    remoteQueue = remoteQueue.then(async () => { if (!disposed) await action(); }).catch(fail);
  };
  const runStart = async () => {
    try {
      options.status("연결 중");
      pc.addTransceiver("video", { direction: "recvonly" });
      // [후보 수집] recvonly는 브라우저가 영상을 받겠다는 협상 조건이다. 카메라 권한을 요청하지 않는다.
      // 후보 없음 이벤트는 수집 종료 표식으로 전달하며 IP 주소 하나를 고정해서 연결하는 방식이 아니다.
      pc.onicecandidate = ({ candidate }) => {
        if (disposed) return;
        const value: Candidate = candidate ? candidate.toJSON() : { endOfCandidates: true };
        if (created) send(value); else pendingLocal.push(value);
      };
      pc.ontrack = (event) => {
        // track 수신은 MediaStream 확보 단계다. 프레임 디코딩 진행과 video 요소의 실제 재생은
        // 아래 getStats 및 LiveView의 재생 이벤트에서 별도로 확인한다.
        if (!disposed) {
          currentStream = event.streams[0] ?? new MediaStream([event.track]);
          options.stream(currentStream);
        }
      };
      pc.onconnectionstatechange = () => {
        if (disposed) return;
        if (pc.connectionState === "connected") {
          clearTimeout(timeout);
          clearTimeout(disconnectTimeout);
          options.stream(currentStream);
          options.status("영상 연결됨");
          lastFrameAt = Date.now();
          clearInterval(mediaWatch);
          let sampling = false;
          // [영상 정지 감시] 연결 상태가 connected여도 프레임이 안 올 수 있다. 5초마다
          // framesDecoded 증가를 확인하고 15초 이상 진전이 없으면 실패로 넘긴다.
          // sampling으로 느린 getStats 호출의 중첩을 막으며, 이 숫자는 실제 화면 표시 FPS가 아니다.
          mediaWatch = setInterval(() => {
            if (disposed || sampling || pc.connectionState !== "connected") return;
            sampling = true;
            void pc.getStats().then((stats) => {
              if (disposed) return;
              let frames = 0;
              stats.forEach((stat) => {
                if (stat.type === "inbound-rtp" && stat.kind === "video") frames += stat.framesDecoded ?? 0;
              });
              if (frames > decodedFrames) lastFrameAt = Date.now();
              decodedFrames = frames;
              if (Date.now() - lastFrameAt >= 15_000) fail(new Error("영상 프레임 수신이 중단되었습니다."));
            }).catch(fail).finally(() => { sampling = false; });
          }, 5_000);
        } else if (pc.connectionState === "failed") fail(new Error("영상 연결 실패"));
        else if (pc.connectionState === "disconnected") {
          // 잠깐의 단절은 8초 유예한다. 그 전에 connected로 돌아오면 위 분기에서 타이머를 지운다.
          // 여기서는 스트림을 감춰 스냅샷 전환을 허용하고, 유예 후 실패 시 상위가 새 세션을 시도한다.
          options.stream(null);
          options.status("연결 확인 중");
          clearTimeout(disconnectTimeout);
          disconnectTimeout = setTimeout(() => fail(new Error("영상 연결 끊김")), 8_000);
        }
      };
      const offer = await pc.createOffer();
      // [세션 생성] 로컬 SDP를 설정한 뒤 Firestore에 offer를 기록한다. await 사이마다 종료 여부를
      // 확인한다. create 성공 후에만 쌓인 로컬 ICE를 전송하므로 세션 없는 후보 쓰기를 피한다.
      if (disposed) return;
      await pc.setLocalDescription(offer);
      if (disposed) return;
      await channel.create({ type: "offer", sdp: offer.sdp });
      created = true;
      if (disposed) return;
      pendingLocal.splice(0).forEach(send);
      unlisten = channel.listen(
        // [원격 응답] 동일 answer가 문서 갱신마다 와도 한 번만 적용한다. 잘못된 SDP는 실패 처리하고,
        // 유효 answer를 적용한 뒤 대기 ICE를 순서대로 추가한다. Firestore 수신 순서를 그대로 신뢰하지 않는다.
        (answer) => enqueue(async () => {
          if (pc.remoteDescription) return;
          if (answer.type !== "answer" || !answer.sdp) throw new Error("잘못된 송신 응답");
          await pc.setRemoteDescription(answer);
          for (const candidate of pendingRemote.splice(0)) {
            if (disposed) return;
            await pc.addIceCandidate(candidate.endOfCandidates ? undefined : candidate);
          }
        }),
        (candidate) => enqueue(async () => {
          if (!pc.remoteDescription) pendingRemote.push(candidate);
          else await pc.addIceCandidate(candidate.endOfCandidates ? undefined : candidate);
        }),
        (reason) => fail(new Error(sessionEndMessage(reason))), fail,
      );
      let renewing = false;
      // [생존/협상 시간] 10초마다 lease를 갱신하되 이전 요청이 끝나지 않았으면 겹쳐 보내지 않는다.
      // 25초 협상 제한은 이 지점부터 시작한다. 위의 최초 create 요청 대기까지 포함한 총시간은 아니다.
      // connected에서 협상 타이머를 지우며, lease/heartbeat 실패는 재생 중이어도 연결 종료 사유가 된다.
      heartbeat = setInterval(() => {
        if (disposed || renewing) return;
        renewing = true;
        void channel.heartbeat().catch(fail).finally(() => { renewing = false; });
      }, 10_000);
      timeout = setTimeout(() => fail(new Error("송신자 응답 시간 초과")), 25_000);
    } catch (error) { fail(error); }
  };
  const start = () => {
    // 같은 viewer의 start를 여러 번 불러도 최초 작업만 실행한다. 재연결은 새 viewer/세션을 만들어야 한다.
    if (disposed) return closeWork ?? Promise.resolve();
    if (!startWork) startWork = runStart();
    return startWork;
  };
  return { start, stop };
}
