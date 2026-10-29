import { collection, doc, onSnapshot, query, where, orderBy, runTransaction,
  serverTimestamp, Timestamp, setDoc } from "firebase/firestore";
import { auth, db } from "./firebase";
import type { Candidate, SignalChannel } from "./webrtc-peer";

// [Firestore 시그널링] robots/{robotId} 아래 세션·잠금·ICE 문서를 읽고 쓴다. 영상/JPEG 저장 경로는 아니다.
// 브라우저 offer / Jetson answer 역할이며, 실제 연결·미디어 처리 순서는 webrtc-peer.ts에서 제어한다.
// 아래 클라이언트 검사는 빠른 오류 안내용이다. 최종 읽기/쓰기 권한은 배포된 Firestore rules가 결정한다.
export type Robot = { id: string; name: string; senderId: string; previewObjectKey?: string;
  remoteControlEnabled?: boolean; remoteLaunchEnabled?: boolean; operatorUids?: string[] };
export const listenRobots = (next: (robots: Robot[]) => void, fail: (error: unknown) => void) => {
  // [로봇 목록] 로그인 UID가 viewerUids에 포함된 활성 로봇만 구독한다. 로그인만으로 모든 로봇을 보지 않는다.
  // 목록이 비면 UID/robots.enabled/viewerUids를, 구독 오류면 권한과 쿼리 조건을 확인한다.
  const uid = auth.currentUser?.uid;
  if (!uid) { next([]); return () => {}; }
  return onSnapshot(query(collection(db, "robots"), where("enabled", "==", true),
    where("viewerUids", "array-contains", uid)), (snapshot) => next(snapshot.docs.map((item) => ({
      id: item.id, name: item.data().name || item.id, senderId: item.data().senderId,
      previewObjectKey: item.data().previewObjectKey,
      remoteControlEnabled: item.data().remoteControlEnabled === true,
      remoteLaunchEnabled: item.data().remoteLaunchEnabled === true,
      operatorUids: Array.isArray(item.data().operatorUids) ? item.data().operatorUids : [],
    }))), fail);
};

export function firebaseSignalChannel(robot: Robot): SignalChannel {
  const uid = auth.currentUser?.uid;
  if (!uid) throw new Error("로그인이 필요합니다.");
  const session = doc(collection(db, "robots", robot.id, "webrtcSessions"));
  // 호출마다 새 세션 ID를 만든다. active 잠금은 로봇당 하나이고, leaseUntil은 30초 생존 임대다.
  // 세션 expiresAt(최대 수명 1시간), heartbeat 시각, leaseUntil을 동일한 만료 기준으로 보지 않는다.
  const lock = doc(db, "robots", robot.id, "webrtcLocks", "active");
  const config = doc(db, "robots", robot.id);
  const lease = () => Timestamp.fromMillis(Date.now() + 30_000);
  return {
    async create(offer) {
      // [생성 트랜잭션] 로봇 권한/송신자와 기존 임대를 읽고 세션+잠금을 함께 기록한다.
      // 두 뷰어가 동시에 시도해도 살아 있는 잠금을 덮어쓰지 않도록 트랜잭션에서 재확인한다.
      // 현재 코드에서는 같은 사용자도 유효한 이전 임대가 남으면 '다른 뷰어' 오류가 날 수 있다.
      await runTransaction(db, async (tx) => {
        const robotDoc = await tx.get(config);
        const existing = await tx.get(lock);
        const data = robotDoc.data();
        if (!data?.enabled || !data.viewerUids?.includes(uid) || data.senderId !== robot.senderId) {
          throw new Error("이 로봇의 시청 권한이 없습니다.");
        }
        if (existing.exists() && existing.data().leaseUntil.toMillis() > Date.now()) {
          throw new Error("다른 뷰어가 시청 중입니다. 잠시 후 다시 연결하세요.");
        }
        tx.set(session, {
          schemaVersion: 1, robotId: robot.id, ownerUid: uid, senderId: robot.senderId,
          offer, answer: null, state: "requested", createdAt: serverTimestamp(),
          viewerHeartbeatAt: serverTimestamp(), expiresAt: Timestamp.fromMillis(Date.now() + 3_600_000),
          endedAt: null, errorCode: null,
        });
        tx.set(lock, { sessionId: session.id, ownerUid: uid, leaseUntil: lease() });
      });
    },
    async heartbeat() {
      // [갱신] 이 세션이 여전히 잠금 주인이고 closed/failed가 아닐 때만 두 생존 시각을 갱신한다.
      // 세션 최대 수명 expiresAt은 연장하지 않는다. 브라우저 시계 기반 lease와 서버 timestamp를 구별한다.
      await runTransaction(db, async (tx) => {
        const current = await tx.get(lock);
        const state = await tx.get(session);
        if (current.data()?.sessionId !== session.id ||
          !state.exists() || ["closed", "failed"].includes(state.data().state)) {
          throw new Error("시청 세션이 종료되었습니다.");
        }
        tx.update(session, { viewerHeartbeatAt: serverTimestamp() });
        tx.update(lock, { leaseUntil: lease() });
      });
    },
    async candidate(value: Candidate) {
      // [브라우저 후보 쓰기] 후보 문자열과 SDP 미디어 위치를 viewerCandidates의 새 문서에 저장한다.
      // 수집 종료 표식에는 주소가 없다. 이 함수 자체는 잠금을 재조회하지 않으므로 늦은 쓰기가
      // rules에서 거절되면 세션 종료 시점/ownerUid/현재 잠금과 함께 원인을 확인한다.
      const data = value.endOfCandidates ? { endOfCandidates: true } : {
        candidate: value.candidate ?? "", sdpMid: value.sdpMid ?? null,
        sdpMLineIndex: value.sdpMLineIndex ?? null, usernameFragment: value.usernameFragment ?? null,
      };
      await setDoc(doc(collection(session, "viewerCandidates")), { ...data, createdAt: serverTimestamp() });
    },
    listen(answer, candidate, ended, failed) {
      // [Jetson 응답 구독] 세션 문서는 answer·종료 상태를, senderCandidates는 원격 ICE를 전달한다.
      // 서로 다른 구독이므로 answer가 ICE보다 먼저 도착한다는 보장이 없다. 순서 보정은 peer가 맡는다.
      const stopSession = onSnapshot(session, (snapshot) => {
        const data = snapshot.data();
        if (!data || ["closed", "failed"].includes(data.state)) {
          ended(typeof data?.errorCode === "string" ? data.errorCode : undefined);
          return;
        }
        if (data.answer) answer(data.answer);
      }, failed);
      const stopIce = onSnapshot(query(collection(session, "senderCandidates"), orderBy("createdAt")), (snapshot) => {
          // 최초 스냅샷의 기존 후보도 added로 전달된다. 수정/삭제 이벤트는 후보 재적용에 사용하지 않는다.
          const added = snapshot.docChanges().filter((change) => change.type === "added");
          for (const change of added) candidate(change.doc.data() as Candidate);
        }, failed);
      return () => { stopSession(); stopIce(); };
    },
    async close() {
      // [종료] 세션을 closed로 표시하되 이미 failed이면 원래 실패 정보를 유지한다.
      // active 잠금이 아직 내 세션일 때만 삭제하여 새 시청자의 잠금을 지우지 않는다.
      // 세션/ICE 문서 자체를 재귀 삭제하는 함수는 아니므로 종료 표시와 보관 문서 정리를 구별한다.
      await runTransaction(db, async (tx) => {
        const current = await tx.get(lock);
        const existing = await tx.get(session);
        if (existing.exists() && !["closed", "failed"].includes(existing.data().state)) {
          tx.update(session, { state: "closed", endedAt: serverTimestamp() });
        }
        if (current.data()?.sessionId === session.id) tx.delete(lock);
      });
    },
  };
}
