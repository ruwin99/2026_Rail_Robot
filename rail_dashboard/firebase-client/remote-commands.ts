import { addDoc, collection, doc, onSnapshot, serverTimestamp, Timestamp } from "firebase/firestore";
import { auth, db } from "./firebase";

// 요청 수명은 서버 createdAt 기준 10초다. 브라우저 시계로 expiresAt을 만들지 않으므로
// PC 시계가 틀려도 규칙이 요청을 거부하지 않는다. Jetson도 같은 기준으로 판정한다.
export const REMOTE_REQUEST_TTL_MS = 10_000;

export type RemoteAction = "START" | "E_STOP";
export type RemoteCommand = {
  action: RemoteAction;
  state: "requested" | "accepted" | "succeeded" | "failed";
  createdAt: Timestamp;
  resultCode: string | null;
};

export async function requestRemoteCommand(robotId: string, action: RemoteAction) {
  const uid = auth.currentUser?.uid;
  if (!uid) throw new Error("로그인이 필요합니다.");
  const result = await addDoc(collection(db, "robots", robotId, "commands"), {
    schemaVersion: 1, robotId, ownerUid: uid, action, state: "requested",
    createdAt: serverTimestamp(), acceptedAt: null, finishedAt: null, resultCode: null,
  });
  return result.id;
}

export function listenRemoteCommand(robotId: string, commandId: string,
  next: (command: RemoteCommand | null) => void, fail: () => void) {
  return onSnapshot(doc(db, "robots", robotId, "commands", commandId), (snapshot) => {
    // 서버 확정 전 로컬 스냅샷도 createdAt 추정값을 갖도록 estimate를 쓴다.
    const data = snapshot.data({ serverTimestamps: "estimate" });
    next(data && ["START", "E_STOP"].includes(data.action)
      && ["requested", "accepted", "succeeded", "failed"].includes(data.state)
      && data.createdAt instanceof Timestamp ? data as RemoteCommand : null);
  }, fail);
}
