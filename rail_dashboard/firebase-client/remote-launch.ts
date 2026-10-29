import { addDoc, collection, doc, onSnapshot, serverTimestamp, Timestamp } from "firebase/firestore";
import { auth, db } from "./firebase";

export type LauncherStatus = {
  schemaVersion: 1;
  senderId: string;
  ready: boolean;
  missionRunning: boolean;
  updatedAt: Timestamp;
};

export type LaunchRequest = {
  action: "LAUNCH_MISSION";
  state: "requested" | "accepted" | "succeeded" | "failed";
  // 수명은 원격 명령과 같이 서버 createdAt + REMOTE_REQUEST_TTL_MS다.
  createdAt: Timestamp;
  finishedAt: Timestamp | null;
  resultCode: string | null;
};

export function listenLauncherStatus(robotId: string,
  next: (status: LauncherStatus | null) => void, fail: () => void) {
  return onSnapshot(doc(db, "robots", robotId, "launcher", "status"), (snapshot) => {
    const data = snapshot.data();
    next(data?.schemaVersion === 1 && typeof data.senderId === "string"
      && typeof data.ready === "boolean" && typeof data.missionRunning === "boolean"
      && data.updatedAt instanceof Timestamp ? data as LauncherStatus : null);
  }, fail);
}

export async function requestMissionLaunch(robotId: string) {
  const uid = auth.currentUser?.uid;
  if (!uid) throw new Error("로그인이 필요합니다.");
  const result = await addDoc(collection(db, "robots", robotId, "launchRequests"), {
    schemaVersion: 1, robotId, ownerUid: uid, action: "LAUNCH_MISSION", state: "requested",
    createdAt: serverTimestamp(), acceptedAt: null, finishedAt: null, resultCode: null,
  });
  return result.id;
}

export function listenMissionLaunch(robotId: string, requestId: string,
  next: (request: LaunchRequest | null) => void, fail: () => void) {
  return onSnapshot(doc(db, "robots", robotId, "launchRequests", requestId), (snapshot) => {
    const data = snapshot.data({ serverTimestamps: "estimate" });
    next(data?.action === "LAUNCH_MISSION"
      && ["requested", "accepted", "succeeded", "failed"].includes(data.state)
      && data.createdAt instanceof Timestamp
      && (data.finishedAt === null || data.finishedAt instanceof Timestamp)
      ? data as LaunchRequest : null);
  }, fail);
}
