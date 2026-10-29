import { httpsCallable } from "firebase/functions";
import { functions } from "./firebase";

// [ICE 서버 선택] 연결 직전에 Cloud Function(turnCredentials)에서 TURN 임시 계정을 받는다.
// 실패하면 기존처럼 STUN만으로 시도한다. 같은 네트워크에서는 STUN만으로도 연결될 수 있다.
export const STUN_FALLBACK: RTCIceServer[] = [{ urls: "stun:stun.l.google.com:19302" }];

type TurnResponse = { iceServers: RTCIceServer[]; ttlSeconds: number };
export type ViewerIceServers = { servers: RTCIceServer[]; relay: boolean };

// 시청 세션은 최대 1시간이다. 받은 계정이 새 세션 전체를 덮을 만큼 남아 있을 때만 재사용한다.
const SESSION_MAX_SECONDS = 61 * 60;
let cached: { robotId: string; servers: RTCIceServer[]; reuseUntil: number } | null = null;

function validServers(value: unknown): value is RTCIceServer[] {
  return Array.isArray(value) && value.length > 0 && value.every((server) =>
    server && typeof server === "object"
    && Array.isArray((server as RTCIceServer).urls)
    && ((server as RTCIceServer).urls as string[]).every((url) => typeof url === "string"));
}

export async function viewerIceServers(robotId: string): Promise<ViewerIceServers> {
  const now = Date.now();
  if (cached && cached.robotId === robotId && now < cached.reuseUntil) {
    return { servers: cached.servers, relay: true };
  }
  try {
    const request = httpsCallable<{ robotId: string }, TurnResponse>(
      functions, "turnCredentials", { timeout: 6_000 });
    const { data } = await request({ robotId });
    if (!validServers(data?.iceServers) || typeof data.ttlSeconds !== "number") throw new Error("invalid");
    const servers = [...data.iceServers, ...STUN_FALLBACK];
    cached = { robotId, servers, reuseUntil: now + Math.max(0, data.ttlSeconds - SESSION_MAX_SECONDS) * 1000 };
    return { servers, relay: true };
  } catch {
    return { servers: STUN_FALLBACK, relay: false };
  }
}
