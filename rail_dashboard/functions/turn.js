// [TURN 임시 계정] 시청 권한 확인과 표준 TURN REST(coturn use-auth-secret) 계정 계산만 담당한다.
// 네트워크·Firebase 호출은 index.js에 두어 이 파일은 로컬 node:test로 바로 검증한다.
import { createHmac } from "node:crypto";

export const ROBOT_ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/;
// 시청 세션의 최대 수명(1시간)보다 조금 길게 준다. 계정이 먼저 만료되면 TURN 할당 갱신이 실패한다.
export const TURN_TTL_SECONDS = 65 * 60;
export const TURN_PORT = 3478;
const IPV4_PATTERN = /^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$/;
const HOSTNAME_PATTERN = /^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/i;

export class TurnRequestError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

// Firestore 규칙의 isViewer와 같은 조건: 인증된 이메일 + 로봇 enabled + viewerUids 포함.
export function authorizeViewer(auth, robotId, robot) {
  if (!auth || typeof auth.uid !== "string" || !auth.uid) {
    throw new TurnRequestError("unauthenticated", "로그인이 필요합니다.");
  }
  if (typeof robotId !== "string" || !ROBOT_ID_PATTERN.test(robotId)) {
    throw new TurnRequestError("invalid-argument", "robotId 형식이 맞지 않습니다.");
  }
  if (auth.token?.email_verified !== true) {
    throw new TurnRequestError("permission-denied", "이메일 인증이 필요합니다.");
  }
  if (!robot || robot.enabled !== true || !Array.isArray(robot.viewerUids)
      || !robot.viewerUids.includes(auth.uid)) {
    throw new TurnRequestError("permission-denied", "이 로봇의 시청 권한이 없습니다.");
  }
}

// coturn use-auth-secret 규약: username = "<만료 unix초>:<식별자>",
// credential = base64(HMAC-SHA1(공유 비밀키, username)). 서버는 같은 계산으로 검증하고 만료를 확인한다.
export function turnRestIceServers({ secret, host, uid, nowSeconds, ttlSeconds = TURN_TTL_SECONDS }) {
  if (typeof secret !== "string" || secret.length < 32) {
    throw new TurnRequestError("unavailable", "TURN 서버 설정이 완료되지 않았습니다.");
  }
  if (typeof host !== "string" || !(IPV4_PATTERN.test(host) || HOSTNAME_PATTERN.test(host))) {
    throw new TurnRequestError("unavailable", "TURN 서버 주소가 설정되지 않았습니다.");
  }
  const username = `${Math.floor(nowSeconds) + ttlSeconds}:${uid}`;
  const credential = createHmac("sha1", secret).update(username).digest("base64");
  return [{
    urls: [
      `turn:${host}:${TURN_PORT}?transport=udp`,
      `turn:${host}:${TURN_PORT}?transport=tcp`,
    ],
    username,
    credential,
  }];
}
