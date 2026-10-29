// [TURN 임시 계정 발급] 시청 권한이 있는 로그인 사용자에게만 자체 coturn 서버의 임시 계정을 준다.
// 브라우저에는 장기 비밀값을 두지 않는다. 공유 비밀키는 Secret Manager(TURN_SHARED_SECRET)에만 있고,
// 같은 값을 coturn VM이 부팅 때 읽는다(turn-server/coturn-startup.sh).
import { onCall, HttpsError } from "firebase-functions/v2/https";
import { defineSecret, defineString } from "firebase-functions/params";
import { logger } from "firebase-functions";
import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import {
  ROBOT_ID_PATTERN, TURN_TTL_SECONDS, TurnRequestError, authorizeViewer, turnRestIceServers,
} from "./turn.js";

initializeApp();

const turnSharedSecret = defineSecret("TURN_SHARED_SECRET");
// 비밀값이 아닌 coturn VM의 고정 외부 IP(또는 도메인). functions/.env에 둔다.
const turnHost = defineString("TURN_HOST");

export const turnCredentials = onCall({
  region: "asia-northeast3",
  secrets: [turnSharedSecret],
  memory: "256MiB",
  timeoutSeconds: 10,
  maxInstances: 3,
}, async (request) => {
  const robotId = request.data?.robotId;
  try {
    if (typeof robotId !== "string" || !ROBOT_ID_PATTERN.test(robotId)) {
      throw new TurnRequestError("invalid-argument", "robotId 형식이 맞지 않습니다.");
    }
    const robot = await getFirestore().doc(`robots/${robotId}`).get();
    authorizeViewer(request.auth, robotId, robot.data());
    const iceServers = turnRestIceServers({
      secret: turnSharedSecret.value(),
      host: turnHost.value(),
      uid: request.auth.uid,
      nowSeconds: Date.now() / 1000,
    });
    return { iceServers, ttlSeconds: TURN_TTL_SECONDS };
  } catch (error) {
    if (error instanceof TurnRequestError) throw new HttpsError(error.code, error.message);
    // 계정·비밀값이 섞이지 않도록 오류 종류만 남긴다.
    logger.error("TURN credential request failed", { name: error?.name ?? "unknown" });
    throw new HttpsError("unavailable", "TURN 계정을 발급하지 못했습니다.");
  }
});
