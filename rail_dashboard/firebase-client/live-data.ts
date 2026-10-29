import { collection, doc, limit, onSnapshot, orderBy, query, Timestamp } from "firebase/firestore";
import { db } from "./firebase";
import type { NodeHealthSignal } from "./node-health";

// [실시간 데이터 계약] Jetson이 기록한 Firestore 문서를 SDK 구독으로 받는다. ROS 토픽에 직접 붙지는 않는다.
// telemetry는 상태/수치, captures는 사진의 상태와 Storage object key다. 이 문서에 JPEG 바이트는 없다.
// 구독 수명과 이미지 다운로드는 LiveTelemetry.tsx의 useLiveTelemetry/CaptureImage에서 추적한다.
export type CameraMetrics = { fps: number | null; rustMs: number | null; crackMs: number | null;
  obstacleMs: number | null; observedAt: string | null };
export type Telemetry = { updatedAt: Timestamp; topics: { name: string; publishers: number; types: string[] }[];
  mission: { phase: string; completed: number; total: number; remainingTasks: string[]; observedAt: string } | null;
  cameras: { side: CameraMetrics; front: CameraMetrics }; nodeHealth?: NodeHealthSignal[] };
export type LiveCapture = { captureId: string; runId: string; cameraRole: "side" | "front";
  phase: string; zone: number; capturedAt: string; analyzedAt: string | null;
  status: "captured" | "processed"; rawObjectKey: string;
  processedObjectKeys: { label: string; objectKey: string }[] };

export function listenTelemetry(robotId: string, next: (data: Telemetry | null) => void, fail: () => void) {
  // [상태 구독] robots/{robotId}/telemetry/current 하나의 최신 문서를 받는다.
  // schemaVersion/Timestamp/topics/cameras의 최소 모양만 확인하며 불일치하면 null을 전달한다.
  // null은 실제 측정값 0이 아니다. 권한/연결 실패는 별도 fail 콜백으로 전달한다.
  return onSnapshot(doc(db, "robots", robotId, "telemetry", "current"), (snapshot) => {
    const data = snapshot.data();
    next(data?.schemaVersion === 1 && data.updatedAt instanceof Timestamp && Array.isArray(data.topics)
      && data.cameras ? data as Telemetry : null);
  }, fail);
}
export function listenCaptures(robotId: string, next: (data: LiveCapture[]) => void, fail: () => void) {
  // [사진 목록 구독] capturedAt 역순의 최근 12개 문서만 가져온다. 전체 임무 기록 조회가 아니다.
  // status=captured는 원본 단계, processed는 분석 단계다. processedObjectKeys 변화로 분석 이미지가 추가된다.
  // 여기서는 object key를 넘길 뿐 실제 이미지의 다운로드 성공 여부는 확인하지 않는다.
  return onSnapshot(query(collection(db, "robots", robotId, "captures"), orderBy("capturedAt", "desc"), limit(12)),
    (snapshot) => next(snapshot.docs.map((item) => item.data()).filter((item) => item.schemaVersion === 1
      && typeof item.captureId === "string" && Array.isArray(item.processedObjectKeys)) as LiveCapture[]), fail);
}

export function isFresh(time: string | undefined | null, now: number, maxAge = 10_000) {
  // [시각 유효성] 기본 10초 미만의 값만 신선하게 보고 최대 5초의 미래 시각 차이를 허용한다.
  // 미수신/파싱 불가와 오래된 값은 false다. Jetson 시계와 브라우저 시계가 다르면 정상 데이터도 낡게 보일 수 있다.
  const age = time ? now - Date.parse(time) : NaN;
  return Number.isFinite(age) && age >= -5_000 && age < maxAge;
}
export function metric(value: unknown, unit: string) {
  // NaN/음수/미수신은 '—'로 구분해 0 FPS나 0ms라는 실제 측정값으로 오해하지 않게 한다.
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)} ${unit}` : "—";
}
