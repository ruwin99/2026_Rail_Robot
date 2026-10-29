import {
  collection,
  doc,
  getDocs,
  orderBy,
  query,
  serverTimestamp,
  writeBatch,
} from "firebase/firestore";
import { getBlob, ref, uploadBytes } from "firebase/storage";
import type { InspectionExport } from "../app/data-contract";
import { db, storage } from "./firebase";

// [검사 기록 입출력] 완료된 검사 JSON은 Firestore inspection_exports, 이미지 바이트는 Storage에 저장한다.
// 실시간 로봇 상태/촬영 문서 구독은 live-data.ts, WebRTC 연결 정보는 webrtc-signaling.ts의 별도 경로다.
// 데이터가 안 보이면 기록 문서 누락과 Storage 이미지 접근 실패를 분리해서 확인한다.
type StoredInspectionExport = {
  payload?: InspectionExport;
};

const isInspectionExport = (value: unknown): value is InspectionExport => {
  // [최소 형식 검사] schema_version과 최상위 배열만 검사한다. 각 배열 항목의 값까지
  // 완전히 검증하는 스키마 파서는 아니므로 형식 통과를 데이터 내용의 정확성 보장으로 보지 않는다.
  if (value == null || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return record.schema_version === 1
    && ["runs", "model_provenance", "captures", "analyses", "artifacts", "run_summaries"]
      .every((key) => Array.isArray(record[key]));
};

const safeStoragePath = (path: string) => (
  // 이 로더가 취급하는 기존 검사 미디어 범위다. 클라이언트 경로 검사와 Storage rules의 권한 검사는 별개다.
  path.startsWith("dashboard/media/") && !path.includes("..") && !path.startsWith("/")
);

const fileNameFromPath = (path: string) => path.split("/").at(-1) ?? "";
const LIVE_PREVIEW_OBJECT_KEY = "live/jetson/gui.jpg";

export async function loadLivePreview() {
  // [고정 경로 미리보기] JPEG 하나를 Blob URL로 바꾼다. 여기에는 반복 타이머/촬영 시각 검사가 없다.
  // 로봇별 5초 폴백은 LiveView.tsx가 담당한다. 호출자는 사용이 끝난 URL을 revoke해야 한다.
  try {
    const blob = await getBlob(ref(storage, LIVE_PREVIEW_OBJECT_KEY));
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

const exportForRun = (document: InspectionExport, runId: string): InspectionExport => {
  // [임무별 분리] runId로 촬영을 선택하고 captureId를 통해 분석·이미지를 함께 묶는다.
  // 브라우저 임시 URL을 서버에 저장하지 않도록 artifact.public_url을 null로 바꾸고 object_key만 유지한다.
  const captures = document.captures.filter((capture) => capture.run_id === runId);
  const captureIds = new Set(captures.map((capture) => capture.capture_id));
  return {
    schema_version: document.schema_version,
    exported_at_utc: document.exported_at_utc,
    runs: document.runs.filter((run) => run.run_id === runId),
    model_provenance: document.model_provenance.filter((item) => item.run_id === runId),
    captures,
    analyses: document.analyses.filter((analysis) => captureIds.has(analysis.capture_id)),
    artifacts: document.artifacts
      .filter((artifact) => captureIds.has(artifact.capture_id))
      .map((artifact) => ({ ...artifact, public_url: null })),
    run_summaries: document.run_summaries.filter((summary) => summary.run_id === runId),
  };
};

export async function importLegacyInspectionFiles(
  files: File[],
  onProgress: (message: string) => void,
) {
  const jsonFile = files.find((file) => file.name === "inspection-export.json");
  if (!jsonFile) throw new Error("inspection-export.json 파일을 찾지 못했습니다.");

  const parsed: unknown = JSON.parse(await jsonFile.text());
  if (!isInspectionExport(parsed)) throw new Error("지원하지 않는 검사 데이터 형식입니다.");

  const filesByName = new Map(files.map((file) => [file.name, file]));
  // [가져올 이미지 선택] 원본 전체가 아니라 지원되는 세 종류의 preview 중 조건에 맞는 것만 업로드한다.
  // 파일 매칭은 object_key의 마지막 파일명 기준이다. 누락 오류에서는 JSON 참조와 선택한 파일명을 비교한다.
  const previewArtifacts = parsed.artifacts.filter((artifact) => (
    (["rust_preview", "crack_preview", "obstacle_preview"].includes(artifact.artifact_type))
      && artifact.public_url
      && safeStoragePath(artifact.object_key)
  ));

  const missingFiles = previewArtifacts.filter(
    (artifact) => !filesByName.has(fileNameFromPath(artifact.object_key)),
  );
  if (missingFiles.length) {
    throw new Error(`마스크 이미지 ${missingFiles.length}개를 찾지 못했습니다.`);
  }

  for (const [index, artifact] of previewArtifacts.entries()) {
    // [Storage 우선 업로드] 이미지를 하나씩 완료한 후 아래 Firestore 기록을 쓴다.
    // Storage 업로드와 Firestore batch는 하나의 원자적 트랜잭션이 아니다. 중간 실패 시
    // 이미 올라간 이미지가 남을 수 있으므로 '기록 없음'을 '아무것도 업로드 안 됨'으로 보지 않는다.
    const file = filesByName.get(fileNameFromPath(artifact.object_key));
    if (!file) continue;
    onProgress(`마스크 이미지 업로드 ${index + 1}/${previewArtifacts.length}`);
    await uploadBytes(ref(storage, artifact.object_key), file, {
      contentType: artifact.media_type || file.type,
      customMetadata: {
        captureId: artifact.capture_id,
        artifactType: artifact.artifact_type,
        source: "legacy-test-data",
      },
    });
  }

  onProgress("검사 기록을 Firestore 문서로 변환 중");
  const batch = writeBatch(db);
  // [Firestore 확정] 문서 ID는 run_id이므로 같은 임무를 다시 가져오면 해당 문서를 set으로 덮어쓴다.
  // imported_at은 서버 수입 시각이고 촬영 시각이 아니다. 임무별 payload 안의 촬영 시각을 따로 확인한다.
  parsed.runs.forEach((run) => {
    batch.set(doc(db, "inspection_exports", run.run_id), {
      payload: exportForRun(parsed, run.run_id),
      source: "legacy-test-data",
      imported_at: serverTimestamp(),
    });
  });
  await batch.commit();

  return {
    runCount: parsed.runs.length,
    previewCount: previewArtifacts.length,
  };
}

export async function loadServerInspectionData() {
  // [기록 읽기] 수입 시각 역순의 일회성 조회다. onSnapshot 구독이 아니므로 이후 서버 변경을
  // 자동으로 계속 받지 않는다. 문서가 없거나 최소 스키마를 통과하지 못하면 document=null이다.
  const snapshot = await getDocs(query(collection(db, "inspection_exports"), orderBy("imported_at", "desc")));
  const exports = snapshot.docs
    .map((item) => (item.data() as StoredInspectionExport).payload)
    .filter((item): item is InspectionExport => isInspectionExport(item));

  if (!exports.length) return { document: null, objectUrls: [] as string[] };

  const combined: InspectionExport = {
    // 유효 문서의 배열들을 합친다. 여기서 중복 captureId 제거/내용 충돌 해결까지 수행하지는 않는다.
    schema_version: 1,
    exported_at_utc: exports[0].exported_at_utc,
    runs: exports.flatMap((item) => item.runs),
    model_provenance: exports.flatMap((item) => item.model_provenance),
    captures: exports.flatMap((item) => item.captures),
    analyses: exports.flatMap((item) => item.analyses),
    artifacts: exports.flatMap((item) => item.artifacts),
    run_summaries: exports.flatMap((item) => item.run_summaries),
  };

  const objectUrls: string[] = [];
  const artifacts = await Promise.all(combined.artifacts.map(async (artifact) => {
    // [이미지 복원] object_key로 인증된 SDK 다운로드를 하고 현재 브라우저용 Blob URL을 만든다.
    // 다운로드 실패는 전체 기록 조회를 실패시키지 않고 원래 artifact를 남긴다.
    // 반환 objectUrls는 호출자가 화면 교체/종료 시 해제해야 할 메모리 자원 목록이다.
    if (!["rust_preview", "crack_preview", "obstacle_preview"].includes(artifact.artifact_type)) {
      return artifact;
    }
    if (!safeStoragePath(artifact.object_key)) return artifact;
    try {
      const blob = await getBlob(ref(storage, artifact.object_key));
      const objectUrl = URL.createObjectURL(blob);
      objectUrls.push(objectUrl);
      return { ...artifact, public_url: objectUrl };
    } catch {
      return artifact;
    }
  }));

  return { document: { ...combined, artifacts }, objectUrls };
}
