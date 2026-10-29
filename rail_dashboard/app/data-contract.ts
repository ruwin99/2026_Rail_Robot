export type Phase = "initial" | "rescan" | "manual";
export type DefectType = "rust" | "crack" | "obstacle";
export type AnalysisStatus = "ready" | "disabled" | "error";
export type CameraRole = "side" | "top";

export type InspectionExport = {
  schema_version: number;
  exported_at_utc: string;
  runs: Array<{
    run_id: string;
    pipeline_version: number;
    started_at_utc: string | null;
    finished_at_utc: string | null;
    display_timezone: string;
    local_date: string | null;
    status: "in_progress" | "complete" | "failed";
    capture_target: number;
    failure_reason: string | null;
    crane_id?: string | null;
    crane_label?: string | null;
  }>;
  model_provenance: Array<Record<string, unknown>>;
  captures: Array<{
    capture_id: string;
    run_id: string;
    camera_role?: CameraRole;
    phase: Phase;
    phase_sequence: number | null;
    logical_zone_number: number | null;
    trigger: string;
    captured_at_utc: string;
    captured_local_date: string;
    display_timezone: string;
    width: number;
    height: number;
    processing_status: "ready" | "error";
    raw_image_key: string;
  }>;
  analyses: Array<{
    capture_id: string;
    defect_type: DefectType;
    status: AnalysisStatus;
    detected: boolean | null;
    positive_pixels: number | null;
    inspected_pixels: number | null;
    ratio_fraction: number | null;
    detector_method: string | null;
    provenance_id: string | null;
    overlap_policy: string | null;
    grade_pixel_counts: Record<string, number> | null;
    metric?: "bbox_area_ratio";
    detection_count?: number | null;
  }>;
  artifacts: Array<{
    artifact_id: string;
    capture_id: string;
    artifact_type: "raw" | "analyzed" | "rust_preview" | "crack_preview" | "obstacle_preview";
    object_key: string;
    public_url: string | null;
    media_type: string;
    width: number;
    height: number;
    sha256: string;
  }>;
  run_summaries: Array<{
    run_id: string;
    initial_capture_count: number;
    rescan_capture_count: number;
    before_positive_pixels: number;
    before_inspected_pixels: number;
    before_ratio_fraction: number | null;
    after_positive_pixels: number;
    after_inspected_pixels: number;
    after_ratio_fraction: number | null;
    absolute_reduction_fraction: number | null;
    relative_improvement_fraction: number | null;
    summary_complete: boolean;
    display_mode?: "user_requested_demo_swap";
    display_note?: string;
  }>;
};

export type DetectionView = {
  id: string;
  captureId: string;
  cameraRole: CameraRole;
  phase: Phase;
  zone: number | null;
  capturedAt: string;
  type: DefectType;
  ratio: number;
  maskUrl: string | null;
  maskObjectKey: string | null;
};

export type AnalysisIssueView = {
  id: string;
  captureId: string;
  cameraRole: CameraRole;
  phase: Phase;
  zone: number | null;
  capturedAt: string;
  type: DefectType;
  status: "disabled" | "error" | "missing_preview" | "capture_error";
  message: string;
};

export type RunView = {
  id: string;
  date: string;
  startedAt: string;
  timezone: string;
  status: "in_progress" | "complete" | "failed";
  beforeRatio: number | null;
  afterRatio: number | null;
  reduction: number | null;
  improvement: number | null;
  summaryComplete: boolean;
  summaryIsDemo: boolean;
  summaryDisplayNote: string | null;
  railSectionTarget: number;
  initialRailSectionCount: number;
  rescanRailSectionCount: number;
  detections: DetectionView[];
  issues: AnalysisIssueView[];
  failureReason: string | null;
  craneId: string;
  craneLabel: string;
  obstacleComparison: ObstacleComparison;
};

export type ObstacleComparison = {
  beforeRatio: number | null;
  afterRatio: number | null;
  reduction: number | null;
  improvement: number | null;
  beforeCount: number;
  afterCount: number;
  complete: boolean;
  pairs: Array<{ zone: number; before: DetectionView | null; after: DetectionView | null }>;
};

function buildObstacleComparison(document: InspectionExport, runId: string, target: number): ObstacleComparison {
  const captures = document.captures.filter((c) => c.run_id === runId && c.camera_role === "top");
  const valid = (capture: InspectionExport["captures"][number]) => {
    const analysis = document.analyses.find((a) => a.capture_id === capture.capture_id && a.defect_type === "obstacle");
    if (capture.processing_status !== "ready" || analysis?.status !== "ready") return null;
    const { positive_pixels: positive, inspected_pixels: inspected, ratio_fraction: ratio } = analysis;
    if (positive == null || inspected == null || ratio == null || !Number.isFinite(ratio)
      || !Number.isInteger(positive) || !Number.isInteger(inspected) || inspected <= 0
      || positive < 0 || positive > inspected || Math.abs(ratio - positive / inspected) > 1e-9) return null;
    return analysis;
  };
  const phaseEntries = (phase: Phase) => Array.from({ length: target }, (_, i) => {
    const candidates = captures.filter((c) => c.phase === phase && c.logical_zone_number === i + 1);
    // Ambiguous duplicate captures cannot form a trustworthy before/after pair.
    if (candidates.length !== 1) return null;
    const capture = candidates[0];
    const analysis = valid(capture);
    return analysis ? { capture, analysis } : null;
  });
  const before = phaseEntries("initial");
  const after = phaseEntries("rescan");
  const aggregate = (entries: typeof before) => {
    if (!target || entries.some((entry) => !entry)) return null;
    const measured = entries.filter((entry) => entry != null);
    return measured.reduce((sum, entry) => sum + entry.analysis.positive_pixels!, 0)
      / measured.reduce((sum, entry) => sum + entry.analysis.inspected_pixels!, 0);
  };
  const preview = (entry: (typeof before)[number]): DetectionView | null => {
    if (!entry) return null;
    const { capture, analysis } = entry;
    const artifact = document.artifacts.find((a) => a.capture_id === capture.capture_id && a.artifact_type === "obstacle_preview");
    return {
      id: `${capture.capture_id}:obstacle`, captureId: capture.capture_id, cameraRole: "top",
      phase: capture.phase, zone: capture.logical_zone_number,
      capturedAt: timeFormatter(capture.display_timezone).format(new Date(capture.captured_at_utc)),
      type: "obstacle", ratio: analysis.ratio_fraction!, maskUrl: artifact?.public_url ?? null,
      maskObjectKey: artifact?.object_key ?? null,
    };
  };
  const beforeRatio = aggregate(before);
  const afterRatio = aggregate(after);
  const complete = beforeRatio != null && afterRatio != null;
  const reduction = complete ? beforeRatio - afterRatio : null;
  return {
    beforeRatio, afterRatio, reduction, complete,
    improvement: reduction != null && beforeRatio! > 0 ? reduction / beforeRatio! : null,
    beforeCount: before.filter(Boolean).length, afterCount: after.filter(Boolean).length,
    pairs: before.map((entry, i) => ({ zone: i + 1, before: preview(entry), after: preview(after[i]) })),
  };
}

const timeFormatter = (timezone: string) => new Intl.DateTimeFormat("ko-KR", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
  timeZone: timezone,
});

export function buildRunViews(document: InspectionExport): RunView[] {
  return document.runs.map((run) => {
    const summary = document.run_summaries.find((item) => item.run_id === run.run_id);
    const captures = document.captures.filter((capture) => capture.run_id === run.run_id);
    const capturesById = new Map(captures.map((capture) => [capture.capture_id, capture]));
    const issues: AnalysisIssueView[] = [];
    const detections = document.analyses.flatMap((analysis): DetectionView[] => {
      const capture = capturesById.get(analysis.capture_id);
      if (!capture) return [];
      const cameraRole = capture.camera_role ?? "side";
      const capturedAt = timeFormatter(capture.display_timezone).format(new Date(capture.captured_at_utc));
      if (capture.processing_status === "error") {
        issues.push({
          id: `${capture.capture_id}:${analysis.defect_type}:capture_error`, captureId: capture.capture_id,
          cameraRole, phase: capture.phase, zone: capture.logical_zone_number, capturedAt, type: analysis.defect_type,
          status: "capture_error", message: "캡처 처리에 실패해 분석 결과를 신뢰할 수 없습니다.",
        });
        return [];
      }
      if (analysis.status !== "ready") {
        issues.push({
          id: `${capture.capture_id}:${analysis.defect_type}:${analysis.status}`, captureId: capture.capture_id,
          cameraRole, phase: capture.phase, zone: capture.logical_zone_number, capturedAt, type: analysis.defect_type,
          status: analysis.status,
          message: analysis.status === "disabled" ? "이 캡처에서는 분석 모델이 비활성화되었습니다." : "모델 분석 중 오류가 발생했습니다.",
        });
        return [];
      }
      if (!analysis.detected || analysis.ratio_fraction == null) return [];
      const artifactType = `${analysis.defect_type}_preview`;
      const artifact = document.artifacts.find(
        (item) => item.capture_id === capture.capture_id && item.artifact_type === artifactType,
      );
      if (!artifact?.public_url) {
        issues.push({
          id: `${capture.capture_id}:${analysis.defect_type}:missing_preview`, captureId: capture.capture_id,
          cameraRole, phase: capture.phase, zone: capture.logical_zone_number, capturedAt, type: analysis.defect_type,
          status: "missing_preview", message: "검출 수치는 있지만 마스크 이미지가 아직 동기화되지 않았습니다.",
        });
      }
      return [{
        id: `${capture.capture_id}:${analysis.defect_type}`,
        captureId: capture.capture_id,
        cameraRole,
        phase: capture.phase,
        zone: capture.logical_zone_number,
        capturedAt,
        type: analysis.defect_type,
        ratio: analysis.ratio_fraction,
        maskUrl: artifact?.public_url ?? null,
        maskObjectKey: artifact?.object_key ?? null,
      }];
    });
    const startedAt = run.started_at_utc
      ? timeFormatter(run.display_timezone).format(new Date(run.started_at_utc)).slice(0, 5)
      : "—";
    return {
      id: run.run_id,
      date: run.local_date ?? "",
      startedAt,
      timezone: run.display_timezone,
      status: run.status,
      beforeRatio: summary?.before_ratio_fraction ?? null,
      afterRatio: summary?.after_ratio_fraction ?? null,
      reduction: summary?.absolute_reduction_fraction ?? null,
      improvement: summary?.relative_improvement_fraction ?? null,
      summaryComplete: summary?.summary_complete ?? false,
      summaryIsDemo: summary?.display_mode === "user_requested_demo_swap",
      summaryDisplayNote: summary?.display_note ?? null,
      railSectionTarget: run.capture_target,
      initialRailSectionCount: summary?.initial_capture_count ?? 0,
      rescanRailSectionCount: summary?.rescan_capture_count ?? 0,
      detections,
      issues,
      failureReason: run.failure_reason,
      craneId: run.crane_id?.trim() || "UNASSIGNED",
      craneLabel: run.crane_label?.trim() || "크레인 미지정",
      obstacleComparison: buildObstacleComparison(document, run.run_id, run.capture_target),
    };
  });
}
