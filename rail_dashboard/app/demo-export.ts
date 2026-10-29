import type { AnalysisStatus, InspectionExport, Phase } from "./data-contract";

type DemoCapture = {
  id: string;
  runId: string;
  date: string;
  timeUtc: string;
  phase: Phase;
  zone: number | null;
  rustPixels: number;
  crackPixels: number;
  rustStatus?: AnalysisStatus;
  crackStatus?: AnalysisStatus;
  rustPreview?: string;
  crackPreview?: string;
};

const inspected = 921_600;
const completeRun = "run_demo_20260820_0914";
const incompleteRun = "run_demo_20260819_1628";

const demoCaptures: DemoCapture[] = [
  { id: "cap_2001", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:15:02Z", phase: "initial", zone: 1, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2002", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:15:18Z", phase: "initial", zone: 2, rustPixels: 65_802, crackPixels: 0, rustPreview: "/demo/rust-web-demo.jpg" },
  { id: "cap_2003", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:15:27Z", phase: "initial", zone: 3, rustPixels: 0, crackPixels: 15_575, crackPreview: "/demo/crack-web-demo.jpg" },
  { id: "cap_2004", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:15:42Z", phase: "initial", zone: 4, rustPixels: 170_865, crackPixels: 0, rustPreview: "/demo/rust-web-demo.jpg" },
  { id: "cap_2010", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:15:55Z", phase: "initial", zone: 5, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2011", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:16:08Z", phase: "initial", zone: 6, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2012", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:16:21Z", phase: "initial", zone: 7, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2013", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:16:34Z", phase: "initial", zone: 8, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2014", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:16:47Z", phase: "initial", zone: 9, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2015", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:17:00Z", phase: "initial", zone: 10, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2005", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:18:28Z", phase: "rescan", zone: 1, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2006", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:18:44Z", phase: "rescan", zone: 2, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2007", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:18:58Z", phase: "rescan", zone: 3, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2008", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:19:12Z", phase: "rescan", zone: 4, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2016", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:19:25Z", phase: "rescan", zone: 5, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2017", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:19:38Z", phase: "rescan", zone: 6, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2018", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:19:51Z", phase: "rescan", zone: 7, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2019", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:20:04Z", phase: "rescan", zone: 8, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2020", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:20:17Z", phase: "rescan", zone: 9, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2021", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:20:30Z", phase: "rescan", zone: 10, rustPixels: 0, crackPixels: 0 },
  { id: "cap_2009", runId: completeRun, date: "2026-08-20", timeUtc: "2026-08-20T00:20:45Z", phase: "manual", zone: null, rustPixels: 4_120, crackPixels: 0, rustPreview: "/demo/rust-web-demo.jpg", crackStatus: "error" },
  { id: "cap_1901", runId: incompleteRun, date: "2026-08-19", timeUtc: "2026-08-19T07:29:08Z", phase: "initial", zone: 1, rustPixels: 48_200, crackPixels: 0, rustPreview: "/demo/rust-web-demo.jpg" },
  { id: "cap_1902", runId: incompleteRun, date: "2026-08-19", timeUtc: "2026-08-19T07:29:24Z", phase: "initial", zone: 2, rustPixels: 0, crackPixels: 0 },
];

const artifacts = demoCaptures.flatMap((capture) => {
  const rows: InspectionExport["artifacts"] = [];
  if (capture.rustPreview) rows.push({
    artifact_id: `${capture.id}:rust_preview`, capture_id: capture.id, artifact_type: "rust_preview",
    object_key: capture.rustPreview.replace(/^\//, ""), public_url: capture.rustPreview,
    media_type: "image/jpeg", width: 910, height: 607, sha256: "a".repeat(64),
  });
  if (capture.crackPreview) rows.push({
    artifact_id: `${capture.id}:crack_preview`, capture_id: capture.id, artifact_type: "crack_preview",
    object_key: capture.crackPreview.replace(/^\//, ""), public_url: capture.crackPreview,
    media_type: "image/jpeg", width: 1024, height: 768, sha256: "b".repeat(64),
  });
  return rows;
});

export const demoExport: InspectionExport = {
  schema_version: 1,
  exported_at_utc: "2026-08-20T01:00:00Z",
  runs: [
    { run_id: completeRun, pipeline_version: 18, started_at_utc: "2026-08-20T00:14:00Z", finished_at_utc: "2026-08-20T00:20:30Z", display_timezone: "Asia/Seoul", local_date: "2026-08-20", status: "complete", capture_target: 10, failure_reason: null, crane_id: "CRANE-01", crane_label: "1호 크레인" },
    { run_id: incompleteRun, pipeline_version: 18, started_at_utc: "2026-08-19T07:28:00Z", finished_at_utc: null, display_timezone: "Asia/Seoul", local_date: "2026-08-19", status: "in_progress", capture_target: 10, failure_reason: null, crane_id: "CRANE-02", crane_label: "2호 크레인" },
  ],
  model_provenance: [],
  captures: demoCaptures.map((capture) => ({
    capture_id: capture.id, run_id: capture.runId, phase: capture.phase,
    phase_sequence: capture.phase === "manual" ? null : capture.zone, logical_zone_number: capture.zone, trigger: capture.phase === "manual" ? "manual" : "CAMERA_CAPTURE",
    captured_at_utc: capture.timeUtc, captured_local_date: capture.date, display_timezone: "Asia/Seoul",
    width: 1280, height: 720, processing_status: "ready", raw_image_key: `captures/raw/${capture.id}.jpg`,
  })),
  analyses: demoCaptures.flatMap((capture) => {
    const rustStatus = capture.rustStatus ?? "ready";
    const crackStatus = capture.crackStatus ?? "ready";
    return [
      { capture_id: capture.id, defect_type: "rust" as const, status: rustStatus, detected: rustStatus === "ready" ? capture.rustPixels > 0 : null, positive_pixels: rustStatus === "ready" ? capture.rustPixels : null, inspected_pixels: rustStatus === "ready" ? inspected : null, ratio_fraction: rustStatus === "ready" ? capture.rustPixels / inspected : null, detector_method: "capture-rust-demo", provenance_id: null, overlap_policy: "crack_priority_v1", grade_pixel_counts: null },
      { capture_id: capture.id, defect_type: "crack" as const, status: crackStatus, detected: crackStatus === "ready" ? capture.crackPixels > 0 : null, positive_pixels: crackStatus === "ready" ? capture.crackPixels : null, inspected_pixels: crackStatus === "ready" ? inspected : null, ratio_fraction: crackStatus === "ready" ? capture.crackPixels / inspected : null, detector_method: "capture-crack-demo", provenance_id: null, overlap_policy: null, grade_pixel_counts: null },
    ];
  }),
  artifacts,
  run_summaries: [
    { run_id: completeRun, initial_capture_count: 10, rescan_capture_count: 10, before_positive_pixels: 236_667, before_inspected_pixels: 9_216_000, before_ratio_fraction: 236_667 / 9_216_000, after_positive_pixels: 0, after_inspected_pixels: 9_216_000, after_ratio_fraction: 0, absolute_reduction_fraction: 236_667 / 9_216_000, relative_improvement_fraction: 1, summary_complete: true },
    { run_id: incompleteRun, initial_capture_count: 2, rescan_capture_count: 0, before_positive_pixels: 48_200, before_inspected_pixels: 1_843_200, before_ratio_fraction: 48_200 / 1_843_200, after_positive_pixels: 0, after_inspected_pixels: 0, after_ratio_fraction: null, absolute_reduction_fraction: null, relative_improvement_fraction: null, summary_complete: false },
  ],
};
