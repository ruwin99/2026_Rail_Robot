import {
  index,
  integer,
  primaryKey,
  real,
  sqliteTable,
  text,
  uniqueIndex,
} from "drizzle-orm/sqlite-core";

export const inspectionRuns = sqliteTable(
  "inspection_runs",
  {
    runId: text("run_id").primaryKey(),
    craneId: text("crane_id"),
    craneLabel: text("crane_label"),
    schemaVersion: integer("schema_version").notNull(),
    pipelineVersion: integer("pipeline_version").notNull(),
    startedAtUtc: text("started_at_utc"),
    finishedAtUtc: text("finished_at_utc"),
    displayTimezone: text("display_timezone").notNull(),
    localDate: text("local_date"),
    status: text("status", { enum: ["in_progress", "complete", "failed"] }).notNull(),
    captureTarget: integer("capture_target").notNull(),
    failureReason: text("failure_reason"),
  },
  (table) => [
    index("inspection_runs_local_date_idx").on(table.localDate),
    index("inspection_runs_crane_date_idx").on(table.craneId, table.localDate),
  ],
);

export const modelProvenance = sqliteTable(
  "model_provenance",
  {
    provenanceId: text("provenance_id").primaryKey(),
    runId: text("run_id").notNull().references(() => inspectionRuns.runId, { onDelete: "cascade" }),
    role: text("role", { enum: ["capture_rust", "capture_crack"] }).notNull(),
    modelFilename: text("model_filename"),
    modelSha256: text("model_sha256"),
    detectorMethod: text("detector_method"),
    probabilityThreshold: real("probability_threshold"),
    minComponentPixels: integer("min_component_pixels"),
    preprocessing: text("preprocessing"),
    inputContract: text("input_contract"),
    outputContract: text("output_contract"),
  },
  (table) => [uniqueIndex("model_provenance_run_role_uq").on(table.runId, table.role)],
);

export const captures = sqliteTable(
  "captures",
  {
    captureId: text("capture_id").primaryKey(),
    runId: text("run_id").notNull().references(() => inspectionRuns.runId, { onDelete: "cascade" }),
    cameraRole: text("camera_role", { enum: ["side", "top"] }).notNull().default("side"),
    phase: text("phase", { enum: ["initial", "rescan", "manual"] }).notNull(),
    phaseSequence: integer("phase_sequence"),
    logicalZoneNumber: integer("logical_zone_number"),
    trigger: text("trigger").notNull(),
    capturedAtUtc: text("captured_at_utc").notNull(),
    capturedLocalDate: text("captured_local_date").notNull(),
    displayTimezone: text("display_timezone").notNull(),
    width: integer("width").notNull(),
    height: integer("height").notNull(),
    processingStatus: text("processing_status", { enum: ["ready", "error"] }).notNull(),
    rawImageKey: text("raw_image_key").notNull(),
  },
  (table) => [
    index("captures_run_phase_idx").on(table.runId, table.phase),
    index("captures_local_date_idx").on(table.capturedLocalDate),
    uniqueIndex("captures_run_phase_sequence_camera_role_uq").on(
      table.runId,
      table.phase,
      table.phaseSequence,
      table.cameraRole,
    ),
  ],
);

export const analyses = sqliteTable(
  "analyses",
  {
    captureId: text("capture_id").notNull().references(() => captures.captureId, { onDelete: "cascade" }),
    defectType: text("defect_type", { enum: ["rust", "crack"] }).notNull(),
    status: text("status", { enum: ["ready", "disabled", "error"] }).notNull(),
    detected: integer("detected", { mode: "boolean" }),
    positivePixels: integer("positive_pixels"),
    inspectedPixels: integer("inspected_pixels"),
    ratioFraction: real("ratio_fraction"),
    detectorMethod: text("detector_method"),
    provenanceId: text("provenance_id").references(() => modelProvenance.provenanceId),
    overlapPolicy: text("overlap_policy"),
    gradePixelCountsJson: text("grade_pixel_counts_json"),
  },
  (table) => [primaryKey({ columns: [table.captureId, table.defectType] })],
);

export const artifacts = sqliteTable(
  "artifacts",
  {
    artifactId: text("artifact_id").primaryKey(),
    captureId: text("capture_id").notNull().references(() => captures.captureId, { onDelete: "cascade" }),
    artifactType: text("artifact_type", {
      enum: ["raw", "analyzed", "rust_preview", "crack_preview"],
    }).notNull(),
    objectKey: text("object_key").notNull(),
    publicUrl: text("public_url"),
    mediaType: text("media_type").notNull(),
    width: integer("width").notNull(),
    height: integer("height").notNull(),
    sha256: text("sha256").notNull(),
  },
  (table) => [uniqueIndex("artifacts_capture_type_uq").on(table.captureId, table.artifactType)],
);

export const runSummaries = sqliteTable("run_summaries", {
  runId: text("run_id").primaryKey().references(() => inspectionRuns.runId, { onDelete: "cascade" }),
  initialCaptureCount: integer("initial_capture_count").notNull(),
  rescanCaptureCount: integer("rescan_capture_count").notNull(),
  beforePositivePixels: integer("before_positive_pixels").notNull(),
  beforeInspectedPixels: integer("before_inspected_pixels").notNull(),
  beforeRatioFraction: real("before_ratio_fraction"),
  afterPositivePixels: integer("after_positive_pixels").notNull(),
  afterInspectedPixels: integer("after_inspected_pixels").notNull(),
  afterRatioFraction: real("after_ratio_fraction"),
  absoluteReductionFraction: real("absolute_reduction_fraction"),
  relativeImprovementFraction: real("relative_improvement_fraction"),
  summaryComplete: integer("summary_complete", { mode: "boolean" }).notNull(),
});

export const plannerEvents = sqliteTable(
  "planner_events",
  {
    eventId: text("event_id").primaryKey(),
    eventDate: text("event_date").notNull(),
    startTime: text("start_time").notNull(),
    endTime: text("end_time").notNull(),
    title: text("title").notNull(),
    category: text("category", {
      enum: ["port_operation", "maintenance", "inspection", "safety"],
    }).notNull(),
    location: text("location"),
    notes: text("notes"),
    status: text("status", { enum: ["scheduled", "in_progress", "done"] }).notNull(),
    source: text("source", { enum: ["starter", "user"] }).notNull(),
    createdAtUtc: text("created_at_utc").notNull(),
    updatedAtUtc: text("updated_at_utc").notNull(),
  },
  (table) => [index("planner_events_date_idx").on(table.eventDate)],
);
