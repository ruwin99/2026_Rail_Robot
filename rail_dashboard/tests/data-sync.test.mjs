import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { syncInspectionData } from "../scripts/sync-dashboard-data.mjs";

test("syncs real DB-shaped exports and publishes only mask previews", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "rail-dashboard-sync-"));
  try {
    const sourceRoot = path.join(root, "outputs", "dashboard");
    const runsRoot = path.join(sourceRoot, "runs");
    const previewPath = path.join(sourceRoot, "media", "run_real", "cap_1_rust.png");
    const outputFile = path.join(root, "public", "data", "inspection-export.json");
    await mkdir(runsRoot, { recursive: true });
    await mkdir(path.dirname(previewPath), { recursive: true });
    await writeFile(previewPath, Buffer.from("lossless-preview"));

    const document = {
      schema_version: 1,
      exported_at_utc: "2026-08-20T01:00:00Z",
      runs: [{ run_id: "run_real", crane_id: "CRANE-03", crane_label: "3호 크레인", pipeline_version: 16, started_at_utc: "2026-08-20T00:00:00Z", finished_at_utc: "2026-08-20T00:01:00Z", display_timezone: "Asia/Seoul", local_date: "2026-08-20", status: "failed", capture_target: 4, failure_reason: "camera timeout" }],
      model_provenance: [],
      captures: [
        { capture_id: "cap_1", run_id: "run_real", phase: "manual", phase_sequence: null, logical_zone_number: null, trigger: "manual", captured_at_utc: "2026-08-20T00:00:30Z", captured_local_date: "2026-08-20", display_timezone: "Asia/Seoul", width: 1280, height: 720, processing_status: "ready", raw_image_key: "captures/raw/manual.jpg", exporter_note: "preserve-me" },
        { capture_id: "cap_top", run_id: "run_real", camera_role: "top", phase: "manual", phase_sequence: null, logical_zone_number: null, trigger: "manual", captured_at_utc: "2026-08-20T00:00:30Z", captured_local_date: "2026-08-20", display_timezone: "Asia/Seoul", width: 1280, height: 720, processing_status: "ready", raw_image_key: "captures/raw/manual_top.jpg" },
      ],
      analyses: [
        { capture_id: "cap_1", defect_type: "rust", status: "ready", detected: true, positive_pixels: 2, inspected_pixels: 4, ratio_fraction: 0.5, detector_method: "rust", provenance_id: null, overlap_policy: "crack_priority_v1", grade_pixel_counts: null },
        { capture_id: "cap_1", defect_type: "crack", status: "error", detected: null, positive_pixels: null, inspected_pixels: null, ratio_fraction: null, detector_method: "crack", provenance_id: null, overlap_policy: null, grade_pixel_counts: null },
      ],
      artifacts: [{ artifact_id: "cap_1:rust_preview", capture_id: "cap_1", artifact_type: "rust_preview", object_key: "dashboard/media/run_real/cap_1_rust.png", public_url: null, media_type: "image/png", width: 1280, height: 720, sha256: "a".repeat(64) }],
      run_summaries: [{ run_id: "run_real", initial_capture_count: 0, rescan_capture_count: 0, before_positive_pixels: 0, before_inspected_pixels: 0, before_ratio_fraction: null, after_positive_pixels: 0, after_inspected_pixels: 0, after_ratio_fraction: null, absolute_reduction_fraction: null, relative_improvement_fraction: null, summary_complete: false }],
    };
    document.analyses.push({ capture_id: "cap_top", defect_type: "obstacle", status: "ready", detected: false, positive_pixels: 0, inspected_pixels: 921600, ratio_fraction: 0, metric: "bbox_area_ratio", detection_count: 0 });
    document.artifacts.push({ artifact_id: "cap_top:obstacle_preview", capture_id: "cap_top", artifact_type: "obstacle_preview", object_key: "dashboard/media/run_real/cap_top_obstacle.png", public_url: null, media_type: "image/png", width: 1280, height: 720, sha256: "b".repeat(64) });
    await writeFile(path.join(path.dirname(previewPath), "cap_top_obstacle.png"), Buffer.from("clean-front-preview"));
    await writeFile(path.join(runsRoot, "real.json"), JSON.stringify(document), "utf8");

    const result = await syncInspectionData({ sourceRoot, outputFile });
    assert.equal(result.runs, 1);
    assert.equal(result.captures, 2);
    const synced = JSON.parse(await readFile(outputFile, "utf8"));
    assert.equal(synced.runs[0].status, "failed");
    assert.equal(synced.runs[0].failure_reason, "camera timeout");
    assert.equal(synced.runs[0].crane_id, "CRANE-03");
    assert.equal(synced.runs[0].crane_label, "3호 크레인");
    assert.equal(synced.runs[0].capture_target, 4);
    assert.equal(synced.captures[0].phase, "manual");
    assert.equal(synced.captures[0].camera_role, "side");
    assert.equal(synced.captures[0].exporter_note, "preserve-me");
    assert.equal(synced.captures[1].camera_role, "top");
    assert.equal(synced.analyses[1].status, "error");
    assert.equal(synced.artifacts[0].public_url, "/data/assets/dashboard/media/run_real/cap_1_rust.png");
    assert.equal(synced.artifacts[1].public_url, "/data/assets/dashboard/media/run_real/cap_top_obstacle.png");
    assert.equal(synced.analyses[2].ratio_fraction, 0);
    assert.equal(
      await readFile(path.join(root, "public", "data", "assets", "dashboard", "media", "run_real", "cap_1_rust.png"), "utf8"),
      "lossless-preview",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects capture camera roles outside side and top, including null", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "rail-dashboard-invalid-role-"));
  try {
    const sourceRoot = path.join(root, "outputs", "dashboard");
    const runsRoot = path.join(sourceRoot, "runs");
    const outputFile = path.join(root, "public", "data", "inspection-export.json");
    await mkdir(runsRoot, { recursive: true });
    const document = {
      schema_version: 1,
      exported_at_utc: "2026-08-20T01:00:00Z",
      runs: [{ run_id: "run_invalid" }],
      model_provenance: [],
      captures: [{ capture_id: "cap_invalid", run_id: "run_invalid", camera_role: "rear" }],
      analyses: [],
      artifacts: [],
      run_summaries: [],
    };

    for (const invalidRole of ["rear", null]) {
      document.captures[0].camera_role = invalidRole;
      await writeFile(path.join(runsRoot, "invalid.json"), JSON.stringify(document), "utf8");
      await assert.rejects(
        syncInspectionData({ sourceRoot, outputFile }),
        /invalid camera_role for capture cap_invalid/,
      );
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("writes an explicit empty dataset when no capture exports exist", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "rail-dashboard-empty-"));
  try {
    const outputFile = path.join(root, "public", "data", "inspection-export.json");
    const result = await syncInspectionData({ sourceRoot: path.join(root, "missing"), outputFile });
    assert.equal(result.runs, 0);
    const synced = JSON.parse(await readFile(outputFile, "utf8"));
    assert.deepEqual(synced.runs, []);
    assert.deepEqual(synced.analyses, []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
