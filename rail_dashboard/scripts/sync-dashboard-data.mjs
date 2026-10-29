import { copyFile, mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const TABLES = ["runs", "model_provenance", "captures", "analyses", "artifacts", "run_summaries"];
const PREVIEW_TYPES = new Set(["rust_preview", "crack_preview", "obstacle_preview"]);

function assertRecord(value, label) {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function assertArray(document, key, filename) {
  if (!Array.isArray(document[key])) throw new Error(`${filename}: ${key} must be an array.`);
}

function assertUnique(rows, keyOf, label) {
  const seen = new Set();
  for (const row of rows) {
    const key = keyOf(row);
    if (typeof key !== "string" || !key || seen.has(key)) {
      throw new Error(`Duplicate or missing ${label}: ${String(key)}`);
    }
    seen.add(key);
  }
}

function withinRoot(root, candidate, label) {
  const relative = path.relative(root, candidate);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`${label} escapes its allowed root.`);
  }
}

function publicAssetUrl(objectKey) {
  return `/data/assets/${objectKey.split("/").map(encodeURIComponent).join("/")}`;
}

export async function syncInspectionData({ sourceRoot, outputFile }) {
  const resolvedSourceRoot = path.resolve(sourceRoot);
  const resolvedOutputFile = path.resolve(outputFile);
  const sourceAssetRoot = path.dirname(resolvedSourceRoot);
  const outputDataRoot = path.dirname(resolvedOutputFile);
  const outputAssetRoot = path.join(outputDataRoot, "assets");
  const runsDirectory = path.join(resolvedSourceRoot, "runs");

  let filenames = [];
  try {
    filenames = (await readdir(runsDirectory))
      .filter((filename) => filename.toLowerCase().endsWith(".json"))
      .sort();
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }

  const combined = {
    schema_version: 1,
    exported_at_utc: new Date().toISOString(),
    runs: [],
    model_provenance: [],
    captures: [],
    analyses: [],
    artifacts: [],
    run_summaries: [],
  };

  for (const filename of filenames) {
    const document = assertRecord(
      JSON.parse(await readFile(path.join(runsDirectory, filename), "utf8")),
      filename,
    );
    if (document.schema_version !== 1) throw new Error(`${filename}: unsupported schema_version.`);
    for (const table of TABLES) assertArray(document, table, filename);
    for (const table of TABLES) {
      if (table !== "captures") {
        combined[table].push(...document[table]);
        continue;
      }
      for (const captureValue of document.captures) {
        const capture = assertRecord(captureValue, `${filename}: capture`);
        const cameraRole = capture.camera_role === undefined ? "side" : capture.camera_role;
        if (cameraRole !== "side" && cameraRole !== "top") {
          throw new Error(`${filename}: invalid camera_role for capture ${String(capture.capture_id)}.`);
        }
        combined.captures.push({ ...capture, camera_role: cameraRole });
      }
    }
  }

  assertUnique(combined.runs, (row) => row.run_id, "run_id");
  assertUnique(combined.model_provenance, (row) => row.provenance_id, "provenance_id");
  assertUnique(combined.captures, (row) => row.capture_id, "capture_id");
  assertUnique(combined.analyses, (row) => `${row.capture_id}:${row.defect_type}`, "analysis key");
  assertUnique(combined.artifacts, (row) => row.artifact_id, "artifact_id");
  assertUnique(combined.run_summaries, (row) => row.run_id, "summary run_id");

  const runIds = new Set(combined.runs.map((row) => row.run_id));
  const captureIds = new Set(combined.captures.map((row) => row.capture_id));
  for (const capture of combined.captures) {
    if (!runIds.has(capture.run_id)) throw new Error(`Capture ${capture.capture_id} references an unknown run.`);
  }
  for (const analysis of combined.analyses) {
    if (!captureIds.has(analysis.capture_id)) throw new Error(`Analysis references unknown capture ${analysis.capture_id}.`);
    if (!["ready", "disabled", "error"].includes(analysis.status)) throw new Error(`Invalid analysis status for ${analysis.capture_id}.`);
    if (analysis.status === "ready") {
      if (typeof analysis.ratio_fraction !== "number" || analysis.ratio_fraction < 0 || analysis.ratio_fraction > 1) {
        throw new Error(`Invalid ratio for ${analysis.capture_id}:${analysis.defect_type}.`);
      }
      if (analysis.detected !== (analysis.positive_pixels > 0)) {
        throw new Error(`Detected/count mismatch for ${analysis.capture_id}:${analysis.defect_type}.`);
      }
    }
  }

  for (const artifact of combined.artifacts) {
    if (!captureIds.has(artifact.capture_id)) throw new Error(`Artifact references unknown capture ${artifact.capture_id}.`);
    artifact.public_url = null;
    if (!PREVIEW_TYPES.has(artifact.artifact_type)) continue;
    if (typeof artifact.object_key !== "string" || !artifact.object_key) throw new Error(`Preview ${artifact.artifact_id} has no object key.`);
    const sourcePath = path.resolve(sourceAssetRoot, ...artifact.object_key.split("/"));
    withinRoot(sourceAssetRoot, sourcePath, `Preview ${artifact.artifact_id}`);
    const sourceInfo = await stat(sourcePath);
    if (!sourceInfo.isFile()) throw new Error(`Preview ${artifact.artifact_id} is not a file.`);
    const targetPath = path.resolve(outputAssetRoot, ...artifact.object_key.split("/"));
    withinRoot(outputAssetRoot, targetPath, `Preview target ${artifact.artifact_id}`);
    await mkdir(path.dirname(targetPath), { recursive: true });
    await copyFile(sourcePath, targetPath);
    artifact.public_url = publicAssetUrl(artifact.object_key);
  }

  combined.runs.sort((left, right) => String(right.started_at_utc ?? "").localeCompare(String(left.started_at_utc ?? "")));
  await mkdir(outputDataRoot, { recursive: true });
  const temporaryFile = `${resolvedOutputFile}.${process.pid}.tmp`;
  await writeFile(temporaryFile, `${JSON.stringify(combined, null, 2)}\n`, "utf8");
  await rename(temporaryFile, resolvedOutputFile);
  return { runs: combined.runs.length, captures: combined.captures.length, outputFile: resolvedOutputFile };
}

function argumentValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const entryPoint = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : "";
if (import.meta.url === entryPoint) {
  const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const sourceRoot = argumentValue("--source")
    ?? process.env.RAIL_DASHBOARD_EXPORT_ROOT
    ?? path.resolve(projectRoot, "../code/outputs/dashboard");
  const outputFile = argumentValue("--output")
    ?? path.resolve(projectRoot, "public/data/inspection-export.json");
  const result = await syncInspectionData({ sourceRoot, outputFile });
  process.stdout.write(`Dashboard data synced: ${result.runs} runs, ${result.captures} camera records -> ${result.outputFile}\n`);
}
