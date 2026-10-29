from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from inspection_report import PIPELINE_VERSION, RAIL_SECTION_TARGET


SCHEMA_VERSION = 1
DISPLAY_TIMEZONE = "Asia/Seoul"
KST = timezone(timedelta(hours=9), name=DISPLAY_TIMEZONE)
RUST_COLORS_BGR = {
    1: (45, 180, 242),
    2: (35, 105, 230),
    3: (35, 35, 185),
}
CRACK_COLOR_BGR = (220, 205, 20)
PREVIEW_ALPHA = 0.58
SIDE_CAMERA_ROLE = "side"
TOP_CAMERA_ROLE = "top"
CAMERA_ROLES = (SIDE_CAMERA_ROLE, TOP_CAMERA_ROLE)
# [크레인 식별] 대시보드는 검사 기록의 crane_id/crane_label로 장비 목록을 만든다.
# 이 로봇이 설치된 크레인을 runtime.env의 RAIL_ROBOT_CRANE_ID/LABEL로 바꿀 수 있다.
DEFAULT_CRANE_ID = "CRANE-01"
DEFAULT_CRANE_LABEL = "1번 크레인"


def crane_identity() -> tuple[str, str]:
    crane_id = os.environ.get("RAIL_ROBOT_CRANE_ID", "").strip() or DEFAULT_CRANE_ID
    label = os.environ.get("RAIL_ROBOT_CRANE_LABEL", "").strip() or DEFAULT_CRANE_LABEL
    if not crane_id.replace("-", "").replace("_", "").isalnum() or len(crane_id) > 64 or len(label) > 64:
        raise ValueError("RAIL_ROBOT_CRANE_ID must be 1-64 letters, digits, - or _; label up to 64 chars.")
    return crane_id, label


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Dashboard preview frame must be a uint8 NumPy array.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Dashboard preview frame must be an HxWx3 BGR image.")


def _blend_selected_pixels(
    frame: np.ndarray,
    selected: np.ndarray,
    color_bgr: tuple[int, int, int],
) -> None:
    if not np.any(selected):
        return
    source = frame[selected].astype(np.float32)
    color = np.asarray(color_bgr, dtype=np.float32)
    frame[selected] = np.rint(
        source * (1.0 - PREVIEW_ALPHA) + color * PREVIEW_ALPHA
    ).astype(np.uint8)


def render_rust_mask_preview(frame: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    """Return the original frame with only the rust class mask blended on top."""

    _validate_frame(frame)
    if not isinstance(class_map, np.ndarray) or class_map.shape != frame.shape[:2]:
        raise ValueError("Rust class map must match the dashboard preview frame.")
    if class_map.dtype != np.uint8:
        raise ValueError("Rust class map must use uint8 class IDs.")
    unique_ids = np.unique(class_map)
    if not np.all(np.isin(unique_ids, np.asarray((0, 1, 2, 3), dtype=np.uint8))):
        raise ValueError("Rust class map contains an unsupported class ID.")

    preview = frame.copy()
    for class_id, color_bgr in RUST_COLORS_BGR.items():
        _blend_selected_pixels(preview, class_map == class_id, color_bgr)
    return preview


def render_crack_mask_preview(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the original frame with only the binary crack mask blended on top."""

    _validate_frame(frame)
    if not isinstance(mask, np.ndarray) or mask.shape != frame.shape[:2]:
        raise ValueError("Crack mask must match the dashboard preview frame.")
    if mask.dtype != np.uint8:
        raise ValueError("Crack mask must use uint8 values.")
    if not np.all(np.isin(np.unique(mask), np.asarray((0, 255), dtype=np.uint8))):
        raise ValueError("Crack mask must contain only 0 and 255.")

    preview = frame.copy()
    _blend_selected_pixels(preview, mask != 0, CRACK_COLOR_BGR)
    return preview


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def obstacle_bbox_measurement(frame: np.ndarray, result) -> tuple[dict, np.ndarray]:
    """Rasterized clipped bbox union; this is an area proxy, not segmentation."""
    from obstacle_detector import DETECTOR_PREFIX

    _validate_frame(frame)
    if not str(result.method).startswith(DETECTOR_PREFIX):
        raise ValueError("Obstacle capture must use the approved detector.")
    height, width = frame.shape[:2]
    union = np.zeros((height, width), dtype=bool)
    preview = frame.copy()
    count = 0
    for detection in result.detections:
        coords = np.asarray(detection.box_xyxy, dtype=float)
        if coords.shape != (4,) or not np.all(np.isfinite(coords)):
            raise ValueError("Obstacle bbox coordinates must be finite xyxy.")
        left, top = np.floor(coords[:2]).astype(int)
        right, bottom = np.ceil(coords[2:]).astype(int)
        left, right = np.clip((left, right), 0, width)
        top, bottom = np.clip((top, bottom), 0, height)
        if right <= left or bottom <= top:
            continue
        union[top:bottom, left:right] = True
        # NumPy borders keep preview generation available in offline exporters.
        preview[top:min(top + 2, bottom), left:right] = (30, 180, 255)
        preview[max(top, bottom - 2):bottom, left:right] = (30, 180, 255)
        preview[top:bottom, left:min(left + 2, right)] = (30, 180, 255)
        preview[top:bottom, max(left, right - 2):right] = (30, 180, 255)
        count += 1
    pixels = int(np.count_nonzero(union))
    return {
        "defect_type": "obstacle", "status": "ready", "detected": count > 0,
        "positive_pixels": pixels, "inspected_pixels": int(union.size),
        "ratio_fraction": pixels / union.size, "metric": "bbox_area_ratio",
        "detection_count": count, "detector_method": str(result.method),
        "confidence_threshold": float(result.confidence_threshold),
        "overlap_policy": "suppress_box_crack_overlap_gte_0.01",
        "grade_pixel_counts": None,
    }, preview


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded_ok, encoded = cv2.imencode(
            ".png", image, (cv2.IMWRITE_PNG_COMPRESSION, 3)
        )
    except cv2.error as exc:
        raise RuntimeError(f"Could not encode dashboard PNG {path.name}.") from exc
    if not encoded_ok:
        raise RuntimeError(f"Could not encode dashboard PNG {path.name}.")
    payload = encoded.tobytes()

    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("xb") as destination:
            written = destination.write(payload)
            if written != len(payload):
                raise OSError(
                    f"Could not write complete dashboard PNG ({written}/{len(payload)})."
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return _sha256_bytes(payload)


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
            newline="\n",
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(document, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _aware_local(captured_at: datetime) -> datetime:
    if captured_at.tzinfo is None:
        return captured_at.replace(tzinfo=KST)
    return captured_at.astimezone(KST)


def _object_key(path: Path, artifact_root: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    root = artifact_root.expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Dashboard artifact must remain under {root}; got {resolved}."
        ) from exc
    return relative.as_posix()


def _artifact(
    *,
    artifact_id: str,
    capture_id: str,
    artifact_type: str,
    path: Path,
    artifact_root: Path,
    width: int,
    height: int,
    media_type: str,
    sha256: str | None = None,
) -> dict:
    return {
        "artifact_id": artifact_id,
        "capture_id": capture_id,
        "artifact_type": artifact_type,
        "object_key": _object_key(path, artifact_root),
        "public_url": None,
        "media_type": media_type,
        "width": width,
        "height": height,
        "sha256": sha256 or _sha256_file(path),
    }


def _new_document(workbook, run_id: str, report_filename: str) -> dict:
    crane_id, crane_label = crane_identity()
    rust_provenance_id = f"{run_id}:capture_rust"
    crack_provenance_id = f"{run_id}:capture_crack"
    provenance = [
        {
            "provenance_id": rust_provenance_id,
            "run_id": run_id,
            "role": "capture_rust",
            "model_filename": getattr(workbook, "capture_model_filename", None),
            "model_sha256": getattr(workbook, "capture_model_sha256", None),
            "detector_method": getattr(workbook, "capture_detector", None),
            "probability_threshold": None,
            "min_component_pixels": None,
            "preprocessing": "BGR->RGB float32/255 ImageNet mean/std",
            "input_contract": "images FP32 [1,3,720,1280]",
            "output_contract": "logits FP32 [1,4,720,1280]",
        }
    ]
    if getattr(workbook, "crack_detector", None) is not None:
        provenance.append(
            {
                "provenance_id": crack_provenance_id,
                "run_id": run_id,
                "role": "capture_crack",
                "model_filename": getattr(workbook, "crack_model_filename", None),
                "model_sha256": getattr(workbook, "crack_model_sha256", None),
                "detector_method": getattr(workbook, "crack_detector", None),
                "probability_threshold": getattr(
                    workbook, "crack_probability_threshold", None
                ),
                "min_component_pixels": getattr(
                    workbook, "capture_crack_min_component_pixels", None
                ),
                "preprocessing": "BGR->RGB float32; RGB/127.5-1",
                "input_contract": "images FP32 [1,3,720,1280]",
                "output_contract": "crack_logits FP32 [1,2,720,1280]",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "exported_at_utc": None,
        "source_report": {"filename": report_filename},
        "runs": [
            {
                "run_id": run_id,
                "pipeline_version": PIPELINE_VERSION,
                "started_at_utc": None,
                "finished_at_utc": None,
                "display_timezone": DISPLAY_TIMEZONE,
                "local_date": None,
                "status": "in_progress",
                # Schema v1 keeps this key for compatibility; it counts rail
                # sections, not raw SIDE/TOP camera files.
                "capture_target": RAIL_SECTION_TARGET,
                "failure_reason": None,
                "crane_id": crane_id,
                "crane_label": crane_label,
            }
        ],
        "model_provenance": provenance,
        "captures": [],
        "analyses": [],
        "artifacts": [],
        "run_summaries": [],
    }


def _load_or_create_document(
    manifest_path: Path,
    workbook,
    run_id: str,
    report_filename: str,
) -> dict:
    if not manifest_path.exists():
        return _new_document(workbook, run_id, report_filename)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read dashboard export {manifest_path}.") from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Dashboard export schema version does not match this runtime.")
    runs = document.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or runs[0].get("run_id") != run_id:
        raise ValueError("Dashboard export run identity is invalid.")
    captures = document.get("captures")
    if not isinstance(captures, list):
        raise ValueError("Dashboard export captures collection is invalid.")
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("Dashboard export capture row is invalid.")
        role = capture.get("camera_role", SIDE_CAMERA_ROLE)
        if role not in CAMERA_ROLES:
            raise ValueError(f"Unsupported dashboard camera role: {role!r}.")
        capture["camera_role"] = role
    # 크레인 필드가 생기기 전에 시작된 진행 중 기록도 같은 로봇의 장비로 채운다.
    crane_id, crane_label = crane_identity()
    runs[0].setdefault("crane_id", crane_id)
    runs[0].setdefault("crane_label", crane_label)
    return document


def _rebuild_summary(document: dict) -> None:
    run = document["runs"][0]
    run_id = run["run_id"]
    captures_by_id = {capture["capture_id"]: capture for capture in document["captures"]}
    rust_by_capture = {
        analysis["capture_id"]: analysis
        for analysis in document["analyses"]
        if analysis["defect_type"] == "rust" and analysis["status"] == "ready"
    }

    phase_totals: dict[str, tuple[int, int, int]] = {}
    for phase in ("initial", "rescan"):
        phase_capture_ids = [
            capture_id
            for capture_id, capture in captures_by_id.items()
            if capture["phase"] == phase and capture["processing_status"] == "ready"
            and capture.get("camera_role", SIDE_CAMERA_ROLE) == SIDE_CAMERA_ROLE
        ]
        measurements = [
            rust_by_capture[capture_id]
            for capture_id in phase_capture_ids
            if capture_id in rust_by_capture
        ]
        positive = sum(int(item["positive_pixels"]) for item in measurements)
        inspected = sum(int(item["inspected_pixels"]) for item in measurements)
        phase_totals[phase] = (len(phase_capture_ids), positive, inspected)

    initial_count, before_positive, before_inspected = phase_totals["initial"]
    rescan_count, after_positive, after_inspected = phase_totals["rescan"]
    before_ratio = before_positive / before_inspected if before_inspected else None
    after_ratio = after_positive / after_inspected if after_inspected else None
    summary_complete = (
        initial_count == RAIL_SECTION_TARGET
        and rescan_count == RAIL_SECTION_TARGET
    )
    reduction = (
        before_ratio - after_ratio
        if summary_complete and before_ratio is not None and after_ratio is not None
        else None
    )
    improvement = None
    if reduction is not None and before_ratio is not None:
        if before_ratio == 0.0:
            improvement = 0.0 if after_ratio == 0.0 else None
        else:
            improvement = reduction / before_ratio

    document["run_summaries"] = [
        {
            "run_id": run_id,
            "initial_capture_count": initial_count,
            "rescan_capture_count": rescan_count,
            "before_positive_pixels": before_positive,
            "before_inspected_pixels": before_inspected,
            "before_ratio_fraction": before_ratio,
            "after_positive_pixels": after_positive,
            "after_inspected_pixels": after_inspected,
            "after_ratio_fraction": after_ratio,
            "absolute_reduction_fraction": reduction,
            "relative_improvement_fraction": improvement,
            "summary_complete": summary_complete,
        }
    ]
    captured_times = sorted(capture["captured_at_utc"] for capture in document["captures"])
    run["started_at_utc"] = captured_times[0] if captured_times else None
    if document["captures"]:
        run["local_date"] = document["captures"][0]["captured_local_date"]


def finalize_dashboard_run(
    *,
    workbook,
    output_directory: str | Path,
    status: str,
    failure_reason: str | None = None,
    finished_at: datetime | None = None,
) -> Path | None:
    """Record the mission outcome separately from rail-section summary readiness.

    A missing manifest means no capture was exported, so there is nothing to finalize.
    Dashboard finalization is deliberately separate from aggregation: the configured
    initial and rescan rail sections make the summary calculable, but only the STM DONE
    path makes the mission complete.
    """

    if status not in ("complete", "failed"):
        raise ValueError("Dashboard run status must be complete or failed.")
    if status == "complete" and failure_reason is not None:
        raise ValueError("A completed dashboard run cannot have a failure reason.")
    if status == "failed" and not failure_reason:
        raise ValueError("A failed dashboard run requires a failure reason.")

    output_directory = Path(output_directory).expanduser()
    artifact_root = output_directory.parent
    dashboard_root = artifact_root / "dashboard"
    report_path = Path(workbook.path).expanduser()
    run_key = hashlib.sha256(
        str(report_path.resolve(strict=False)).encode("utf-8")
    ).hexdigest()[:24]
    manifest_path = dashboard_root / "runs" / f"{run_key}.json"
    if not manifest_path.exists():
        return None

    run_id = f"run_{run_key}"
    document = _load_or_create_document(
        manifest_path, workbook, run_id, report_path.name
    )
    run = document["runs"][0]
    if status == "complete":
        summary = document.get("run_summaries", [])
        if (
            len(summary) != 1
            or summary[0].get("run_id") != run_id
            or not summary[0].get("summary_complete")
        ):
            raise ValueError(
                "Dashboard mission cannot be completed before the "
                f"{RAIL_SECTION_TARGET}+{RAIL_SECTION_TARGET} "
                "rail-section summary is ready."
            )

    completed_local = _aware_local(finished_at or datetime.now(KST))
    run["status"] = status
    run["finished_at_utc"] = completed_local.astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    run["failure_reason"] = failure_reason
    document["exported_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    _write_json_atomic(manifest_path, document)
    return manifest_path


def export_capture_record(
    *,
    frame: np.ndarray,
    rust_result,
    crack_result,
    workbook,
    capture_path: str | Path,
    raw_capture_path: str | Path,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    captured_at: datetime,
    output_directory: str | Path,
) -> Path:
    """Write two mask-only PNGs and one normalized, DB-importable JSON snapshot."""

    _validate_frame(frame)
    if phase not in ("initial", "rescan", "manual"):
        raise ValueError(f"Unsupported dashboard capture phase: {phase}")
    if phase in ("initial", "rescan") and (
        isinstance(phase_sequence, bool)
        or not isinstance(phase_sequence, int)
        or not 1 <= phase_sequence <= RAIL_SECTION_TARGET
    ):
        raise ValueError(
            "Automatic dashboard rail-section sequence must be from 1 to "
            f"{RAIL_SECTION_TARGET}."
        )
    if phase == "manual":
        phase_sequence = None

    class_map = getattr(rust_result, "class_map", None)
    if not isinstance(class_map, np.ndarray) or class_map.shape != frame.shape[:2]:
        raise ValueError("Dashboard rust result is missing its full-frame class map.")

    output_directory = Path(output_directory).expanduser()
    artifact_root = output_directory.parent
    dashboard_root = artifact_root / "dashboard"
    report_path = Path(workbook.path).expanduser()
    run_key = hashlib.sha256(str(report_path.resolve(strict=False)).encode("utf-8")).hexdigest()[:24]
    manifest_path = dashboard_root / "runs" / f"{run_key}.json"
    run_id = f"run_{run_key}"
    document = _load_or_create_document(
        manifest_path, workbook, run_id, report_path.name
    )

    if any(
        capture["phase"] == phase
        and capture["phase_sequence"] == phase_sequence
        and capture.get("camera_role", SIDE_CAMERA_ROLE) == SIDE_CAMERA_ROLE
        and phase != "manual"
        for capture in document["captures"]
    ):
        raise ValueError(
            f"Dashboard export already contains SIDE {phase} capture "
            f"{phase_sequence}."
        )

    capture_id = f"cap_{uuid.uuid4().hex}"
    media_directory = dashboard_root / "media" / run_id
    rust_preview_path = media_directory / f"{capture_id}_rust.png"
    rust_preview_sha = _write_png(
        rust_preview_path, render_rust_mask_preview(frame, class_map)
    )
    crack_preview_path = None
    crack_preview_sha = None
    if crack_result is not None:
        crack_preview_path = media_directory / f"{capture_id}_crack.png"
        crack_preview_sha = _write_png(
            crack_preview_path,
            render_crack_mask_preview(frame, crack_result.mask),
        )

    local_time = _aware_local(captured_at)
    utc_time = local_time.astimezone(timezone.utc)
    height, width = frame.shape[:2]
    raw_capture_path = Path(raw_capture_path)
    capture_path = Path(capture_path)
    document["captures"].append(
        {
            "capture_id": capture_id,
            "run_id": run_id,
            "phase": phase,
            "phase_sequence": phase_sequence,
            "logical_zone_number": phase_sequence,
            "camera_role": SIDE_CAMERA_ROLE,
            "trigger": str(trigger),
            "captured_at_utc": utc_time.isoformat().replace("+00:00", "Z"),
            "captured_local_date": local_time.date().isoformat(),
            "display_timezone": DISPLAY_TIMEZONE,
            "width": width,
            "height": height,
            "processing_status": "ready",
            "raw_image_key": _object_key(raw_capture_path, artifact_root),
        }
    )

    counts = np.bincount(class_map.ravel(), minlength=4)[:4]
    inspected_pixels = int(class_map.size)
    rust_pixels = int(inspected_pixels - int(counts[0]))
    rust_provenance_id = f"{run_id}:capture_rust"
    document["analyses"].append(
        {
            "capture_id": capture_id,
            "defect_type": "rust",
            "status": "ready",
            "detected": rust_pixels > 0,
            "positive_pixels": rust_pixels,
            "inspected_pixels": inspected_pixels,
            "ratio_fraction": rust_pixels / inspected_pixels,
            "detector_method": str(rust_result.method),
            "provenance_id": rust_provenance_id,
            "overlap_policy": "crack_priority_v1",
            "grade_pixel_counts": {
                "good": int(counts[0]),
                "fair": int(counts[1]),
                "poor": int(counts[2]),
                "severe": int(counts[3]),
            },
        }
    )
    if crack_result is None:
        document["analyses"].append(
            {
                "capture_id": capture_id,
                "defect_type": "crack",
                "status": "disabled",
                "detected": None,
                "positive_pixels": None,
                "inspected_pixels": None,
                "ratio_fraction": None,
                "detector_method": None,
                "provenance_id": None,
                "overlap_policy": None,
                "grade_pixel_counts": None,
            }
        )
    else:
        crack_pixels = int(np.count_nonzero(crack_result.mask))
        crack_inspected = int(crack_result.mask.size)
        document["analyses"].append(
            {
                "capture_id": capture_id,
                "defect_type": "crack",
                "status": "ready",
                "detected": crack_pixels > 0,
                "positive_pixels": crack_pixels,
                "inspected_pixels": crack_inspected,
                "ratio_fraction": crack_pixels / crack_inspected,
                "detector_method": str(crack_result.method),
                "provenance_id": f"{run_id}:capture_crack",
                "overlap_policy": None,
                "grade_pixel_counts": None,
            }
        )

    document["artifacts"].extend(
        [
            _artifact(
                artifact_id=f"{capture_id}:raw",
                capture_id=capture_id,
                artifact_type="raw",
                path=raw_capture_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/jpeg",
            ),
            _artifact(
                artifact_id=f"{capture_id}:analyzed",
                capture_id=capture_id,
                artifact_type="analyzed",
                path=capture_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/jpeg",
            ),
            _artifact(
                artifact_id=f"{capture_id}:rust_preview",
                capture_id=capture_id,
                artifact_type="rust_preview",
                path=rust_preview_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/png",
                sha256=rust_preview_sha,
            ),
        ]
    )
    if crack_preview_path is not None:
        document["artifacts"].append(
            _artifact(
                artifact_id=f"{capture_id}:crack_preview",
                capture_id=capture_id,
                artifact_type="crack_preview",
                path=crack_preview_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/png",
                sha256=crack_preview_sha,
            )
        )

    _rebuild_summary(document)
    document["exported_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    _write_json_atomic(manifest_path, document)
    return manifest_path


def export_top_crack_record(
    *,
    frame: np.ndarray,
    crack_result,
    workbook,
    capture_path: str | Path,
    raw_capture_path: str | Path,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    captured_at: datetime,
    output_directory: str | Path,
    obstacle_result=None,
    obstacle_provenance: dict | None = None,
) -> Path:
    """Export one TOP-camera crack capture without inventing rust data."""

    _validate_frame(frame)
    if phase not in ("initial", "rescan", "manual"):
        raise ValueError(f"Unsupported dashboard capture phase: {phase}")
    if phase in ("initial", "rescan") and (
        isinstance(phase_sequence, bool)
        or not isinstance(phase_sequence, int)
        or not 1 <= phase_sequence <= RAIL_SECTION_TARGET
    ):
        raise ValueError(
            "Automatic dashboard rail-section sequence must be from 1 to "
            f"{RAIL_SECTION_TARGET}."
        )
    if phase == "manual":
        phase_sequence = None
    if getattr(crack_result, "status", "ready") != "ready":
        raise ValueError("Dashboard top crack result must be ready.")
    crack_mask = getattr(crack_result, "mask", None)
    if not isinstance(crack_mask, np.ndarray) or crack_mask.shape != frame.shape[:2]:
        raise ValueError("Dashboard top crack result is missing its full-frame mask.")

    output_directory = Path(output_directory).expanduser()
    artifact_root = output_directory.parent
    dashboard_root = artifact_root / "dashboard"
    report_path = Path(workbook.path).expanduser()
    run_key = hashlib.sha256(
        str(report_path.resolve(strict=False)).encode("utf-8")
    ).hexdigest()[:24]
    manifest_path = dashboard_root / "runs" / f"{run_key}.json"
    run_id = f"run_{run_key}"
    document = _load_or_create_document(
        manifest_path,
        workbook,
        run_id,
        report_path.name,
    )

    if any(
        capture["phase"] == phase
        and capture["phase_sequence"] == phase_sequence
        and capture.get("camera_role", SIDE_CAMERA_ROLE) == TOP_CAMERA_ROLE
        and phase != "manual"
        for capture in document["captures"]
    ):
        raise ValueError(
            f"Dashboard export already contains TOP {phase} capture "
            f"{phase_sequence}."
        )

    capture_id = f"cap_{uuid.uuid4().hex}"
    media_directory = dashboard_root / "media" / run_id
    crack_preview_path = media_directory / f"{capture_id}_crack.png"
    crack_preview_sha = _write_png(
        crack_preview_path,
        render_crack_mask_preview(frame, crack_mask),
    )

    local_time = _aware_local(captured_at)
    utc_time = local_time.astimezone(timezone.utc)
    height, width = frame.shape[:2]
    raw_capture_path = Path(raw_capture_path)
    capture_path = Path(capture_path)
    document["captures"].append(
        {
            "capture_id": capture_id,
            "run_id": run_id,
            "phase": phase,
            "phase_sequence": phase_sequence,
            "logical_zone_number": phase_sequence,
            "camera_role": TOP_CAMERA_ROLE,
            "trigger": str(trigger),
            "captured_at_utc": utc_time.isoformat().replace("+00:00", "Z"),
            "captured_local_date": local_time.date().isoformat(),
            "display_timezone": DISPLAY_TIMEZONE,
            "width": width,
            "height": height,
            "processing_status": "ready",
            "raw_image_key": _object_key(raw_capture_path, artifact_root),
        }
    )

    crack_pixels = int(np.count_nonzero(crack_mask))
    crack_inspected = int(crack_mask.size)
    document["analyses"].append(
        {
            "capture_id": capture_id,
            "defect_type": "crack",
            "status": "ready",
            "detected": crack_pixels > 0,
            "positive_pixels": crack_pixels,
            "inspected_pixels": crack_inspected,
            "ratio_fraction": crack_pixels / crack_inspected,
            "detector_method": str(crack_result.method),
            "provenance_id": f"{run_id}:capture_crack",
            "overlap_policy": None,
            "grade_pixel_counts": None,
        }
    )
    document["artifacts"].extend(
        [
            _artifact(
                artifact_id=f"{capture_id}:raw",
                capture_id=capture_id,
                artifact_type="raw",
                path=raw_capture_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/jpeg",
            ),
            _artifact(
                artifact_id=f"{capture_id}:analyzed",
                capture_id=capture_id,
                artifact_type="analyzed",
                path=capture_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/jpeg",
            ),
            _artifact(
                artifact_id=f"{capture_id}:crack_preview",
                capture_id=capture_id,
                artifact_type="crack_preview",
                path=crack_preview_path,
                artifact_root=artifact_root,
                width=width,
                height=height,
                media_type="image/png",
                sha256=crack_preview_sha,
            ),
        ]
    )

    if obstacle_result is not None:
        measurement, preview = obstacle_bbox_measurement(frame, obstacle_result)
        provenance_id = f"{run_id}:capture_obstacle"
        measurement.update(capture_id=capture_id, provenance_id=provenance_id)
        document["analyses"].append(measurement)
        if not any(p["provenance_id"] == provenance_id for p in document["model_provenance"]):
            document["model_provenance"].append({
                **(obstacle_provenance or {}), "provenance_id": provenance_id,
                "run_id": run_id, "role": "capture_obstacle",
                "detector_method": str(obstacle_result.method),
            })
        obstacle_path = media_directory / f"{capture_id}_obstacle.png"
        digest = _write_png(obstacle_path, preview)
        document["artifacts"].append(_artifact(
            artifact_id=f"{capture_id}:obstacle_preview", capture_id=capture_id,
            artifact_type="obstacle_preview", path=obstacle_path,
            artifact_root=artifact_root, width=width, height=height,
            media_type="image/png", sha256=digest,
        ))
    _rebuild_summary(document)
    document["exported_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    _write_json_atomic(manifest_path, document)
    return manifest_path
