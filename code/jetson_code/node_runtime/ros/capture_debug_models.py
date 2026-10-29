"""Build and validate only the full-frame models required by a capture role."""
from __future__ import annotations

import math

import numpy as np

from .deployment import (
    _thresholds,
    _validate_profile,
    close_detectors,
    read_parameters,
    require_tf32_baseline,
    validate_engine,
)


def create_capture_debug_detectors(node, role):
    """Build only the models required for SIDE, FRONT, or foreign-only capture."""
    if role not in ("front", "side", "foreign"):
        raise ValueError("capture_role must be front, side, or foreign.")
    require_tf32_baseline()
    primary_names = (("capture_rust_engine", "capture_rust_sha256", "capture_rust_profile")
                     if role == "side" else
                     ("obstacle_engine", "obstacle_sha256", "obstacle_confidence_threshold"))
    crack_names = (() if role == "foreign" else (
        "capture_crack_engine", "capture_crack_sha256",
        "capture_crack_probability_threshold", "capture_crack_min_component_pixels",
    ))
    names = primary_names + crack_names
    values = read_parameters(node, names)
    primary = "capture_rust" if role == "side" else "obstacle"
    if role == "side":
        _validate_profile(values, "capture")
    else:
        confidence = values["obstacle_confidence_threshold"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("Obstacle confidence threshold must be finite and between 0 and 1.")
    if role != "foreign":
        threshold, components = _thresholds(values, "capture")
    # Validate every required identity before allocating any CUDA execution context.
    detector_names = (primary,) if role == "foreign" else (primary, "capture_crack")
    plans = {
        name: validate_engine(values[f"{name}_engine"], values[f"{name}_sha256"])
        for name in detector_names
    }
    created = []
    try:
        if role == "side":
            from rust_detector import RustDetector, TEACHER_PROFILE
            created.append(RustDetector(plans[primary][0], TEACHER_PROFILE, plans[primary][1]))
        else:
            from obstacle_detector import ObstacleDetector
            created.append(ObstacleDetector(*plans[primary], float(confidence)))
        if role != "foreign":
            from hrsegnet_crack_detector import HrSegNetCrackDetector
            created.append(HrSegNetCrackDetector(*plans["capture_crack"], threshold, components, role="capture"))
        return (created[0], created[1] if len(created) > 1 else None)
    except Exception:
        close_detectors(created)
        raise


def validate_primary_result(result, role):
    """Reject a wrong execution role or malformed full-frame result before drawing."""
    if getattr(result, "status", None) != "ready":
        raise ValueError("Primary capture model did not return a ready result.")
    if role == "side":
        if not str(result.method).startswith("deeplabv3plus-tensorrt/teacher/"):
            raise ValueError("SIDE capture requires the teacher rust model.")
        class_map = result.class_map
        if not isinstance(class_map, np.ndarray) or class_map.shape != (720, 1280) or class_map.dtype != np.uint8 or np.any(class_map > 3):
            raise ValueError("Teacher capture requires a native uint8 class map with classes 0 through 3.")
    else:
        from obstacle_detector import ObstacleDetectionResult, DETECTOR_PREFIX
        if not isinstance(result, ObstacleDetectionResult) or not result.method.startswith(DETECTOR_PREFIX):
            role_name = "FOREIGN" if role == "foreign" else "FRONT"
            raise ValueError(f"{role_name} capture requires the approved obstacle detector result.")


def validate_capture_crack_result(result):
    from hrsegnet_crack_detector import CAPTURE_DETECTOR_PREFIX
    if getattr(result, "status", None) != "ready" or not str(getattr(result, "method", "")).startswith(CAPTURE_DETECTOR_PREFIX):
        raise ValueError("Crack capture requires a ready capture HrSegNet result.")
    mask = result.mask
    if not isinstance(mask, np.ndarray) or mask.shape != (720, 1280) or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
        raise ValueError("Crack capture requires a native uint8 binary mask.")
    pixels = int(np.count_nonzero(mask))
    ratio = pixels / mask.size
    if not math.isfinite(result.crack_ratio) or abs(float(result.crack_ratio) - ratio) > 1e-9 or result.crack_pixels != pixels or result.inspected_pixels != mask.size or result.detected != (pixels > 0):
        raise ValueError("Crack capture measurements do not match the native filtered mask.")
    return ratio
