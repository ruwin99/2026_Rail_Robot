from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial

from dashboard_export import (
    export_capture_record,
    export_top_crack_record,
    finalize_dashboard_run,
)
from firebase_uploader import (
    FirebaseLivePreviewPublisher,
    FirebaseUploadError,
    upload_dashboard_manifest_from_environment,
)
from crack_detector import (
    CRACK_COLOR_BGR,
    DEFAULT_MIN_COMPONENT_PIXELS,
    DEFAULT_PROBABILITY_THRESHOLD,
    REALTIME_DETECTOR_PREFIX as REALTIME_CRACK_DETECTOR_PREFIX,
    REALTIME_PROFILE as CRACK_REALTIME_PROFILE,
    CrackDetector,
    annotate_cracks,
)
from inspection_report import (
    INITIAL_PHASE,
    MANUAL_PHASE,
    RAIL_SECTION_TARGET,
    RESCAN_PHASE,
    InspectionWorkbook,
)
from hrsegnet_crack_detector import (
    CAPTURE_DETECTOR_PREFIX as HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX,
    DEFAULT_MIN_COMPONENT_PIXELS as HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS,
    DEFAULT_PROBABILITY_THRESHOLD as HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD,
    DETECTOR_PREFIX as HRSEGNET_CRACK_DETECTOR_PREFIX,
    HrSegNetCrackDetector,
    HrSegNetStageTiming,
)
from multitask_detector import (
    CRACK_METHOD_PREFIX as MULTITASK_CRACK_METHOD_PREFIX,
    OptimizedMultitaskDetector,
)
from optimized_rust_detector import OptimizedRustDetector
from obstacle_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD as OBSTACLE_DEFAULT_CONFIDENCE_THRESHOLD,
    DETECTOR_PREFIX as OBSTACLE_DETECTOR_PREFIX,
    ObstacleDetector,
    ObstacleDetectionResult,
    ObstacleStageTiming,
)
from rust_detector import (
    CLASS_COLORS_BGR,
    LetterboxTransform,
    STUDENT_PROFILE,
    TEACHER_PROFILE,
    RustDetector,
    RustStageTiming,
    annotate,
    result_from_class_map,
)


from node_runtime.camera import (
    LatestFrameCamera,
    open_latest_frame_camera,
    validate_distinct_camera_devices,
    realtime_camera_frame_is_fresh,
    read_stopped_capture_frame,
    read_stopped_capture_pair,
)
from node_runtime.control import (
    RealtimeActuatorHoldSnapshot,
    RealtimeActuatorDecisionHold,
    RealtimeActuatorArbiter,
    highest_rust_grade,
    water_control_blocked_by_rust,
    validated_crack_ratio,
    cleaning_blocked_by_crack,
    obstacle_detected_in_control_roi,
    update_cleaning_actuators,
    publish_realtime_actuator_decision,
    clear_realtime_actuator_role,
    maintain_waiting_role_control,
    ControlOffDiagnostic,
    select_control_off_reason,
    DualControlOffTransitionLogger,
    classify_runtime_control_off,
    current_dual_actuator_state,
    RealtimeControlDecision,
    FrontControlDecision,
    FrontControlHistory,
    RealtimeControlHistory,
    side_outcome_requires_immediate_off,
    front_outcome_requires_immediate_off,
    build_control_off_diagnostic,
    force_cleaning_safe_off,
    force_cleaning_pairs_safe_off,
)
from node_runtime.data import (
    CaptureAnalysisTask,
    CameraFrame,
    DualRealtimeFrameRead,
    TimedRealtimeResult,
    RealtimeInferenceOutcome,
    TopRealtimeInferenceOutcome,
)
from node_runtime.display import (
    LatestFrameDisplay,
    open_latest_frame_display,
    graphical_display_environment_available,
    annotate_realtime_control_results,
    annotate_top_realtime_results,
    annotate_front_capture_results,
    draw_roi_guide,
    draw_realtime_roi_guide,
    stack_dual_camera_displays,
    draw_realtime_result_summary,
    draw_realtime_test_statistics,
    format_realtime_test_statistics,
    RealtimeDisplayCache,
)
from node_runtime.inspection import (
    firebase_upload_is_required,
    db_test_manual_capture_is_enabled,
    publish_dashboard_manifest,
    validate_db_test_dashboard_manifest,
    dashboard_manifest_for_publish,
    save_capture,
    queue_capture_for_analysis,
    prepare_capture_analysis_task,
    queue_dual_capture_for_analysis,
    prioritize_capture_crack_over_rust,
    capture_and_analyze,
    capture_top_crack_only,
    CaptureAnalysisWorker,
)
from node_runtime.uart import (
    read_uart_messages,
    send_uart_test_command,
    enter_uart_actuator_test,
    CleanerController,
    PumpController,
    send_capture_ok,
)
from node_runtime.vision import (
    water_control_roi_bounds,
    crack_stop_roi_bounds,
    extract_realtime_control_rois,
    detect_realtime_control,
    AlternatingTopRealtimeInference,
    AlternatingRealtimeInference,
    dual_realtime_inputs_are_fresh,
    realtime_role_inputs_are_fresh,
    suppress_front_obstacles_overlapping_cracks,
)
from node_runtime.settings import *


def db_test_capture_phase(
    initial_count: int,
    rescan_count: int,
) -> tuple[str, int]:
    """Map the next accepted DB-test keypress onto the fixed initial+rescan section sequence."""

    if initial_count < AUTOMATIC_RAIL_SECTION_TARGET:
        return INITIAL_PHASE, initial_count + 1
    if rescan_count < AUTOMATIC_RAIL_SECTION_TARGET:
        return RESCAN_PHASE, rescan_count + 1
    raise RuntimeError(f"DB test already accepted all {2 * AUTOMATIC_RAIL_SECTION_TARGET} captures.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B0441 camera rust-detection preview")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--capture-camera-device",
        help=(
            "stable V4L2 device for single-camera --capture-test; "
            "mutually exclusive with explicit SIDE/TOP dual-camera devices"
        ),
    )
    parser.add_argument(
        "--side-camera-device",
        help=(
            "stable V4L2 by-path for the existing side camera; when provided, "
            "enables explicit dual-camera role binding"
        ),
    )
    parser.add_argument(
        "--top-camera-device",
        help="stable V4L2 by-path for the new top camera",
    )
    parser.add_argument(
        "--obstacle-engine",
        type=Path,
        help=(
            "FP16 TensorRT 10.3 YOLO obstacle plan for the fixed 1280x720 "
            f"top-camera frame (recommended: {DEFAULT_OBSTACLE_ENGINE})"
        ),
    )
    parser.add_argument(
        "--obstacle-engine-sha256",
        help="required approved SHA-256 for the top-camera obstacle plan",
    )
    parser.add_argument(
        "--obstacle-confidence-threshold",
        type=float,
        default=OBSTACLE_DEFAULT_CONFIDENCE_THRESHOLD,
        help=(
            "top-camera obstacle confidence threshold "
            f"(default: {OBSTACLE_DEFAULT_CONFIDENCE_THRESHOLD:g})"
        ),
    )
    parser.add_argument(
        "--student-engine",
        "--engine",
        dest="student_engine",
        type=Path,
        default=None,
        help=(
            "realtime student TensorRT plan "
            f"(default: {DEFAULT_STUDENT_ENGINE})"
        ),
    )
    parser.add_argument(
        "--student-engine-sha256",
        "--engine-sha256",
        dest="student_engine_sha256",
        help=(
            "approved student engine digest; required whenever UART is enabled; "
            "with --realtime-multitask-engine, selects the separate student "
            "for realtime rust while multitask supplies crack"
        ),
    )
    parser.add_argument(
        "--optimized-student-engine",
        type=Path,
        help=(
            "realtime student TensorRT plan with GPU rust argmax/count/finite "
            "outputs; mutually exclusive with --student-engine"
        ),
    )
    parser.add_argument(
        "--optimized-student-engine-sha256",
        help="approved digest for the GPU-postprocessed student rust plan",
    )
    parser.add_argument(
        "--teacher-engine",
        type=Path,
        default=DEFAULT_TEACHER_ENGINE,
        help=(
            "capture teacher TensorRT plan "
            f"(default: {DEFAULT_TEACHER_ENGINE})"
        ),
    )
    parser.add_argument(
        "--teacher-engine-sha256",
        help="approved teacher engine digest; required whenever UART is enabled",
    )
    parser.add_argument(
        "--capture-crack-engine",
        type=Path,
        help=(
            "LEGACY/UNSUPPORTED: capture BGCrack was replaced by "
            "--capture-hrsegnet-crack-engine"
        ),
    )
    parser.add_argument(
        "--capture-crack-engine-sha256",
        help="LEGACY/UNSUPPORTED capture BGCrack digest",
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-engine",
        type=Path,
        help=(
            "original HrSegNet-B32 TensorRT 10.3 plan for the native "
            "1280x720 capture frame; no resize or padding; approved path: "
            f"{DEFAULT_CAPTURE_HRSEGNET_CRACK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-engine-sha256",
        help="required approved SHA-256 for the capture HrSegNet-B32 plan",
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-probability-threshold",
        type=float,
        help=(
            "capture crack probability threshold converted to the equivalent "
            "class-1 minus class-0 logit margin (default: 0.55)"
        ),
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-min-component-pixels",
        type=int,
        help="capture HrSegNet 8-connected component minimum (default: 20 pixels)",
    )
    parser.add_argument(
        "--realtime-crack-engine",
        type=Path,
        help=(
            "realtime BGCrack TensorRT plan for the fixed 1280x128 ROI; "
            f"recommended deployment path: {DEFAULT_REALTIME_CRACK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--realtime-crack-engine-sha256",
        help=(
            "approved realtime crack engine digest; required whenever UART is enabled"
        ),
    )
    parser.add_argument(
        "--realtime-multitask-engine",
        type=Path,
        help=(
            "optional optimized realtime rust+crack TensorRT plan with GPU "
            "postprocessed uint8 maps/scalars; a raw 5-channel plan is rejected; "
            f"recommended path: {DEFAULT_REALTIME_MULTITASK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--realtime-multitask-engine-sha256",
        help="required approved digest for the optimized realtime multitask plan",
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-engine",
        type=Path,
        help=(
            "HrSegNet-B32 TensorRT 10.3 plan for the fixed 1280x128 "
            "realtime crack ROI; this never replaces the capture crack model; "
            f"approved deployment path: {DEFAULT_REALTIME_HRSEGNET_CRACK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-engine-sha256",
        help="required approved SHA-256 for the realtime HrSegNet-B32 plan",
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-probability-threshold",
        type=float,
        help=(
            "approved realtime crack probability threshold; converted to an "
            "equivalent class-1 minus class-0 logit margin (default: 0.55)"
        ),
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-min-component-pixels",
        type=int,
        help="approved realtime 8-connected component minimum (default: 20 pixels)",
    )
    parser.add_argument(
        "--no-crack",
        action="store_true",
        help=(
            "TEST ONLY: run realtime cleaning without the crack detector or "
            "fixed y=540:720 crack safety interlock"
        ),
    )
    parser.add_argument(
        "--capture-crack-threshold",
        type=float,
        help="LEGACY/UNSUPPORTED capture BGCrack threshold",
    )
    parser.add_argument(
        "--realtime-crack-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help="realtime crack threshold calibrated on Validation (default: 0.5)",
    )
    parser.add_argument(
        "--capture-crack-min-component-pixels",
        type=int,
        help="LEGACY/UNSUPPORTED capture BGCrack component minimum",
    )
    parser.add_argument(
        "--realtime-crack-min-component-pixels",
        type=int,
        default=DEFAULT_MIN_COMPONENT_PIXELS,
        help="discard smaller realtime crack candidates (default: 20 pixels)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="new XLSX path; defaults to a unique timestamped report",
    )
    parser.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help="UART device (default: /dev/ttyACM0 for ST-LINK USB VCP)",
    )
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument(
        "--no-uart",
        action="store_true",
        help="disable the UART trigger for manual capture testing",
    )
    parser.add_argument(
        "--realtime-test",
        action="store_true",
        help=(
            "TEST ONLY: run the realtime student and crack models from the "
            "camera with simulated actuator output by default; captures and "
            "reports remain disabled"
        ),
    )
    parser.add_argument(
        "--realtime-test-uart",
        action="store_true",
        help=(
            "TEST ONLY: with --realtime-test and two cameras, enter the STM "
            "actuator-test state and apply FRONT/SIDE cleaner and pump commands"
        ),
    )
    parser.add_argument(
        "--dual-timing",
        action="store_true",
        help=(
            "TEST ONLY: after a 10-second warmup, print 5-second dual-camera "
            "loop, inference, frame-age, stale, and ready-to-off statistics"
        ),
    )
    parser.add_argument(
        "--dual-debug-benchmark",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dual-rust-debug-benchmark",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yolo-debug-benchmark",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--capture-test",
        action="store_true",
        help=(
            "TEST ONLY: preview the full camera frame and run only the "
            "capture rust and crack models when S is pressed"
        ),
    )
    parser.add_argument(
        "--front-capture-test",
        action="store_true",
        help=(
            "TEST ONLY: preview the fixed FRONT/TOP camera and run the "
            "obstacle and full-frame capture crack models when S is pressed"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "disable the camera preview and display-only overlays while keeping "
            "camera inference, UART/FSM control, automatic rail sections, and reports"
        ),
    )
    args = parser.parse_args()
    explicit_headless = args.headless
    args.headless_auto = (
        not explicit_headless and not graphical_display_environment_available()
    )
    args.headless = explicit_headless or args.headless_auto
    yolo_debug_benchmark = args.yolo_debug_benchmark
    dual_camera_device_configured = (
        args.side_camera_device is not None
        or (args.top_camera_device is not None and not yolo_debug_benchmark)
    )
    capture_only_test = args.capture_test or args.front_capture_test
    if args.capture_camera_device is not None and not capture_only_test:
        parser.error(
            "--capture-camera-device requires --capture-test or "
            "--front-capture-test"
        )
    if args.capture_camera_device is not None and dual_camera_device_configured:
        parser.error(
            "--capture-camera-device cannot be combined with SIDE/TOP camera devices"
        )
    if dual_camera_device_configured and (
        args.side_camera_device is None or args.top_camera_device is None
    ):
        parser.error(
            "dual-camera operation requires both --side-camera-device and "
            "--top-camera-device"
        )
    args.dual_camera = bool(
        args.side_camera_device is not None and args.top_camera_device is not None
    )
    if yolo_debug_benchmark and args.side_camera_device is not None:
        parser.error("--yolo-debug-benchmark uses only --top-camera-device")
    if yolo_debug_benchmark and args.top_camera_device is None:
        parser.error("--yolo-debug-benchmark requires --top-camera-device")
    if yolo_debug_benchmark and (
        args.capture_test or args.front_capture_test or args.realtime_test
    ):
        parser.error("--yolo-debug-benchmark cannot be combined with other test modes")
    if args.front_capture_test and args.capture_camera_device is None:
        parser.error("--front-capture-test requires --capture-camera-device")
    if args.front_capture_test and args.dual_camera:
        parser.error(
            "--front-capture-test uses one pinned TOP camera, not dual cameras"
        )
    if args.realtime_test_uart and not args.realtime_test:
        parser.error("--realtime-test-uart requires --realtime-test")
    if args.realtime_test_uart and args.no_uart:
        parser.error("--realtime-test-uart cannot be combined with --no-uart")
    if args.realtime_test_uart and not args.dual_camera:
        parser.error(
            "--realtime-test-uart requires explicit side and top camera devices"
        )
    if args.dual_timing and (not args.realtime_test or not args.dual_camera):
        parser.error("--dual-timing requires --realtime-test with two cameras")
    if args.dual_debug_benchmark and (
        not args.realtime_test
        or not args.dual_camera
        or args.realtime_test_uart
        or args.dual_timing
    ):
        parser.error(
            "--dual-debug-benchmark requires a two-camera --realtime-test "
            "without UART or --dual-timing"
        )
    if args.dual_rust_debug_benchmark and (
        not args.realtime_test
        or not args.dual_camera
        or args.realtime_test_uart
        or args.dual_timing
        or args.dual_debug_benchmark
    ):
        parser.error(
            "--dual-rust-debug-benchmark requires a two-camera --realtime-test "
            "without UART, --dual-timing, or --dual-debug-benchmark"
        )
    obstacle_configured = (
        args.obstacle_engine is not None or args.obstacle_engine_sha256 is not None
    )
    if obstacle_configured and not (
        args.dual_camera or args.front_capture_test or yolo_debug_benchmark
    ):
        parser.error(
            "--obstacle-engine options require explicit side and top camera "
            "devices or --front-capture-test"
        )
    if args.obstacle_engine is None and args.obstacle_engine_sha256 is not None:
        parser.error("--obstacle-engine-sha256 requires --obstacle-engine")
    if args.obstacle_engine is not None and args.obstacle_engine_sha256 is None:
        parser.error("--obstacle-engine requires --obstacle-engine-sha256")
    if args.front_capture_test and args.obstacle_engine is None:
        parser.error(
            "--front-capture-test requires --obstacle-engine and its SHA-256"
        )
    if yolo_debug_benchmark and args.obstacle_engine is None:
        parser.error(
            "--yolo-debug-benchmark requires --obstacle-engine and its SHA-256"
        )
    if args.dual_camera and not capture_only_test and args.obstacle_engine is None:
        parser.error(
            "dual-camera realtime operation requires --obstacle-engine and its SHA-256"
        )
    if args.dual_camera and args.no_crack:
        parser.error("dual-camera operation cannot bypass either crack detector")
    if args.capture_test and args.obstacle_engine is not None:
        parser.error(
            "--capture-test does not run the obstacle model; omit --obstacle-engine"
        )
    if not 0.0 < args.obstacle_confidence_threshold < 1.0:
        parser.error("--obstacle-confidence-threshold must be between zero and one")
    if yolo_debug_benchmark:
        return args
    raw_student_configured = (
        args.student_engine is not None or args.student_engine_sha256 is not None
    )
    optimized_student_configured = (
        args.optimized_student_engine is not None
        or args.optimized_student_engine_sha256 is not None
    )
    if raw_student_configured and optimized_student_configured:
        parser.error(
            "raw --student-engine options cannot be combined with optimized "
            "student engine options"
        )
    if (
        args.optimized_student_engine is None
        and args.optimized_student_engine_sha256 is not None
    ):
        parser.error(
            "--optimized-student-engine-sha256 requires "
            "--optimized-student-engine"
        )
    if args.student_engine is None and args.student_engine_sha256 is not None:
        args.student_engine = DEFAULT_STUDENT_ENGINE
    if sum((args.capture_test, args.front_capture_test, args.realtime_test)) > 1:
        parser.error(
            "--capture-test, --front-capture-test, and --realtime-test are "
            "mutually exclusive"
        )
    if capture_only_test and args.headless:
        parser.error(
            "capture tests require a graphical display because S/Q/Esc are "
            "interactive; normal operation and --realtime-test support --headless"
        )
    if capture_only_test and args.no_crack:
        parser.error("capture tests cannot be combined with --no-crack")
    legacy_capture_options = (
        args.capture_crack_engine,
        args.capture_crack_engine_sha256,
        args.capture_crack_threshold,
        args.capture_crack_min_component_pixels,
    )
    if any(value is not None for value in legacy_capture_options):
        parser.error(
            "capture BGCrack options are no longer supported; use the explicit "
            "--capture-hrsegnet-crack-* options"
        )
    capture_hrsegnet_options = (
        args.capture_hrsegnet_crack_engine,
        args.capture_hrsegnet_crack_engine_sha256,
        args.capture_hrsegnet_crack_probability_threshold,
        args.capture_hrsegnet_crack_min_component_pixels,
    )
    capture_hrsegnet_configured = any(
        value is not None for value in capture_hrsegnet_options
    )
    capture_hrsegnet_enabled = args.capture_hrsegnet_crack_engine is not None
    if capture_hrsegnet_configured and not capture_hrsegnet_enabled:
        parser.error(
            "capture HrSegNet threshold, component, and SHA options require "
            "--capture-hrsegnet-crack-engine"
        )
    if capture_hrsegnet_enabled and args.capture_hrsegnet_crack_engine_sha256 is None:
        parser.error(
            "--capture-hrsegnet-crack-engine requires its approved "
            "--capture-hrsegnet-crack-engine-sha256"
        )
    if capture_only_test and not capture_hrsegnet_enabled:
        parser.error("capture tests require --capture-hrsegnet-crack-engine")
    if capture_hrsegnet_enabled:
        if args.capture_hrsegnet_crack_probability_threshold is None:
            args.capture_hrsegnet_crack_probability_threshold = (
                HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD
            )
        if args.capture_hrsegnet_crack_min_component_pixels is None:
            args.capture_hrsegnet_crack_min_component_pixels = (
                HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS
            )
    else:
        args.capture_hrsegnet_crack_probability_threshold = (
            HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD
        )
        args.capture_hrsegnet_crack_min_component_pixels = (
            HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS
        )
    if args.realtime_test and args.no_crack:
        parser.error("--realtime-test cannot be combined with --no-crack")
    if args.no_crack and not args.no_uart:
        parser.error("--no-crack is test-only and requires --no-uart")
    multitask_enabled = args.realtime_multitask_engine is not None
    hrsegnet_options = (
        args.realtime_hrsegnet_crack_engine,
        args.realtime_hrsegnet_crack_engine_sha256,
        args.realtime_hrsegnet_crack_probability_threshold,
        args.realtime_hrsegnet_crack_min_component_pixels,
    )
    hrsegnet_configured = any(value is not None for value in hrsegnet_options)
    hrsegnet_enabled = args.realtime_hrsegnet_crack_engine is not None
    if hrsegnet_configured and not hrsegnet_enabled:
        parser.error(
            "HrSegNet threshold, component, and SHA options require "
            "--realtime-hrsegnet-crack-engine"
        )
    if hrsegnet_enabled and args.realtime_hrsegnet_crack_engine_sha256 is None:
        parser.error(
            "--realtime-hrsegnet-crack-engine requires its approved "
            "--realtime-hrsegnet-crack-engine-sha256"
        )
    if hrsegnet_enabled and (
        multitask_enabled
        or args.no_crack
        or args.realtime_crack_engine is not None
        or args.realtime_crack_engine_sha256 is not None
    ):
        parser.error(
            "--realtime-hrsegnet-crack-engine is mutually exclusive with "
            "--no-crack, BGCrack, and multitask realtime "
            "crack engine options"
        )
    if hrsegnet_enabled:
        if args.realtime_hrsegnet_crack_probability_threshold is None:
            args.realtime_hrsegnet_crack_probability_threshold = (
                HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD
            )
        if args.realtime_hrsegnet_crack_min_component_pixels is None:
            args.realtime_hrsegnet_crack_min_component_pixels = (
                HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS
            )
    if (
        not capture_only_test
        and not multitask_enabled
        and args.optimized_student_engine is None
        and args.student_engine is None
    ):
        args.student_engine = DEFAULT_STUDENT_ENGINE
    if (
        args.realtime_multitask_engine is None
        and args.realtime_multitask_engine_sha256 is not None
    ):
        parser.error(
            "--realtime-multitask-engine-sha256 requires "
            "--realtime-multitask-engine"
        )
    if (
        multitask_enabled
        and not capture_only_test
        and args.realtime_multitask_engine_sha256 is None
    ):
        parser.error(
            "--realtime-multitask-engine requires its approved "
            "--realtime-multitask-engine-sha256"
        )
    if (
        args.realtime_test
        and not multitask_enabled
        and not hrsegnet_enabled
        and args.realtime_crack_engine is None
    ):
        parser.error(
            "--realtime-test requires --realtime-crack-engine or "
            "--realtime-multitask-engine or --realtime-hrsegnet-crack-engine"
        )
    if not capture_only_test and multitask_enabled and (
        args.no_crack
        or args.realtime_crack_engine is not None
        or args.realtime_crack_engine_sha256 is not None
    ):
        parser.error(
            "--realtime-multitask-engine is mutually exclusive with --no-crack "
            "and separate realtime crack engine options"
        )
    if args.no_crack and (
        args.capture_hrsegnet_crack_engine is not None
        or args.capture_hrsegnet_crack_engine_sha256 is not None
        or args.realtime_crack_engine is not None
        or args.realtime_crack_engine_sha256 is not None
    ):
        parser.error(
            "--no-crack cannot be combined with capture/realtime crack engines"
        )
    if args.realtime_crack_engine is None and args.realtime_crack_engine_sha256 is not None:
        parser.error(
            "--realtime-crack-engine-sha256 requires --realtime-crack-engine"
        )
    if (
        not capture_only_test
        and multitask_enabled
        and not args.realtime_test
        and not capture_hrsegnet_enabled
    ):
        parser.error(
            "normal multitask operation still requires "
            "--capture-hrsegnet-crack-engine "
            "for the independent capture/report path"
        )
    if (
        not capture_only_test
        and not multitask_enabled
        and not args.realtime_test
        and not args.no_crack
        and (
            not capture_hrsegnet_enabled
            or (args.realtime_crack_engine is None) == (not hrsegnet_enabled)
        )
    ):
        parser.error(
            "normal operation requires the independent "
            "--capture-hrsegnet-crack-engine "
            "and exactly one realtime crack engine: BGCrack or HrSegNet"
        )
    if not 0.0 < args.realtime_crack_threshold < 1.0:
        parser.error("--realtime-crack-threshold must be between zero and one")
    if args.realtime_crack_min_component_pixels <= 0:
        parser.error("--realtime-crack-min-component-pixels must be positive")
    if capture_hrsegnet_enabled:
        if not 0.0 < args.capture_hrsegnet_crack_probability_threshold < 1.0:
            parser.error(
                "--capture-hrsegnet-crack-probability-threshold must be "
                "between zero and one"
            )
        if args.capture_hrsegnet_crack_min_component_pixels <= 0:
            parser.error(
                "--capture-hrsegnet-crack-min-component-pixels must be positive"
            )
    if hrsegnet_enabled:
        if not 0.0 < args.realtime_hrsegnet_crack_probability_threshold < 1.0:
            parser.error(
                "--realtime-hrsegnet-crack-probability-threshold must be "
                "between zero and one"
            )
        if args.realtime_hrsegnet_crack_min_component_pixels <= 0:
            parser.error(
                "--realtime-hrsegnet-crack-min-component-pixels must be positive"
            )
    if args.report is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.report = DEFAULT_REPORT_DIRECTORY / f"rust_inspection_{timestamp}.xlsx"
    if not capture_only_test and (
        args.realtime_multitask_engine is not None
        or args.realtime_crack_engine is not None
    ):
        parser.error(
            "The bottom-quarter 180-row crack ROI requires "
            "--realtime-hrsegnet-crack-engine. Legacy BGCrack and multitask "
            "plans have an incompatible fixed 128-row crack ROI."
        )
    return args


def read_terminal_command() -> tuple[str | None, bool]:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None, False
    if not readable:
        return None, True

    line = sys.stdin.readline()
    if not line:
        return None, False
    command = line.strip().lower()
    return command or None, True


class DualTimingLogger:
    """Aggregate dual-camera test timings without printing on every loop."""

    _METRIC_NAMES = (
        "loop",
        "side_update",
        "top_update",
        "side_rust",
        "side_crack",
        "top_obstacle",
        "top_crack",
        "side_frame_age",
        "top_frame_age",
    )

    def __init__(
        self,
        *,
        warmup_seconds: float = DUAL_TIMING_WARMUP_SECONDS,
        report_interval_seconds: float = DUAL_TIMING_REPORT_INTERVAL_SECONDS,
        started_at: float | None = None,
    ) -> None:
        if warmup_seconds < 0.0:
            raise ValueError("Dual timing warmup must not be negative.")
        if report_interval_seconds <= 0.0:
            raise ValueError("Dual timing report interval must be positive.")
        if started_at is None:
            started_at = time.monotonic()
        self.warmup_seconds = float(warmup_seconds)
        self.report_interval_seconds = float(report_interval_seconds)
        self.measurement_started_at = float(started_at) + self.warmup_seconds
        self.next_report_at = (
            self.measurement_started_at + self.report_interval_seconds
        )
        self._samples = {name: [] for name in self._METRIC_NAMES}
        self._stale_counts = {"side": 0, "top": 0}
        self._ready_to_off_counts = {"side": 0, "front": 0}
        self._previous_ready = {"side": None, "front": None}

    @staticmethod
    def _append_positive(samples: list[float], seconds: float | None) -> None:
        if seconds is not None and seconds > 0.0:
            samples.append(float(seconds) * 1000.0)

    @staticmethod
    def _format_samples(name: str, samples: list[float]) -> str:
        if not samples:
            return f"{name}_ms(n=0)"
        values = np.asarray(samples, dtype=np.float64)
        p50, p95, p99 = np.percentile(values, (50, 95, 99))
        return (
            f"{name}_ms(n={len(samples)},mean={values.mean():.1f},"
            f"p50={p50:.1f},p95={p95:.1f},p99={p99:.1f},"
            f"max={values.max():.1f})"
        )

    def record(
        self,
        *,
        loop_seconds: float,
        side_update_seconds: float | None,
        top_update_seconds: float | None,
        side_outcome: RealtimeInferenceOutcome | None,
        top_outcome: TopRealtimeInferenceOutcome | None,
        side_frame_age_seconds: float | None,
        top_frame_age_seconds: float | None,
        side_stale: bool,
        top_stale: bool,
        side_ready: bool,
        front_ready: bool,
        now: float | None = None,
    ) -> str | None:
        """Record one loop and return a summary when the 5-second window closes."""

        if now is None:
            now = time.monotonic()
        current_ready = {"side": bool(side_ready), "front": bool(front_ready)}
        if now < self.measurement_started_at:
            self._previous_ready = current_ready
            return None

        self._append_positive(self._samples["loop"], loop_seconds)
        self._append_positive(
            self._samples["side_update"], side_update_seconds
        )
        self._append_positive(self._samples["top_update"], top_update_seconds)
        self._append_positive(
            self._samples["side_frame_age"], side_frame_age_seconds
        )
        self._append_positive(
            self._samples["top_frame_age"], top_frame_age_seconds
        )
        if side_outcome is not None:
            self._append_positive(
                self._samples["side_rust"], side_outcome.rust_seconds
            )
            self._append_positive(
                self._samples["side_crack"], side_outcome.crack_seconds
            )
        if top_outcome is not None:
            self._append_positive(
                self._samples["top_obstacle"], top_outcome.obstacle_seconds
            )
            self._append_positive(
                self._samples["top_crack"], top_outcome.crack_seconds
            )
        self._stale_counts["side"] += int(side_stale)
        self._stale_counts["top"] += int(top_stale)
        for role, ready in current_ready.items():
            if self._previous_ready[role] is True and not ready:
                self._ready_to_off_counts[role] += 1
        self._previous_ready = current_ready

        if now < self.next_report_at:
            return None
        window_seconds = max(
            self.report_interval_seconds,
            now - (self.next_report_at - self.report_interval_seconds),
        )
        parts = [
            f"[DUAL_TIMING] window={window_seconds:.1f}s",
            *(
                self._format_samples(name, self._samples[name])
                for name in self._METRIC_NAMES
            ),
            (
                "stale(side="
                f"{self._stale_counts['side']},top={self._stale_counts['top']})"
            ),
            (
                "ready_to_off(side="
                f"{self._ready_to_off_counts['side']},"
                f"front={self._ready_to_off_counts['front']})"
            ),
        ]
        summary = " ".join(parts)
        self._samples = {name: [] for name in self._METRIC_NAMES}
        self._stale_counts = {"side": 0, "top": 0}
        self._ready_to_off_counts = {"side": 0, "front": 0}
        while self.next_report_at <= now:
            self.next_report_at += self.report_interval_seconds
        return summary


class DualDebugBenchmarkSamples:
    """Collect one bounded debug window without per-frame logging."""

    _HRSEG_FIELDS = (
        "trt_inference_ms",
        "gpu_logit_margin_threshold_ms",
        "candidate_mask_d2h_ms",
        "cpu_8ccl_filter_ms",
        "detector_total_ms",
    )

    def __init__(self) -> None:
        self.hrseg = {name: [] for name in self._HRSEG_FIELDS}
        self.rust_total_ms: list[float] = []
        self.yolo_total_ms: list[float] = []

    def record_hrseg(self, timing: HrSegNetStageTiming) -> None:
        if not isinstance(timing, HrSegNetStageTiming):
            raise ValueError("HrSegNet debug timing sample is unavailable.")
        for name in self._HRSEG_FIELDS:
            value = float(getattr(timing, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid HrSegNet debug timing value for {name}.")
            self.hrseg[name].append(value)

    @staticmethod
    def _record_total(samples: list[float], elapsed_ms: float, label: str) -> None:
        elapsed_ms = float(elapsed_ms)
        if not np.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise ValueError(f"Invalid {label} debug timing value.")
        samples.append(elapsed_ms)

    def record_rust(self, elapsed_ms: float) -> None:
        self._record_total(self.rust_total_ms, elapsed_ms, "Rust")

    def record_yolo(self, elapsed_ms: float) -> None:
        self._record_total(self.yolo_total_ms, elapsed_ms, "YOLO")

    @staticmethod
    def _format(name: str, samples: list[float]) -> str:
        if not samples:
            return f"{name}(mean=nan,n=0)"
        values = np.asarray(samples, dtype=np.float64)
        return f"{name}(mean={values.mean():.3f},n={len(samples)})"

    def summary(self, elapsed_seconds: float) -> str:
        parts = [
            "[DUAL_DEBUG_BENCHMARK]",
            f"window_s={DUAL_DEBUG_BENCHMARK_SECONDS:.3f}",
            f"elapsed_s={float(elapsed_seconds):.3f}",
        ]
        parts.extend(
            self._format(name, self.hrseg[name]) for name in self._HRSEG_FIELDS
        )
        parts.append(self._format("rust_total_ms", self.rust_total_ms))
        parts.append(self._format("yolo_total_ms", self.yolo_total_ms))
        return " | ".join(parts)


class RustDebugBenchmarkSamples:
    """Collect detailed Rust detector timings for one bounded window."""

    _FIELDS = (
        "preprocess_cpu_ms",
        "input_u8_h2d_ms",
        "gpu_preprocess_ms",
        "trt_inference_ms",
        "gpu_argmax_ms",
        "class_map_d2h_ms",
        "cpu_postprocess_ms",
        "total_ms",
    )

    def __init__(self) -> None:
        self.samples = {name: [] for name in self._FIELDS}

    def record(self, timing: RustStageTiming) -> None:
        if not isinstance(timing, RustStageTiming):
            raise ValueError("Rust debug timing sample is unavailable.")
        for name in self._FIELDS:
            value = float(getattr(timing, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid Rust debug timing value for {name}.")
            self.samples[name].append(value)

    @staticmethod
    def _format(name: str, samples: list[float]) -> str:
        if not samples:
            return f"rust_{name}(mean=nan,n=0)"
        values = np.asarray(samples, dtype=np.float64)
        return f"rust_{name}(mean={values.mean():.3f},n={len(samples)})"

    def summary(self, elapsed_seconds: float) -> str:
        parts = [
            "[RUST_DEBUG_BENCHMARK]",
            f"window_s={DUAL_DEBUG_BENCHMARK_SECONDS:.3f}",
            f"elapsed_s={float(elapsed_seconds):.3f}",
        ]
        parts.extend(self._format(name, self.samples[name]) for name in self._FIELDS)
        return " | ".join(parts)


class YoloDebugBenchmarkSamples:
    """Collect detailed obstacle detector timings for one three-second window."""

    _FIELDS = (
        "preprocess_cpu_ms",
        "input_u8_h2d_ms",
        "gpu_preprocess_ms",
        "trt_inference_ms",
        "gpu_postprocess_ms",
        "compact_d2h_ms",
        "cpu_result_ms",
        "total_ms",
    )

    def __init__(self) -> None:
        self.samples = {name: [] for name in self._FIELDS}

    def record(self, timing: ObstacleStageTiming) -> None:
        if not isinstance(timing, ObstacleStageTiming):
            raise ValueError("YOLO debug timing sample is unavailable.")
        for name in self._FIELDS:
            value = float(getattr(timing, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid YOLO debug timing value for {name}.")
            self.samples[name].append(value)

    @staticmethod
    def _format(name: str, samples: list[float]) -> str:
        if not samples:
            return f"yolo_{name}(mean=nan,n=0)"
        values = np.asarray(samples, dtype=np.float64)
        return f"yolo_{name}(mean={values.mean():.3f},n={len(samples)})"

    def summary(self, elapsed_seconds: float) -> str:
        parts = [
            "[YOLO_DEBUG_BENCHMARK]",
            f"window_s={DUAL_DEBUG_BENCHMARK_SECONDS:.3f}",
            f"elapsed_s={float(elapsed_seconds):.3f}",
        ]
        parts.extend(self._format(name, self.samples[name]) for name in self._FIELDS)
        return " | ".join(parts)


def read_realtime_camera_frame(
    camera: LatestFrameCamera,
    *,
    after_sequence: int,
    inference: AlternatingRealtimeInference,
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
) -> tuple[CameraFrame, bool]:
    """Wait for a new frame, stopping outputs exactly when cached results expire."""

    started_at = time.monotonic()
    camera_deadline = started_at + REALTIME_FRAME_TIMEOUT_SECONDS
    cache_ttl = inference.remaining_fresh_seconds(now=started_at)
    cache_expired = cache_ttl == 0.0
    if cache_expired:
        force_cleaning_safe_off(cleaner_controller, pump_controller)

    first_wait = REALTIME_FRAME_TIMEOUT_SECONDS
    if cache_ttl is not None:
        first_wait = min(first_wait, cache_ttl)
    if first_wait > 0.0:
        try:
            return (
                camera.read_latest(
                    after_sequence=after_sequence,
                    timeout=first_wait,
                ),
                cache_expired,
            )
        except TimeoutError:
            now = time.monotonic()
            remaining_cache_ttl = inference.remaining_fresh_seconds(now=now)
            if remaining_cache_ttl is None or remaining_cache_ttl > 1e-9:
                raise
            force_cleaning_safe_off(cleaner_controller, pump_controller)
            cache_expired = True

    remaining_camera_wait = camera_deadline - time.monotonic()
    if remaining_camera_wait <= 0.0:
        raise TimeoutError(
            "No fresh realtime camera frame arrived within "
            f"{REALTIME_FRAME_TIMEOUT_SECONDS:.3f} seconds."
        )
    return (
        camera.read_latest(
            after_sequence=after_sequence,
            timeout=remaining_camera_wait,
        ),
        cache_expired,
    )


def read_dual_realtime_camera_frames(
    side_camera: LatestFrameCamera,
    top_camera: LatestFrameCamera,
    *,
    after_side_sequence: int,
    after_top_sequence: int,
    side_inference: AlternatingRealtimeInference,
    top_inference: AlternatingTopRealtimeInference,
    side_cleaner_controller: CleanerController,
    side_pump_controller: PumpController,
    front_cleaner_controller: CleanerController,
    front_pump_controller: PumpController,
    actuator_arbiter: RealtimeActuatorArbiter | None = None,
) -> DualRealtimeFrameRead:
    """Poll both cameras under one deadline without one role blocking the other."""

    started_at = time.monotonic()
    camera_deadline = started_at + REALTIME_FRAME_TIMEOUT_SECONDS
    roles = {
        "side": {
            "actuator_role": "side",
            "camera": side_camera,
            "after_sequence": after_side_sequence,
            "inference": side_inference,
            "cleaner": side_cleaner_controller,
            "pump": side_pump_controller,
            "frame": None,
            "error": None,
            "expired": False,
        },
        "top": {
            "actuator_role": "front",
            "camera": top_camera,
            "after_sequence": after_top_sequence,
            "inference": top_inference,
            "cleaner": front_cleaner_controller,
            "pump": front_pump_controller,
            "frame": None,
            "error": None,
            "expired": False,
        },
    }
    for role_name, role in roles.items():
        cache_ttl = role["inference"].remaining_fresh_seconds(now=started_at)
        role["cache_deadline"] = (
            None if cache_ttl is None else started_at + cache_ttl
        )
        if cache_ttl == 0.0:
            role["expired"] = True

    while time.monotonic() < camera_deadline:
        ordered_roles = sorted(
            roles.items(),
            key=lambda item: (
                item[1]["cache_deadline"] is None,
                item[1]["cache_deadline"] or 0.0,
            ),
        )
        for role_name, role in ordered_roles:
            if role["frame"] is not None or role["error"] is not None:
                continue
            now = time.monotonic()
            remaining = camera_deadline - now
            if remaining <= 0.0:
                break
            cache_deadline = role["cache_deadline"]
            if not role["expired"] and cache_deadline is not None:
                remaining = min(remaining, max(0.0, cache_deadline - now))
            timeout = min(DUAL_CAMERA_READ_POLL_SECONDS, remaining)
            if timeout <= 0.0:
                role["expired"] = True
                continue
            try:
                role["frame"] = role["camera"].read_latest(
                    after_sequence=role["after_sequence"],
                    timeout=timeout,
                )
            except TimeoutError:
                now = time.monotonic()
                if (
                    not role["expired"]
                    and cache_deadline is not None
                    and now >= cache_deadline - 1e-9
                ):
                    role["expired"] = True
            except RuntimeError as exc:
                clear_realtime_actuator_role(
                    role["cleaner"],
                    role["pump"],
                    role=role["actuator_role"],
                    arbiter=actuator_arbiter,
                )
                role["expired"] = True
                role["error"] = f"{role_name}: {exc}"
        if any(role["frame"] is not None for role in roles.values()):
            break
        if all(role["error"] is not None for role in roles.values()):
            break

    deadline_reached = time.monotonic() >= camera_deadline
    for role_name, role in roles.items():
        if (
            deadline_reached
            and role["frame"] is None
            and role["error"] is None
        ):
            role["error"] = (
                f"{role_name}: no fresh frame within the shared "
                f"{REALTIME_FRAME_TIMEOUT_SECONDS:.3f}s deadline"
            )

    side_frame = roles["side"]["frame"]
    top_frame = roles["top"]["frame"]
    side_cache_expired = bool(roles["side"]["expired"])
    top_cache_expired = bool(roles["top"]["expired"])
    side_error = roles["side"]["error"]
    top_error = roles["top"]["error"]
    side_waiting = (
        side_frame is None
        and side_error is None
        and not side_cache_expired
    )
    top_waiting = (
        top_frame is None
        and top_error is None
        and not top_cache_expired
    )
    if side_frame is None and top_frame is None:
        raise RuntimeError(
            "Both realtime camera streams failed: "
            f"side={side_error}; top={top_error}"
        )
    return DualRealtimeFrameRead(
        side_frame=side_frame,
        top_frame=top_frame,
        side_cache_expired=side_cache_expired,
        top_cache_expired=top_cache_expired,
        side_error=side_error,
        top_error=top_error,
        side_waiting=side_waiting,
        top_waiting=top_waiting,
    )


def validate_engine(
    engine_path: Path,
    expected_sha256: str | None = None,
    sha_option: str = "--engine-sha256",
) -> tuple[Path, str]:
    engine_path = engine_path.expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with engine_path.open("rb") as engine_file:
            for chunk in iter(lambda: engine_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Could not read TensorRT engine: {engine_path}") from exc
    if engine_path.stat().st_size == 0:
        raise ValueError(f"TensorRT engine is empty: {engine_path}")
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256).strip().lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(
                f"{sha_option} must be a 64-character hexadecimal digest."
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"TensorRT engine SHA-256 for {sha_option} does not match the "
                "approved digest: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    return engine_path, actual_sha256


def validate_tf32_runtime_environment() -> None:
    """Require the process-wide math setting used to build the crack plans."""

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "NVIDIA_TF32_OVERRIDE must be exactly 0 before starting Python; "
            "the setting applies to all four TensorRT plans in this process."
        )


def configured_student_engine(
    args: argparse.Namespace,
) -> tuple[Path | None, str | None, bool]:
    if getattr(args, "optimized_student_engine", None) is not None:
        return (
            args.optimized_student_engine,
            args.optimized_student_engine_sha256,
            True,
        )
    return args.student_engine, args.student_engine_sha256, False


def prepare_realtime_transition(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    analysis_worker: CaptureAnalysisWorker,
    *,
    additional_pairs: tuple[
        tuple[CleanerController, PumpController], ...
    ] = (),
) -> None:
    """Keep cleaning off and require a bounded capture-analysis drain."""

    first_error = None
    for cleaner, pump in (
        (cleaner_controller, pump_controller),
        *additional_pairs,
    ):
        for controller in (cleaner, pump):
            try:
                controller.force_off()
            except RuntimeError as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error
    if not analysis_worker.wait_until_idle(
        CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS
    ):
        raise RuntimeError(
            "Capture analysis did not become idle within "
            f"{CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS:.1f} seconds; "
            "cleaning remains OFF and the mission is stopping."
        )
    worker_failure = analysis_worker.failure_message()
    if worker_failure is not None:
        raise RuntimeError(worker_failure)


def close_runtime_resources(
    *,
    display=None,
    uart=None,
    student_detector=None,
    teacher_detector=None,
    capture_crack_detector=None,
    realtime_crack_detector=None,
    obstacle_detector=None,
    camera=None,
    top_camera=None,
    close_windows: bool = False,
) -> None:
    resources = (
        ("display", None if display is None else display.close),
        ("UART", None if uart is None else uart.close),
        (
            "student detector",
            None if student_detector is None else student_detector.close,
        ),
        (
            "teacher detector",
            None if teacher_detector is None else teacher_detector.close,
        ),
        (
            "capture crack detector",
            None if capture_crack_detector is None else capture_crack_detector.close,
        ),
        (
            "realtime crack detector",
            None if realtime_crack_detector is None else realtime_crack_detector.close,
        ),
        (
            "obstacle detector",
            None if obstacle_detector is None else obstacle_detector.close,
        ),
        ("camera", None if camera is None else camera.release),
        ("top camera", None if top_camera is None else top_camera.release),
        ("OpenCV windows", cv2.destroyAllWindows if close_windows else None),
    )
    for resource_name, close_resource in resources:
        if close_resource is None:
            continue
        try:
            close_resource()
        except Exception as exc:
            print(f"Could not close {resource_name}: {exc}", file=sys.stderr)


def run_yolo_debug_benchmark(args: argparse.Namespace) -> int:
    """Continuously measure the pinned TOP-camera obstacle detector stages."""

    try:
        validate_tf32_runtime_environment()
        obstacle_path, obstacle_sha = validate_engine(
            args.obstacle_engine,
            args.obstacle_engine_sha256,
            "--obstacle-engine-sha256",
        )
    except Exception as exc:
        print(f"YOLO debug benchmark configuration failed: {exc}", file=sys.stderr)
        return 2

    obstacle_detector = None
    top_camera = None
    try:
        obstacle_detector = ObstacleDetector(
            obstacle_path,
            obstacle_sha,
            args.obstacle_confidence_threshold,
            stage_timing=True,
        )
        top_camera = open_latest_frame_camera(args.top_camera_device)
        last_sequence = None

        warmup_started = time.monotonic()
        warmup_deadline = warmup_started + RUST_DEBUG_WARMUP_SECONDS
        while time.monotonic() < warmup_deadline:
            camera_frame = top_camera.read_latest(
                after_sequence=last_sequence,
                timeout=LATEST_FRAME_TIMEOUT_SECONDS,
            )
            last_sequence = camera_frame.sequence
            if camera_frame.frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise RuntimeError(
                    "TOP camera must provide "
                    f"{FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
                )
            obstacle_detector.detect(camera_frame.frame)

        while True:
            samples = YoloDebugBenchmarkSamples()
            measurement_started = time.monotonic()
            deadline = measurement_started + DUAL_DEBUG_BENCHMARK_SECONDS
            while time.monotonic() < deadline:
                camera_frame = top_camera.read_latest(
                    after_sequence=last_sequence,
                    timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                )
                last_sequence = camera_frame.sequence
                if camera_frame.frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                    raise RuntimeError(
                        "TOP camera must provide "
                        f"{FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
                    )
                obstacle_detector.detect(camera_frame.frame)
                samples.record(obstacle_detector.last_stage_timing)
            elapsed = time.monotonic() - measurement_started
            print(samples.summary(elapsed), flush=True)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"YOLO debug benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close_runtime_resources(
            obstacle_detector=obstacle_detector,
            top_camera=top_camera,
        )


def run_dual_debug_benchmark(args: argparse.Namespace) -> int:
    """Measure the two-camera inference pipeline for one actuator-free window."""

    try:
        validate_distinct_camera_devices(
            args.side_camera_device,
            args.top_camera_device,
        )
        validate_tf32_runtime_environment()
        student_path, student_sha, optimized_rust = configured_student_engine(args)
        if student_path is None:
            raise ValueError("Dual debug benchmark requires a realtime rust engine.")
        student_path, student_sha = validate_engine(
            student_path,
            student_sha,
            (
                "--optimized-student-engine-sha256"
                if optimized_rust
                else "--student-engine-sha256"
            ),
        )
        crack_path, crack_sha = validate_engine(
            args.realtime_hrsegnet_crack_engine,
            args.realtime_hrsegnet_crack_engine_sha256,
            "--realtime-hrsegnet-crack-engine-sha256",
        )
        obstacle_path, obstacle_sha = validate_engine(
            args.obstacle_engine,
            args.obstacle_engine_sha256,
            "--obstacle-engine-sha256",
        )
        if len({student_sha, crack_sha, obstacle_sha}) != 3:
            raise ValueError("Dual debug TensorRT engines must be different.")
    except Exception as exc:
        print(f"Dual debug benchmark configuration failed: {exc}", file=sys.stderr)
        return 2

    student_detector = None
    crack_detector = None
    obstacle_detector = None
    side_camera = None
    top_camera = None
    try:
        if optimized_rust:
            student_detector = OptimizedRustDetector(student_path, student_sha)
        else:
            student_detector = RustDetector(
                student_path,
                STUDENT_PROFILE,
                student_sha,
                gpu_argmax=True,
            )
        crack_detector = HrSegNetCrackDetector(
            crack_path,
            crack_sha,
            args.realtime_hrsegnet_crack_probability_threshold,
            args.realtime_hrsegnet_crack_min_component_pixels,
            role="realtime",
            stage_timing=True,
        )
        obstacle_detector = ObstacleDetector(
            obstacle_path,
            obstacle_sha,
            args.obstacle_confidence_threshold,
        )
        side_camera = open_latest_frame_camera(args.side_camera_device)
        top_camera = open_latest_frame_camera(args.top_camera_device)

        side_warmup = side_camera.read_latest().frame
        top_warmup = top_camera.read_latest().frame
        for role, frame in (("Side", side_warmup), ("Top", top_warmup)):
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise RuntimeError(
                    f"{role} camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
                )
        side_rust_roi, side_crack_roi = extract_realtime_control_rois(side_warmup)
        _, top_crack_roi = extract_realtime_control_rois(top_warmup)
        student_detector.detect(side_rust_roi)
        crack_detector.detect(side_crack_roi)
        obstacle_detector.detect(top_warmup)
        crack_detector.detect(top_crack_roi)

        samples = DualDebugBenchmarkSamples()
        measurement_started = time.monotonic()
        deadline = measurement_started + DUAL_DEBUG_BENCHMARK_SECONDS
        while time.monotonic() < deadline:
            side_frame = side_camera.read_latest().frame
            top_frame = top_camera.read_latest().frame
            for role, frame in (("Side", side_frame), ("Top", top_frame)):
                if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                    raise RuntimeError(
                        f"{role} camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
                    )
            side_rust_roi, side_crack_roi = extract_realtime_control_rois(side_frame)
            _, top_crack_roi = extract_realtime_control_rois(top_frame)

            started = time.perf_counter()
            student_detector.detect(side_rust_roi)
            samples.record_rust((time.perf_counter() - started) * 1000.0)
            crack_detector.detect(side_crack_roi)
            samples.record_hrseg(crack_detector.last_stage_timing)

            started = time.perf_counter()
            obstacle_detector.detect(top_frame)
            samples.record_yolo((time.perf_counter() - started) * 1000.0)
            crack_detector.detect(top_crack_roi)
            samples.record_hrseg(crack_detector.last_stage_timing)

        elapsed = time.monotonic() - measurement_started
        print(samples.summary(elapsed))
        return 0
    except Exception as exc:
        print(f"Dual debug benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            obstacle_detector=obstacle_detector,
            camera=side_camera,
            top_camera=top_camera,
        )


def run_dual_rust_debug_benchmark(args: argparse.Namespace) -> int:
    """Measure detailed Rust stages for one actuator-free window."""

    try:
        validate_distinct_camera_devices(
            args.side_camera_device,
            args.top_camera_device,
        )
        validate_tf32_runtime_environment()
        student_path, student_sha, optimized_rust = configured_student_engine(args)
        if student_path is None or optimized_rust:
            raise ValueError(
                "Rust debug benchmark requires the raw realtime rust engine."
            )
        student_path, student_sha = validate_engine(
            student_path,
            student_sha,
            "--student-engine-sha256",
        )
    except Exception as exc:
        print(f"Rust debug benchmark configuration failed: {exc}", file=sys.stderr)
        return 2

    student_detector = None
    side_camera = None
    try:
        student_detector = RustDetector(
            student_path,
            STUDENT_PROFILE,
            student_sha,
            gpu_argmax=True,
            stage_timing=True,
        )
        side_camera = open_latest_frame_camera(args.side_camera_device)

        warmup_started = time.monotonic()
        warmup_deadline = warmup_started + RUST_DEBUG_WARMUP_SECONDS
        while time.monotonic() < warmup_deadline:
            warmup_frame = side_camera.read_latest().frame
            if warmup_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise RuntimeError(
                    f"Side camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
                )
            warmup_rust_roi, _ = extract_realtime_control_rois(warmup_frame)
            student_detector.detect(warmup_rust_roi)

        samples = RustDebugBenchmarkSamples()
        measurement_started = time.monotonic()
        deadline = measurement_started + DUAL_DEBUG_BENCHMARK_SECONDS
        while time.monotonic() < deadline:
            side_frame = side_camera.read_latest().frame
            if side_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise RuntimeError(
                    f"Side camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
                )
            rust_roi, _ = extract_realtime_control_rois(side_frame)
            student_detector.detect(rust_roi)
            samples.record(student_detector.last_stage_timing)

        elapsed = time.monotonic() - measurement_started
        print(samples.summary(elapsed))
        return 0
    except Exception as exc:
        print(f"Rust debug benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close_runtime_resources(
            student_detector=student_detector,
            camera=side_camera,
        )


def run_dual_camera_realtime_test(args: argparse.Namespace) -> int:
    """Run the dual-camera pipeline in simulated or explicit UART test mode."""

    try:
        validate_distinct_camera_devices(
            args.side_camera_device,
            args.top_camera_device,
        )
        validate_tf32_runtime_environment()
        student_path, student_sha, optimized_rust = configured_student_engine(args)
        if student_path is None:
            raise ValueError("Dual-camera test requires a realtime rust engine.")
        student_path, student_sha = validate_engine(
            student_path,
            student_sha,
            (
                "--optimized-student-engine-sha256"
                if optimized_rust
                else "--student-engine-sha256"
            ),
        )
        crack_path, crack_sha = validate_engine(
            args.realtime_hrsegnet_crack_engine,
            args.realtime_hrsegnet_crack_engine_sha256,
            "--realtime-hrsegnet-crack-engine-sha256",
        )
        obstacle_path, obstacle_sha = validate_engine(
            args.obstacle_engine,
            args.obstacle_engine_sha256,
            "--obstacle-engine-sha256",
        )
        if len({student_sha, crack_sha, obstacle_sha}) != 3:
            raise ValueError("Dual-camera realtime TensorRT engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    student_detector = None
    crack_detector = None
    obstacle_detector = None
    side_camera = None
    top_camera = None
    display_controller = None
    uart = None
    actuator_arbiter = None
    actuator_shutdown_failure = None
    uart_safe_to_close = True
    uart_enabled = bool(getattr(args, "realtime_test_uart", False))
    front_cleaner = CleanerController(None, FRONT_ACTUATOR_COMMANDS)
    front_pump = PumpController(None, FRONT_ACTUATOR_COMMANDS)
    side_cleaner = CleanerController(None, SIDE_ACTUATOR_COMMANDS)
    side_pump = PumpController(None, SIDE_ACTUATOR_COMMANDS)
    control_off_logger = DualControlOffTransitionLogger()
    try:
        if optimized_rust:
            student_detector = OptimizedRustDetector(student_path, student_sha)
        else:
            student_detector = RustDetector(
                student_path,
                STUDENT_PROFILE,
                student_sha,
                gpu_argmax=True,
            )
        crack_detector = HrSegNetCrackDetector(
            crack_path,
            crack_sha,
            args.realtime_hrsegnet_crack_probability_threshold,
            args.realtime_hrsegnet_crack_min_component_pixels,
            role="realtime",
        )
        obstacle_detector = ObstacleDetector(
            obstacle_path,
            obstacle_sha,
            args.obstacle_confidence_threshold,
        )
        side_camera = open_latest_frame_camera(args.side_camera_device)
        try:
            top_camera = open_latest_frame_camera(args.top_camera_device)
        except Exception:
            side_camera.release()
            side_camera = None
            raise

        side_warmup = side_camera.read_latest()
        top_warmup = top_camera.read_latest()
        if side_warmup.frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                f"Side camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
            )
        if top_warmup.frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                f"Top camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
            )
        side_rust_roi, side_crack_roi = extract_realtime_control_rois(
            side_warmup.frame
        )
        _, top_crack_roi = extract_realtime_control_rois(
            top_warmup.frame
        )
        student_detector.detect(side_rust_roi)
        crack_detector.detect(side_crack_roi)
        obstacle_detector.detect(top_warmup.frame)
        crack_detector.detect(top_crack_roi)

        if uart_enabled:
            try:
                uart = serial.Serial(
                    args.serial_port,
                    args.baud_rate,
                    timeout=0,
                    write_timeout=UART_WRITE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not open UART {args.serial_port}: {exc}"
                ) from exc
            enter_uart_actuator_test(uart)
            actuator_arbiter = RealtimeActuatorArbiter(uart)
            actuator_arbiter.start()

        side_inference = AlternatingRealtimeInference(
            student_detector,
            crack_detector,
            multitask_enabled=False,
            hybrid_enabled=False,
        )
        top_inference = AlternatingTopRealtimeInference(
            obstacle_detector,
            crack_detector,
        )
        front_control_history = FrontControlHistory()
        side_control_history = RealtimeControlHistory()
        front_command_hold = RealtimeActuatorDecisionHold()
        side_command_hold = RealtimeActuatorDecisionHold()
        side_display_cache = RealtimeDisplayCache()
        top_obstacle_display = None
        top_crack_display = None
        last_side_sequence = side_warmup.sequence
        last_top_sequence = top_warmup.sequence
        side_preview_frame = side_warmup.frame
        top_preview_frame = top_warmup.frame
        last_actuator_state = None
        last_actuator_log_at = None
        headless = bool(getattr(args, "headless", False))
        if not headless:
            display_controller = open_latest_frame_display()

        mode_label = (
            f"UART:{args.serial_port}" if uart_enabled else "SIMULATED:NO-UART"
        )
        print(f"Dual-camera realtime test: {mode_label}")
        print(f"Side camera: {args.side_camera_device}")
        print(f"Top camera: {args.top_camera_device}")
        print(f"Side rust detector: {student_detector.method}")
        print(f"Shared side/top crack detector: {crack_detector.method}")
        print(f"Top obstacle detector: {obstacle_detector.method}")
        print(
            "Obstacle inference: full 1280x720 frame; control ROI y=480:720, "
            f"confidence>={args.obstacle_confidence_threshold:g}."
        )
        timing_logger = (
            DualTimingLogger()
            if bool(getattr(args, "dual_timing", False))
            else None
        )
        if timing_logger is not None:
            print(
                "[DUAL_TIMING] enabled: first "
                f"{DUAL_TIMING_WARMUP_SECONDS:g}s excluded; summaries every "
                f"{DUAL_TIMING_REPORT_INTERVAL_SECONDS:g}s."
            )

        while True:
            if actuator_arbiter is not None:
                actuator_arbiter.raise_if_failed()
            if display_controller is not None and display_controller.check_status():
                break
            loop_started_at = time.perf_counter()
            actuator_state_before_frame_read = None
            try:
                actuator_state_before_frame_read = current_dual_actuator_state(
                    front_cleaner,
                    front_pump,
                    side_cleaner,
                    side_pump,
                    arbiter=actuator_arbiter,
                )
                frame_read = read_dual_realtime_camera_frames(
                    side_camera,
                    top_camera,
                    after_side_sequence=last_side_sequence,
                    after_top_sequence=last_top_sequence,
                    side_inference=side_inference,
                    top_inference=top_inference,
                    side_cleaner_controller=side_cleaner,
                    side_pump_controller=side_pump,
                    front_cleaner_controller=front_cleaner,
                    front_pump_controller=front_pump,
                    actuator_arbiter=actuator_arbiter,
                )
                actuator_state_before_frame_read = None
                side_frame = frame_read.side_frame
                top_frame = frame_read.top_frame
                if side_frame is not None:
                    last_side_sequence = side_frame.sequence
                    side_preview_frame = side_frame.frame
                if top_frame is not None:
                    last_top_sequence = top_frame.sequence
                    top_preview_frame = top_frame.frame
                if frame_read.side_cache_expired:
                    side_inference.expire_cache()
                    side_control_history.prune()
                if frame_read.top_cache_expired:
                    top_inference.expire_cache()
                    front_control_history.prune()
                side_outcome = None
                top_outcome = None
                side_update_seconds = None
                top_update_seconds = None
                side_inputs_fresh = realtime_role_inputs_are_fresh(
                    side_frame,
                    side_inference,
                )
                top_inputs_fresh = realtime_role_inputs_are_fresh(
                    top_frame,
                    top_inference,
                )
                if side_inputs_fresh:
                    update_started_at = time.perf_counter()
                    side_rust_roi, side_crack_roi = extract_realtime_control_rois(
                        side_frame.frame
                    )
                    side_outcome = side_inference.process(
                        side_frame,
                        side_rust_roi,
                        side_crack_roi,
                    )
                    side_update_seconds = time.perf_counter() - update_started_at
                if top_inputs_fresh:
                    update_started_at = time.perf_counter()
                    _, top_crack_roi = extract_realtime_control_rois(
                        top_frame.frame
                    )
                    top_outcome = top_inference.process(
                        top_frame,
                        top_frame.frame,
                        top_crack_roi,
                    )
                    top_update_seconds = time.perf_counter() - update_started_at
                if actuator_arbiter is not None:
                    actuator_arbiter.raise_if_failed()

                hold_snapshot_at = time.monotonic()
                side_hold_snapshot = side_command_hold.diagnostic_snapshot(
                    now=hold_snapshot_at
                )
                front_hold_snapshot = front_command_hold.diagnostic_snapshot(
                    now=hold_snapshot_at
                )
                side_history_decision = None
                front_history_decision = None

                if frame_read.side_error is not None:
                    side_control_history.reset()
                    side_inference.reset()
                    side_display_cache.reset()
                    side_command_hold.clear()
                    clear_realtime_actuator_role(
                        side_cleaner,
                        side_pump,
                        role="side",
                        arbiter=actuator_arbiter,
                    )
                elif frame_read.side_waiting:
                    maintain_waiting_role_control(
                        side_cleaner,
                        side_pump,
                        history=side_control_history,
                        inference=side_inference,
                        role="side",
                        hold=side_command_hold,
                        arbiter=actuator_arbiter,
                    )
                elif not side_inputs_fresh:
                    side_control_history.prune()
                    side_inference.expire_cache()
                    side_display_cache.reset()
                    maintain_waiting_role_control(
                        side_cleaner,
                        side_pump,
                        history=side_control_history,
                        inference=side_inference,
                        role="side",
                        hold=side_command_hold,
                        arbiter=actuator_arbiter,
                    )
                else:
                    side_control_history.update(side_outcome)
                    if side_outcome_requires_immediate_off(
                        side_outcome,
                        history_ready=side_control_history.ready,
                    ):
                        side_command_hold.clear()
                        clear_realtime_actuator_role(
                            side_cleaner,
                            side_pump,
                            role="side",
                            arbiter=actuator_arbiter,
                        )
                    elif side_control_history.ready:
                        side_history_decision = side_control_history.decision()
                        publish_realtime_actuator_decision(
                            side_cleaner,
                            side_pump,
                            role="side",
                            history=side_control_history,
                            decision=side_history_decision,
                            hold=side_command_hold,
                            arbiter=actuator_arbiter,
                        )
                    else:
                        maintain_waiting_role_control(
                            side_cleaner,
                            side_pump,
                            history=side_control_history,
                            inference=side_inference,
                            role="side",
                            hold=side_command_hold,
                            arbiter=actuator_arbiter,
                        )

                if frame_read.top_error is not None:
                    front_control_history.reset()
                    top_inference.reset()
                    top_obstacle_display = None
                    top_crack_display = None
                    front_command_hold.clear()
                    clear_realtime_actuator_role(
                        front_cleaner,
                        front_pump,
                        role="front",
                        arbiter=actuator_arbiter,
                    )
                elif frame_read.top_waiting:
                    maintain_waiting_role_control(
                        front_cleaner,
                        front_pump,
                        history=front_control_history,
                        inference=top_inference,
                        role="front",
                        hold=front_command_hold,
                        arbiter=actuator_arbiter,
                    )
                elif not top_inputs_fresh:
                    front_control_history.prune()
                    top_inference.expire_cache()
                    top_obstacle_display = None
                    top_crack_display = None
                    maintain_waiting_role_control(
                        front_cleaner,
                        front_pump,
                        history=front_control_history,
                        inference=top_inference,
                        role="front",
                        hold=front_command_hold,
                        arbiter=actuator_arbiter,
                    )
                else:
                    front_control_history.update(top_outcome)
                    if front_outcome_requires_immediate_off(top_outcome):
                        front_command_hold.clear()
                        clear_realtime_actuator_role(
                            front_cleaner,
                            front_pump,
                            role="front",
                            arbiter=actuator_arbiter,
                        )
                    elif front_control_history.ready:
                        front_history_decision = front_control_history.decision()
                        publish_realtime_actuator_decision(
                            front_cleaner,
                            front_pump,
                            role="front",
                            history=front_control_history,
                            decision=front_history_decision,
                            hold=front_command_hold,
                            arbiter=actuator_arbiter,
                        )
                    else:
                        maintain_waiting_role_control(
                            front_cleaner,
                            front_pump,
                            history=front_control_history,
                            inference=top_inference,
                            role="front",
                            hold=front_command_hold,
                            arbiter=actuator_arbiter,
                        )

                side_display_cache.update(side_outcome)
                if (
                    top_outcome is not None
                    and top_outcome.display_obstacle_result is not None
                ):
                    top_obstacle_display = top_outcome.display_obstacle_result
                if (
                    top_outcome is not None
                    and top_outcome.display_crack_result is not None
                ):
                    top_crack_display = top_outcome.display_crack_result

                actuator_state = current_dual_actuator_state(
                    front_cleaner,
                    front_pump,
                    side_cleaner,
                    side_pump,
                    arbiter=actuator_arbiter,
                )
                now = time.monotonic()
                front_off_diagnostic = None
                if control_off_logger.is_turning_off("front", actuator_state):
                    front_off_diagnostic = build_control_off_diagnostic(
                        role="front",
                        outcome=top_outcome,
                        history_decision=front_history_decision,
                        history_ready=front_control_history.ready_without_prune,
                        camera_error=frame_read.top_error is not None,
                        camera_error_detail=(
                            None
                            if frame_read.top_error is None
                            else str(frame_read.top_error)
                        ),
                        hold_snapshot=front_hold_snapshot,
                        hold=front_command_hold,
                        frame=top_frame,
                        now=now,
                    )
                side_off_diagnostic = None
                if control_off_logger.is_turning_off("side", actuator_state):
                    side_off_diagnostic = build_control_off_diagnostic(
                        role="side",
                        outcome=side_outcome,
                        history_decision=side_history_decision,
                        history_ready=side_control_history.ready_without_prune,
                        camera_error=frame_read.side_error is not None,
                        camera_error_detail=(
                            None
                            if frame_read.side_error is None
                            else str(frame_read.side_error)
                        ),
                        hold_snapshot=side_hold_snapshot,
                        hold=side_command_hold,
                        frame=side_frame,
                        now=now,
                    )
                for control_off_line in control_off_logger.observe(
                    actuator_state,
                    front=front_off_diagnostic,
                    side=side_off_diagnostic,
                ):
                    print(control_off_line)
                    print(control_off_line, flush=True)
                if (
                    actuator_state != last_actuator_state
                    or last_actuator_log_at is None
                    or now - last_actuator_log_at
                    >= REALTIME_TEST_ACTUATOR_LOG_INTERVAL_SECONDS
                ):
                    front_cleaner_label = (
                        "OFF"
                        if actuator_state[0] == 0.0
                        else f"{actuator_state[0]:g}%"
                    )
                    front_pump_label = "ON" if actuator_state[1] else "OFF"
                    side_cleaner_label = (
                        "OFF"
                        if actuator_state[2] == 0.0
                        else f"{actuator_state[2]:g}%"
                    )
                    side_pump_label = "ON" if actuator_state[3] else "OFF"
                    sent_label = "UART_TX_SENT " if uart_enabled else ""
                    print(
                        f"[dual-camera-test][{mode_label}] {sent_label}"
                        f"FRONT_CLEANER_PWM={front_cleaner_label} "
                        f"FRONT_WATER_PUMP={front_pump_label} "
                        f"SIDE_CLEANER_PWM={side_cleaner_label} "
                        f"SIDE_WATER_PUMP={side_pump_label}"
                        f"SIDE_WATER_PUMP={side_pump_label}",
                        flush=True,
                    )
                    last_actuator_state = actuator_state
                    last_actuator_log_at = now

                display_requested_stop = False
                if display_controller is not None:
                    side_display = annotate_realtime_control_results(
                        side_preview_frame,
                        side_display_cache.rust_result,
                        side_display_cache.crack_result,
                    )
                    side_display = draw_realtime_roi_guide(
                        side_preview_frame,
                        side_display,
                    )
                    top_display = annotate_top_realtime_results(
                        top_preview_frame,
                        top_obstacle_display,
                        top_crack_display,
                    )
                    top_display = draw_realtime_roi_guide(
                        top_preview_frame,
                        top_display,
                        primary_roi_label="OBSTACLE CONTROL",
                    )
                    side_small = cv2.resize(side_display, (640, 360))
                    top_small = cv2.resize(top_display, (640, 360))
                    if not display_controller.submit(
                        np.ascontiguousarray(np.hstack((side_small, top_small)))
                    ):
                        display_requested_stop = True
                if timing_logger is not None:
                    timing_now = time.monotonic()
                    summary = timing_logger.record(
                        loop_seconds=time.perf_counter() - loop_started_at,
                        side_update_seconds=side_update_seconds,
                        top_update_seconds=top_update_seconds,
                        side_outcome=side_outcome,
                        top_outcome=top_outcome,
                        side_frame_age_seconds=(
                            None
                            if side_frame is None
                            else max(
                                0.0,
                                timing_now - side_frame.read_completed_at,
                            )
                        ),
                        top_frame_age_seconds=(
                            None
                            if top_frame is None
                            else max(
                                0.0,
                                timing_now - top_frame.read_completed_at,
                            )
                        ),
                        side_stale=(
                            frame_read.side_cache_expired
                            or (
                                not frame_read.side_waiting
                                and not side_inputs_fresh
                            )
                        ),
                        top_stale=(
                            frame_read.top_cache_expired
                            or (
                                not frame_read.top_waiting
                                and not top_inputs_fresh
                            )
                        ),
                        side_ready=side_control_history.ready,
                        front_ready=front_control_history.ready,
                        now=timing_now,
                    )
                    if summary is not None:
                        print(summary)
                if display_requested_stop:
                    break
            except (RuntimeError, TimeoutError, ValueError) as exc:
                actuator_state_before_off = actuator_state_before_frame_read
                if actuator_state_before_off is None:
                    actuator_state_before_off = current_dual_actuator_state(
                        front_cleaner,
                        front_pump,
                        side_cleaner,
                        side_pump,
                        arbiter=actuator_arbiter,
                    )
                if actuator_arbiter is not None:
                    actuator_arbiter.clear_all()
                else:
                    force_cleaning_safe_off(front_cleaner, front_pump)
                    force_cleaning_safe_off(side_cleaner, side_pump)
                runtime_reason, runtime_detail = classify_runtime_control_off(exc)
                for control_off_line in control_off_logger.observe_forced_off(
                    actuator_state_before_off,
                    reason=runtime_reason,
                    detail=runtime_detail,
                    now=time.monotonic(),
                ):
                    print(control_off_line, file=sys.stderr)
                front_control_history.reset()
                side_control_history.reset()
                front_command_hold.clear()
                side_command_hold.clear()
                print(f"Dual-camera realtime test failed: {exc}", file=sys.stderr)
                return 1
    except Exception as exc:
        actuator_state_before_off = current_dual_actuator_state(
            front_cleaner,
            front_pump,
            side_cleaner,
            side_pump,
            arbiter=actuator_arbiter,
        )
        runtime_reason, runtime_detail = classify_runtime_control_off(exc)
        for control_off_line in control_off_logger.observe_forced_off(
            actuator_state_before_off,
            reason=runtime_reason,
            detail=runtime_detail,
            now=time.monotonic(),
        ):
            print(control_off_line, file=sys.stderr)
        print(f"Dual-camera realtime test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if actuator_arbiter is not None:
            try:
                actuator_arbiter.close()
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                actuator_shutdown_failure = str(exc)
                uart_safe_to_close = not actuator_arbiter.worker_is_alive()
        if uart_safe_to_close:
            for controller in (
                front_cleaner,
                front_pump,
                side_cleaner,
                side_pump,
            ):
                try:
                    controller.force_off()
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
        if uart is not None and uart_safe_to_close:
            try:
                send_uart_test_command(uart, ACTUATOR_TEST_STOP_COMMAND)
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                actuator_shutdown_failure = str(exc)
        close_runtime_resources(
            display=display_controller,
            uart=uart if uart_safe_to_close else None,
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            obstacle_detector=obstacle_detector,
            camera=side_camera,
            top_camera=top_camera,
        )
    if actuator_shutdown_failure is not None:
        return 1
    return 0


def run_realtime_test(args: argparse.Namespace) -> int:
    """Run only the student rust and crack models on live camera frames."""

    if getattr(args, "dual_camera", False):
        return run_dual_camera_realtime_test(args)

    multitask_enabled = getattr(args, "realtime_multitask_engine", None) is not None
    hrsegnet_engine_argument = getattr(
        args, "realtime_hrsegnet_crack_engine", None
    )
    hrsegnet_engine_sha256_argument = getattr(
        args, "realtime_hrsegnet_crack_engine_sha256", None
    )
    hrsegnet_probability_threshold = getattr(
        args,
        "realtime_hrsegnet_crack_probability_threshold",
        HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD,
    )
    hrsegnet_min_component_pixels = getattr(
        args,
        "realtime_hrsegnet_crack_min_component_pixels",
        HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS,
    )
    hrsegnet_enabled = hrsegnet_engine_argument is not None
    (
        configured_student_path,
        configured_student_sha256,
        optimized_rust_enabled,
    ) = configured_student_engine(args)
    hybrid_enabled = multitask_enabled and configured_student_path is not None
    if getattr(args, "no_crack", False):
        print("--realtime-test cannot be combined with --no-crack", file=sys.stderr)
        return 2
    if hrsegnet_enabled and hrsegnet_engine_sha256_argument is None:
        print(
            "--realtime-hrsegnet-crack-engine requires its approved "
            "--realtime-hrsegnet-crack-engine-sha256",
            file=sys.stderr,
        )
        return 2
    if (
        not multitask_enabled
        and not hrsegnet_enabled
        and args.realtime_crack_engine is None
    ):
        print(
            "--realtime-test requires --realtime-crack-engine or "
            "--realtime-multitask-engine or --realtime-hrsegnet-crack-engine",
            file=sys.stderr,
        )
        return 2

    try:
        if getattr(args, "dual_camera", False):
            validate_distinct_camera_devices(
                args.side_camera_device,
                args.top_camera_device,
            )
        validate_tf32_runtime_environment()
        if multitask_enabled:
            multitask_engine, multitask_engine_sha256 = validate_engine(
                args.realtime_multitask_engine,
                args.realtime_multitask_engine_sha256,
                "--realtime-multitask-engine-sha256",
            )
            if hybrid_enabled:
                student_engine, student_engine_sha256 = validate_engine(
                    configured_student_path,
                    configured_student_sha256,
                    (
                        "--optimized-student-engine-sha256"
                        if optimized_rust_enabled
                        else "--student-engine-sha256"
                    ),
                )
                if student_engine_sha256 == multitask_engine_sha256:
                    raise ValueError(
                        "Student rust and multitask TensorRT engines must be different."
                    )
            else:
                student_engine = multitask_engine
                student_engine_sha256 = multitask_engine_sha256
            realtime_crack_engine = multitask_engine
            realtime_crack_engine_sha256 = multitask_engine_sha256
        else:
            if configured_student_path is None:
                raise ValueError("A realtime student rust engine is required.")
            student_engine, student_engine_sha256 = validate_engine(
                configured_student_path,
                configured_student_sha256,
                (
                    "--optimized-student-engine-sha256"
                    if optimized_rust_enabled
                    else "--student-engine-sha256"
                ),
            )
            crack_engine_argument = (
                hrsegnet_engine_argument
                if hrsegnet_enabled
                else args.realtime_crack_engine
            )
            crack_sha256_argument = (
                hrsegnet_engine_sha256_argument
                if hrsegnet_enabled
                else args.realtime_crack_engine_sha256
            )
            crack_sha256_option = (
                "--realtime-hrsegnet-crack-engine-sha256"
                if hrsegnet_enabled
                else "--realtime-crack-engine-sha256"
            )
            realtime_crack_engine, realtime_crack_engine_sha256 = validate_engine(
                crack_engine_argument,
                crack_sha256_argument,
                crack_sha256_option,
            )
            if student_engine_sha256 == realtime_crack_engine_sha256:
                raise ValueError("Student and crack TensorRT engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    if hybrid_enabled:
        student_kind = "optimized student rust" if optimized_rust_enabled else "student rust"
        student_pin = "pinned" if configured_student_sha256 is not None else "unpinned"
        print(f"Realtime test {student_kind} engine ({student_pin}): {student_engine}")
        print(f"Realtime test student rust SHA-256: {student_engine_sha256}")
        print(f"Realtime test multitask crack engine (pinned): {multitask_engine}")
        print(f"Realtime test multitask crack SHA-256: {multitask_engine_sha256}")
    elif multitask_enabled:
        print(f"Realtime test optimized multitask engine (pinned): {student_engine}")
        print(f"Realtime test optimized multitask SHA-256: {student_engine_sha256}")
    else:
        student_pin = "pinned" if configured_student_sha256 is not None else "unpinned"
        crack_pin = (
            "pinned"
            if (
                hrsegnet_enabled
                or args.realtime_crack_engine_sha256 is not None
            )
            else "unpinned"
        )
        print(f"Realtime test student engine ({student_pin}): {student_engine}")
        print(f"Realtime test student SHA-256: {student_engine_sha256}")
        print(f"Realtime test crack engine ({crack_pin}): {realtime_crack_engine}")
        print(f"Realtime test crack SHA-256: {realtime_crack_engine_sha256}")

    student_detector = None
    crack_detector = None
    try:
        if hybrid_enabled:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
            crack_detector = OptimizedMultitaskDetector(
                realtime_crack_engine,
                realtime_crack_engine_sha256,
                args.realtime_crack_threshold,
                args.realtime_crack_min_component_pixels,
            )
        elif multitask_enabled:
            student_detector = OptimizedMultitaskDetector(
                student_engine,
                student_engine_sha256,
                args.realtime_crack_threshold,
                args.realtime_crack_min_component_pixels,
            )
        else:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
            if hrsegnet_enabled:
                crack_detector = HrSegNetCrackDetector(
                    realtime_crack_engine,
                    realtime_crack_engine_sha256,
                    hrsegnet_probability_threshold,
                    hrsegnet_min_component_pixels,
                    role="realtime",
                )
            else:
                crack_detector = CrackDetector(
                    realtime_crack_engine,
                    CRACK_REALTIME_PROFILE,
                    realtime_crack_engine_sha256,
                    args.realtime_crack_threshold,
                    args.realtime_crack_min_component_pixels,
                )
    except Exception as exc:
        print(f"Could not initialize realtime test models: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
        )
        return 2

    camera = None
    try:
        camera = open_latest_frame_camera(args.camera_index)
    except Exception as exc:
        print(f"Could not initialize camera: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            camera=camera,
        )
        return 1
    try:
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        warmup_water_roi, warmup_crack_roi = extract_realtime_control_rois(
            warmup_frame
        )
        detect_realtime_control(
            warmup_water_roi,
            warmup_crack_roi,
            student_detector,
            crack_detector,
            multitask_enabled=multitask_enabled,
            hybrid_enabled=hybrid_enabled,
        )
    except Exception as exc:
        print(f"Realtime test model warm-up failed: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            camera=camera,
        )
        return 1

    frame_height, frame_width = warmup_frame.shape[:2]
    print(f"Realtime student detector: {student_detector.method}")
    realtime_crack_method = (
        student_detector.crack_method
        if multitask_enabled and not hybrid_enabled
        else (
            crack_detector.crack_method if hybrid_enabled else crack_detector.method
        )
    )
    print(f"Realtime crack detector: {realtime_crack_method}")
    if hrsegnet_enabled:
        print(
            "Operator-selected pinned HrSegNet baseline: threshold="
            f"{hrsegnet_probability_threshold:g}, "
            "min_component_pixels="
            f"{hrsegnet_min_component_pixels}; field accuracy is not certified."
        )
    print(
        f"Camera index={args.camera_index}, requested={FRAME_WIDTH}x{FRAME_HEIGHT}"
        f"@{FRAME_RATE}, actual frame={frame_width}x{frame_height}, "
        f"rust ROI={warmup_water_roi.shape[1]}x{warmup_water_roi.shape[0]}, "
        f"crack ROI={warmup_crack_roi.shape[1]}x{warmup_crack_roi.shape[0]}"
    )
    headless = bool(getattr(args, "headless", False))
    if headless:
        headless_source = (
            "automatic: no DISPLAY/WAYLAND_DISPLAY"
            if getattr(args, "headless_auto", False)
            else "explicit --headless"
        )
        print(
            "Display mode: headless ("
            f"{headless_source}); preview and display-only overlays are disabled."
        )
        print(
            "Warm-up: 1 frame excluded. Camera inference and simulated control "
            "remain active; press Ctrl+C to stop."
        )
    else:
        print("Display mode: GUI")
        print("Warm-up: 1 frame excluded. Press Q/Esc to close the camera window.")

    measured_frames = 0
    first_decision_completed_at = None
    last_decision_completed_at = None
    total_control_latency_seconds = 0.0
    total_rust_seconds = 0.0
    total_crack_seconds = 0.0
    simulated_cleaner = CleanerController(None)
    simulated_pump = PumpController(None)
    realtime_inference = AlternatingRealtimeInference(
        student_detector,
        crack_detector,
        multitask_enabled=multitask_enabled,
        hybrid_enabled=hybrid_enabled,
    )
    realtime_control_history = RealtimeControlHistory()
    realtime_display_cache = RealtimeDisplayCache()
    last_camera_sequence = warmup_camera_frame.sequence
    last_simulated_state = None
    last_simulated_log_at = None
    display_controller = None
    if not headless:
        try:
            display_controller = open_latest_frame_display()
        except Exception as exc:
            print(f"Could not initialize display: {exc}", file=sys.stderr)
            close_runtime_resources(
                student_detector=student_detector,
                realtime_crack_detector=crack_detector,
                camera=camera,
            )
            return 1
    try:
        while True:
            if (
                display_controller is not None
                and display_controller.check_status()
            ):
                break
            camera_frame, cache_expired_while_waiting = read_realtime_camera_frame(
                camera,
                after_sequence=last_camera_sequence,
                inference=realtime_inference,
                cleaner_controller=simulated_cleaner,
                pump_controller=simulated_pump,
            )
            if cache_expired_while_waiting:
                realtime_control_history.reset()
            last_camera_sequence = camera_frame.sequence
            frame = camera_frame.frame

            try:
                water_roi, crack_roi = extract_realtime_control_rois(frame)
                if not realtime_camera_frame_is_fresh(camera_frame):
                    force_cleaning_safe_off(simulated_cleaner, simulated_pump)
                    realtime_control_history.reset()
                    outcome = None
                else:
                    outcome = realtime_inference.process(
                        camera_frame,
                        water_roi,
                        crack_roi,
                    )
                    if (
                        display_controller is not None
                        and display_controller.check_status()
                    ):
                        force_cleaning_safe_off(simulated_cleaner, simulated_pump)
                        realtime_control_history.reset()
                        break
                    realtime_control_history.update(outcome)
                    if outcome.ready and realtime_control_history.ready:
                        update_cleaning_actuators(
                            simulated_cleaner,
                            simulated_pump,
                            decision=realtime_control_history.decision(),
                        )
                    else:
                        force_cleaning_safe_off(simulated_cleaner, simulated_pump)
                if display_controller is not None:
                    realtime_display_cache.update(outcome)
                decision_completed_at = time.perf_counter()
                decision_completed_monotonic = time.monotonic()
                simulated_state = (
                    simulated_cleaner.duty_percent,
                    simulated_pump.is_on,
                )
                simulated_log_at = time.monotonic()
                if (
                    simulated_state != last_simulated_state
                    or last_simulated_log_at is None
                    or simulated_log_at - last_simulated_log_at
                    >= REALTIME_TEST_ACTUATOR_LOG_INTERVAL_SECONDS
                ):
                    cleaner_label = (
                        "OFF"
                        if simulated_state[0] == 0.0
                        else f"{simulated_state[0]:g}%"
                    )
                    pump_label = "ON" if simulated_state[1] else "OFF"
                    print(
                        "[realtime-test][SIMULATED:NO-UART] "
                        f"CLEANER_PWM={cleaner_label} WATER_PUMP={pump_label}"
                        f"CLEANER_PWM={cleaner_label} WATER_PUMP={pump_label}",
                        flush=True,
                    )
                    last_simulated_state = simulated_state
                    last_simulated_log_at = simulated_log_at

                if display_controller is not None:
                    annotated = annotate_realtime_control_results(
                        frame,
                        outcome.display_rust_result if outcome is not None else None,
                        outcome.display_crack_result if outcome is not None else None,
                    )
                    display = draw_realtime_roi_guide(frame, annotated)
                    display = draw_realtime_result_summary(
                        display,
                        realtime_display_cache.rust_result,
                        realtime_display_cache.crack_result,
                    )
            except (RuntimeError, ValueError) as exc:
                print(f"Realtime test inference failed: {exc}", file=sys.stderr)
                return 1

            measured_frames += 1
            if first_decision_completed_at is None:
                first_decision_completed_at = decision_completed_at
            last_decision_completed_at = decision_completed_at
            decision_span_seconds = max(
                0.0,
                last_decision_completed_at - first_decision_completed_at,
            )
            total_control_latency_seconds += max(
                0.0,
                decision_completed_monotonic - camera_frame.read_completed_at,
            )
            if outcome is not None:
                total_rust_seconds += outcome.rust_seconds
                total_crack_seconds += outcome.crack_seconds
            if display_controller is not None:
                completed_app_frames, total_app_seconds = (
                    display_controller.completed_statistics()
                )
                statistics_label = format_realtime_test_statistics(
                    measured_frames,
                    decision_span_seconds,
                    total_control_latency_seconds,
                    total_rust_seconds,
                    total_crack_seconds,
                    app_frame_count=completed_app_frames,
                    total_app_seconds=total_app_seconds,
                    multitask=multitask_enabled and not hybrid_enabled,
                )
                display = draw_realtime_test_statistics(display, statistics_label)
                if not display_controller.submit(display):
                    break
    except Exception as exc:
        print(f"Realtime test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if simulated_cleaner.is_on or simulated_pump.is_on:
            simulated_cleaner.force_off()
            simulated_pump.force_off()
            print(
                "[realtime-test][SIMULATED:NO-UART] "
                "CLEANER_PWM=OFF WATER_PUMP=OFF (test shutdown)"
            )
        close_runtime_resources(
            display=display_controller,
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            camera=camera,
        )
    return 0


def run_capture_test(args: argparse.Namespace) -> int:
    """Run only the full-frame capture models when the operator presses S."""

    if getattr(args, "headless", False):
        print(
            "--capture-test requires a graphical display because S/Q/Esc are "
            "interactive. Use normal operation for headless STM-triggered captures.",
            file=sys.stderr,
        )
        return 2

    try:
        if getattr(args, "dual_camera", False):
            validate_distinct_camera_devices(
                args.side_camera_device,
                args.top_camera_device,
            )
        validate_tf32_runtime_environment()
        teacher_engine, teacher_engine_sha256 = validate_engine(
            args.teacher_engine,
            args.teacher_engine_sha256,
            "--teacher-engine-sha256",
        )
        capture_crack_engine, capture_crack_engine_sha256 = validate_engine(
            args.capture_hrsegnet_crack_engine,
            args.capture_hrsegnet_crack_engine_sha256,
            "--capture-hrsegnet-crack-engine-sha256",
        )
        if teacher_engine_sha256 == capture_crack_engine_sha256:
            raise ValueError("Capture rust and crack engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    rust_pin = "pinned" if args.teacher_engine_sha256 is not None else "unpinned"
    crack_pin = (
        "pinned"
        if args.capture_hrsegnet_crack_engine_sha256 is not None
        else "unpinned"
    )
    print(f"Capture test rust engine ({rust_pin}): {teacher_engine}")
    print(f"Capture test rust SHA-256: {teacher_engine_sha256}")
    print(f"Capture test crack engine ({crack_pin}): {capture_crack_engine}")
    print(f"Capture test crack SHA-256: {capture_crack_engine_sha256}")

    teacher_detector = None
    capture_crack_detector = None
    try:
        teacher_detector = RustDetector(
            teacher_engine,
            TEACHER_PROFILE,
            teacher_engine_sha256,
        )
        capture_crack_detector = HrSegNetCrackDetector(
            capture_crack_engine,
            capture_crack_engine_sha256,
            args.capture_hrsegnet_crack_probability_threshold,
            args.capture_hrsegnet_crack_min_component_pixels,
            role="capture",
        )
    except Exception as exc:
        print(f"Could not initialize capture test models: {exc}", file=sys.stderr)
        close_runtime_resources(
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
        )
        return 2

    camera = None
    top_camera = None
    try:
        camera_source = (
            args.side_camera_device
            if getattr(args, "dual_camera", False)
            else (
                args.capture_camera_device
                if getattr(args, "capture_camera_device", None) is not None
                else args.camera_index
            )
        )
        camera = open_latest_frame_camera(camera_source)
        if getattr(args, "dual_camera", False):
            top_camera = open_latest_frame_camera(args.top_camera_device)
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        warmup_top_frame = (
            top_camera.read_latest().frame
            if getattr(args, "dual_camera", False)
            else None
        )
        if warmup_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                "Capture test requires the native 1280x720 BGR camera frame; "
                f"received {warmup_frame.shape}."
            )
        if warmup_top_frame is not None and warmup_top_frame.shape != (
            FRAME_HEIGHT,
            FRAME_WIDTH,
            3,
        ):
            raise RuntimeError(
                "Top capture test camera must provide a native 1280x720 BGR frame."
            )
        teacher_detector.detect(warmup_frame)
        capture_crack_detector.detect(warmup_frame)
        if warmup_top_frame is not None:
            capture_crack_detector.detect(warmup_top_frame)
    except Exception as exc:
        print(f"Capture test model warm-up failed: {exc}", file=sys.stderr)
        close_runtime_resources(
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    print(f"Capture rust detector: {teacher_detector.method}")
    print(f"Capture crack detector: {capture_crack_detector.method}")
    print(
        "Capture HrSegNet baseline: threshold="
        f"{args.capture_hrsegnet_crack_probability_threshold:g}, "
        "min_component_pixels="
        f"{args.capture_hrsegnet_crack_min_component_pixels}; "
        "operator-selected and not field-calibrated."
    )
    if getattr(args, "dual_camera", False):
        print(f"Capture side camera: {args.side_camera_device}")
        print(f"Capture top camera: {args.top_camera_device}")
        print(
            "Full-frame input: 1280x720. S runs side rust+crack and top "
            "crack-only; Q/Esc closes."
        )
    else:
        print("Full-frame input: 1280x720. Press S to infer; Q/Esc to close.")
    result_display = None
    last_camera_sequence = warmup_camera_frame.sequence
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            try:
                camera_frame = camera.read_latest(
                    after_sequence=last_camera_sequence,
                    timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                )
            except (RuntimeError, TimeoutError) as exc:
                print(f"Capture test camera stream failed: {exc}", file=sys.stderr)
                return 1
            last_camera_sequence = camera_frame.sequence
            frame = camera_frame.frame
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                print(
                    "Capture test requires the native 1280x720 BGR camera frame; "
                    f"received {frame.shape}.",
                    file=sys.stderr,
                )
                return 1

            display = draw_roi_guide(
                frame,
                result_display if result_display is not None else None,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            try:
                total_started = time.perf_counter()
                top_capture_frame = None
                if getattr(args, "dual_camera", False):
                    side_capture, top_capture = read_stopped_capture_pair(
                        camera,
                        top_camera,
                    )
                    frame = side_capture.frame
                    top_capture_frame = top_capture.frame
                    last_camera_sequence = side_capture.sequence
                rust_started = total_started
                rust_result = teacher_detector.detect(frame)
                rust_finished = time.perf_counter()
                crack_result = capture_crack_detector.detect(frame)
                crack_finished = time.perf_counter()
                rust_result = prioritize_capture_crack_over_rust(
                    rust_result,
                    crack_result,
                )
                result_display = annotate(frame, rust_result)
                result_display = annotate_cracks(
                    result_display,
                    crack_result,
                    None,
                )
                if top_capture_frame is not None:
                    top_crack_started = time.perf_counter()
                    top_crack_result = capture_crack_detector.detect(
                        top_capture_frame
                    )
                    top_crack_finished = time.perf_counter()
                    top_display = annotate_cracks(
                        top_capture_frame,
                        top_crack_result,
                        None,
                    )
                    result_display = np.ascontiguousarray(
                        np.hstack(
                            (
                                cv2.resize(result_display, (640, 360)),
                                cv2.resize(top_display, (640, 360)),
                            )
                        )
                    )
            except Exception as exc:
                print(f"Capture test inference failed: {exc}", file=sys.stderr)
                return 1

            rust_ms = (rust_finished - rust_started) * 1000.0
            crack_ms = (crack_finished - rust_finished) * 1000.0
            total_ms = (crack_finished - total_started) * 1000.0
            crack_label = "YES" if crack_result.detected else "NO"
            print(
                "[capture-test] "
                f"rust={rust_ms:.1f} ms crack={crack_ms:.1f} ms "
                f"total={total_ms:.1f} ms | "
                f"rust_ratio={rust_result.rust_ratio * 100:.2f}% "
                f"crack_detected={crack_label} "
                f"crack_ratio={crack_result.crack_ratio * 100:.2f}%"
            )
            if top_capture_frame is not None:
                print(
                    "[capture-test][top] crack="
                    f"{(top_crack_finished - top_crack_started) * 1000.0:.1f} ms "
                    f"detected={'YES' if top_crack_result.detected else 'NO'} "
                    f"ratio={top_crack_result.crack_ratio * 100:.2f}%"
                )
    except Exception as exc:
        print(f"Capture test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close_runtime_resources(
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            camera=camera,
            top_camera=top_camera,
            close_windows=True,
        )
    return 0


def run_front_capture_test(args: argparse.Namespace) -> int:
    """Run FRONT/TOP obstacle and full-frame crack models on operator keypress."""

    if getattr(args, "headless", False):
        print(
            "--front-capture-test requires a graphical display because S/Q/Esc "
            "are interactive.",
            file=sys.stderr,
        )
        return 2

    try:
        validate_tf32_runtime_environment()
        obstacle_engine, obstacle_engine_sha256 = validate_engine(
            args.obstacle_engine,
            args.obstacle_engine_sha256,
            "--obstacle-engine-sha256",
        )
        capture_crack_engine, capture_crack_engine_sha256 = validate_engine(
            args.capture_hrsegnet_crack_engine,
            args.capture_hrsegnet_crack_engine_sha256,
            "--capture-hrsegnet-crack-engine-sha256",
        )
        if obstacle_engine_sha256 == capture_crack_engine_sha256:
            raise ValueError("FRONT obstacle and crack engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    obstacle_detector = None
    capture_crack_detector = None
    camera = None
    try:
        obstacle_detector = ObstacleDetector(
            obstacle_engine,
            obstacle_engine_sha256,
            args.obstacle_confidence_threshold,
        )
        capture_crack_detector = HrSegNetCrackDetector(
            capture_crack_engine,
            capture_crack_engine_sha256,
            args.capture_hrsegnet_crack_probability_threshold,
            args.capture_hrsegnet_crack_min_component_pixels,
            role="capture",
        )
        camera = open_latest_frame_camera(args.capture_camera_device)
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        if warmup_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                "FRONT capture test requires the native 1280x720 BGR camera "
                f"frame; received {warmup_frame.shape}."
            )
        obstacle_detector.detect(warmup_frame)
        capture_crack_detector.detect(warmup_frame)
    except Exception as exc:
        print(f"Could not initialize FRONT capture test: {exc}", file=sys.stderr)
        close_runtime_resources(
            capture_crack_detector=capture_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
        )
        return 2

    print(f"FRONT camera: {args.capture_camera_device}")
    print(f"FRONT obstacle detector: {obstacle_detector.method}")
    print(f"FRONT capture crack detector: {capture_crack_detector.method}")
    print(
        "Full-frame input: 1280x720; obstacle control ROI: y=480:720. "
        "Press S to infer both models on one frame; Q/Esc to close."
    )

    result_display = None
    last_camera_sequence = warmup_camera_frame.sequence
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            camera_frame = camera.read_latest(
                after_sequence=last_camera_sequence,
                timeout=LATEST_FRAME_TIMEOUT_SECONDS,
            )
            last_camera_sequence = camera_frame.sequence
            frame = camera_frame.frame
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                raise RuntimeError(
                    "FRONT capture test requires the native 1280x720 BGR "
                    f"camera frame; received {frame.shape}."
                )

            cv2.imshow(
                WINDOW_NAME,
                frame if result_display is None else result_display,
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            total_started = time.perf_counter()
            obstacle_started = total_started
            obstacle_result = obstacle_detector.detect(frame)
            obstacle_finished = time.perf_counter()
            crack_result = capture_crack_detector.detect(frame)
            crack_finished = time.perf_counter()
            filtered_obstacle_result = suppress_front_obstacles_overlapping_cracks(
                obstacle_result,
                crack_result.mask,
            )
            suppressed_obstacle_count = (
                len(obstacle_result.detections)
                - len(filtered_obstacle_result.detections)
            )
            result_display = annotate_front_capture_results(
                frame,
                filtered_obstacle_result,
                crack_result,
            )
            print(
                "[front-capture-test] "
                f"obstacle={(obstacle_finished - obstacle_started) * 1000.0:.1f} ms "
                f"crack={(crack_finished - obstacle_finished) * 1000.0:.1f} ms "
                f"total={(crack_finished - total_started) * 1000.0:.1f} ms | "
                f"obstacle_detected={'YES' if filtered_obstacle_result.detected else 'NO'} "
                f"crack_detected={'YES' if crack_result.detected else 'NO'} "
                f"crack_ratio={crack_result.crack_ratio * 100:.2f}%"
            )
            if suppressed_obstacle_count:
                print(
                    "[front-capture-test] "
                    f"suppressed_obstacles={suppressed_obstacle_count} "
                    "reason=crack_overlap "
                    f"threshold={FRONT_CAPTURE_OBSTACLE_CRACK_OVERLAP_RATIO:.1%}"
                )
    except Exception as exc:
        print(f"FRONT capture test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close_runtime_resources(
            capture_crack_detector=capture_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            close_windows=True,
        )
    return 0


def main() -> int:
    args = parse_args()
    if getattr(args, "yolo_debug_benchmark", False):
        return run_yolo_debug_benchmark(args)
    if getattr(args, "front_capture_test", False):
        return run_front_capture_test(args)
    if getattr(args, "capture_test", False):
        return run_capture_test(args)
    if getattr(args, "dual_rust_debug_benchmark", False):
        return run_dual_rust_debug_benchmark(args)
    if getattr(args, "dual_debug_benchmark", False):
        return run_dual_debug_benchmark(args)
    if getattr(args, "realtime_test", False):
        return run_realtime_test(args)
    firebase_upload_required = firebase_upload_is_required()
    db_test_manual_capture = db_test_manual_capture_is_enabled()
    headless = bool(getattr(args, "headless", False))
    if db_test_manual_capture and headless:
        print(
            "DB test manual capture requires the GUI; remove --headless and press S "
            f"for each of the {2 * AUTOMATIC_RAIL_SECTION_TARGET} captures.",
            file=sys.stderr,
        )
        return 2
    dual_camera_enabled = bool(getattr(args, "dual_camera", False))
    multitask_enabled = getattr(args, "realtime_multitask_engine", None) is not None
    hrsegnet_enabled = (
        getattr(args, "realtime_hrsegnet_crack_engine", None) is not None
    )
    capture_hrsegnet_enabled = (
        getattr(args, "capture_hrsegnet_crack_engine", None) is not None
    )
    (
        configured_student_path,
        configured_student_sha256,
        optimized_rust_enabled,
    ) = configured_student_engine(args)
    hybrid_enabled = multitask_enabled and configured_student_path is not None
    if capture_hrsegnet_enabled or multitask_enabled or hrsegnet_enabled:
        try:
            validate_tf32_runtime_environment()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
    capture_crack_engine_sha256 = None
    realtime_crack_engine_sha256 = None
    multitask_engine_sha256 = None
    hrsegnet_engine_sha256 = None
    obstacle_engine_sha256 = None
    try:
        if dual_camera_enabled:
            validate_distinct_camera_devices(
                args.side_camera_device,
                args.top_camera_device,
            )
        args.teacher_engine, teacher_engine_sha256 = validate_engine(
            args.teacher_engine,
            args.teacher_engine_sha256,
            "--teacher-engine-sha256",
        )
        if multitask_enabled:
            args.realtime_multitask_engine, multitask_engine_sha256 = validate_engine(
                args.realtime_multitask_engine,
                args.realtime_multitask_engine_sha256,
                "--realtime-multitask-engine-sha256",
            )
            if hybrid_enabled:
                student_engine, student_engine_sha256 = validate_engine(
                    configured_student_path,
                    configured_student_sha256,
                    (
                        "--optimized-student-engine-sha256"
                        if optimized_rust_enabled
                        else "--student-engine-sha256"
                    ),
                )
            else:
                student_engine = args.realtime_multitask_engine
                student_engine_sha256 = multitask_engine_sha256
        else:
            if configured_student_path is None:
                raise ValueError("A realtime student rust engine is required.")
            student_engine, student_engine_sha256 = validate_engine(
                configured_student_path,
                configured_student_sha256,
                (
                    "--optimized-student-engine-sha256"
                    if optimized_rust_enabled
                    else "--student-engine-sha256"
                ),
            )
        if teacher_engine_sha256 == student_engine_sha256:
            raise ValueError("Teacher and realtime TensorRT engines must be different.")
        if hrsegnet_enabled:
            (
                args.realtime_hrsegnet_crack_engine,
                hrsegnet_engine_sha256,
            ) = validate_engine(
                args.realtime_hrsegnet_crack_engine,
                args.realtime_hrsegnet_crack_engine_sha256,
                "--realtime-hrsegnet-crack-engine-sha256",
            )
        if capture_hrsegnet_enabled:
            (
                args.capture_hrsegnet_crack_engine,
                capture_crack_engine_sha256,
            ) = validate_engine(
                args.capture_hrsegnet_crack_engine,
                args.capture_hrsegnet_crack_engine_sha256,
                "--capture-hrsegnet-crack-engine-sha256",
            )
            if not multitask_enabled and not hrsegnet_enabled:
                args.realtime_crack_engine, realtime_crack_engine_sha256 = validate_engine(
                    args.realtime_crack_engine,
                    args.realtime_crack_engine_sha256,
                    "--realtime-crack-engine-sha256",
                )
            engine_digests = {
                teacher_engine_sha256,
                student_engine_sha256,
                capture_crack_engine_sha256,
            }
            if hybrid_enabled:
                engine_digests.add(multitask_engine_sha256)
            if hrsegnet_enabled:
                engine_digests.add(hrsegnet_engine_sha256)
            elif not multitask_enabled:
                engine_digests.add(realtime_crack_engine_sha256)
            expected_engine_count = 4 if hybrid_enabled or not multitask_enabled else 3
            if len(engine_digests) != expected_engine_count:
                raise ValueError(
                    "Capture and realtime TensorRT engine files must all be different."
                )
        if dual_camera_enabled:
            args.obstacle_engine, obstacle_engine_sha256 = validate_engine(
                args.obstacle_engine,
                args.obstacle_engine_sha256,
                "--obstacle-engine-sha256",
            )
            if obstacle_engine_sha256 in engine_digests:
                raise ValueError(
                    "Obstacle TensorRT engine must differ from every rust/crack engine."
                )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    missing_engine_digests = []
    if args.teacher_engine_sha256 is None:
        missing_engine_digests.append(
            ("teacher", "--teacher-engine-sha256", teacher_engine_sha256)
        )
    if (not multitask_enabled or hybrid_enabled) and configured_student_sha256 is None:
        missing_engine_digests.append(
            (
                "optimized student" if optimized_rust_enabled else "student",
                (
                    "--optimized-student-engine-sha256"
                    if optimized_rust_enabled
                    else "--student-engine-sha256"
                ),
                student_engine_sha256,
            )
        )
    if (
        capture_hrsegnet_enabled
        and args.capture_hrsegnet_crack_engine_sha256 is None
    ):
        missing_engine_digests.append(
            (
                "capture crack",
                "--capture-hrsegnet-crack-engine-sha256",
                capture_crack_engine_sha256,
            )
        )
    if (
        not multitask_enabled
        and not hrsegnet_enabled
        and
        args.realtime_crack_engine is not None
        and args.realtime_crack_engine_sha256 is None
    ):
        missing_engine_digests.append(
            (
                "realtime crack",
                "--realtime-crack-engine-sha256",
                realtime_crack_engine_sha256,
            )
        )
    if hrsegnet_enabled and args.realtime_hrsegnet_crack_engine_sha256 is None:
        missing_engine_digests.append(
            (
                "realtime HrSegNet crack",
                "--realtime-hrsegnet-crack-engine-sha256",
                hrsegnet_engine_sha256,
            )
        )
    if missing_engine_digests:
        if not args.no_uart:
            missing_options = ", ".join(
                option for _, option, _ in missing_engine_digests
            )
            print(
                "UART operation requires approved digests for every enabled "
                "TensorRT engine. "
                f"Missing: {missing_options}. First verify them with --no-uart.",
                file=sys.stderr,
            )
            for role, _, digest in missing_engine_digests:
                print(f"Current {role} SHA-256: {digest}", file=sys.stderr)
            return 2
        for role, _, digest in missing_engine_digests:
            print(
                f"TensorRT {role} engine is unpinned because UART is disabled; "
                f"current SHA-256: {digest}"
            )

    teacher_detector = None
    student_detector = None
    capture_crack_detector = None
    realtime_crack_detector = None
    obstacle_detector = None
    try:
        teacher_detector = RustDetector(
            args.teacher_engine,
            TEACHER_PROFILE,
            teacher_engine_sha256,
        )
        if hybrid_enabled:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
        elif multitask_enabled:
            student_detector = OptimizedMultitaskDetector(
                args.realtime_multitask_engine,
                multitask_engine_sha256,
                args.realtime_crack_threshold,
                args.realtime_crack_min_component_pixels,
            )
        else:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
        if capture_hrsegnet_enabled:
            capture_crack_detector = HrSegNetCrackDetector(
                args.capture_hrsegnet_crack_engine,
                capture_crack_engine_sha256,
                args.capture_hrsegnet_crack_probability_threshold,
                args.capture_hrsegnet_crack_min_component_pixels,
                role="capture",
            )
            if hybrid_enabled:
                realtime_crack_detector = OptimizedMultitaskDetector(
                    args.realtime_multitask_engine,
                    multitask_engine_sha256,
                    args.realtime_crack_threshold,
                    args.realtime_crack_min_component_pixels,
                )
            elif hrsegnet_enabled:
                realtime_crack_detector = HrSegNetCrackDetector(
                    args.realtime_hrsegnet_crack_engine,
                    hrsegnet_engine_sha256,
                    args.realtime_hrsegnet_crack_probability_threshold,
                    args.realtime_hrsegnet_crack_min_component_pixels,
                    role="realtime",
                )
            elif not multitask_enabled:
                realtime_crack_detector = CrackDetector(
                    args.realtime_crack_engine,
                    CRACK_REALTIME_PROFILE,
                    realtime_crack_engine_sha256,
                    args.realtime_crack_threshold,
                    args.realtime_crack_min_component_pixels,
                )
        if dual_camera_enabled:
            obstacle_detector = ObstacleDetector(
                args.obstacle_engine,
                obstacle_engine_sha256,
                args.obstacle_confidence_threshold,
            )
    except Exception as exc:
        print(exc, file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
        )
        return 2

    camera = None
    top_camera = None
    try:
        camera_source = (
            args.side_camera_device if dual_camera_enabled else args.camera_index
        )
        camera = open_latest_frame_camera(camera_source)
        if dual_camera_enabled:
            top_camera = open_latest_frame_camera(args.top_camera_device)
    except Exception as exc:
        print(f"Could not initialize camera: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    try:
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        warmup_top_camera_frame = (
            top_camera.read_latest() if dual_camera_enabled else None
        )
    except Exception as exc:
        print(f"Could not read a camera frame for model warm-up: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1
    try:
        warmup_water_roi, warmup_crack_roi = extract_realtime_control_rois(
            warmup_frame
        )
        teacher_detector.detect(warmup_frame)
        if capture_crack_detector is not None:
            capture_crack_detector.detect(warmup_frame)
        detect_realtime_control(
            warmup_water_roi,
            warmup_crack_roi,
            student_detector,
            realtime_crack_detector,
            multitask_enabled=multitask_enabled,
            hybrid_enabled=hybrid_enabled,
        )
        if dual_camera_enabled:
            if warmup_top_camera_frame.frame.shape != (
                FRAME_HEIGHT,
                FRAME_WIDTH,
                3,
            ):
                raise RuntimeError(
                    "Top camera did not provide a native 1280x720 BGR frame."
                )
            _, warmup_top_crack_roi = extract_realtime_control_rois(
                warmup_top_camera_frame.frame
            )
            obstacle_detector.detect(warmup_top_camera_frame.frame)
            realtime_crack_detector.detect(warmup_top_crack_roi)
    except Exception as exc:
        print(f"TensorRT model warm-up failed: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    try:
        workbook = InspectionWorkbook(
            args.report,
            capture_model_filename=args.teacher_engine.name,
            capture_model_sha256=teacher_detector.engine_sha256,
            realtime_model_filename=(
                args.realtime_multitask_engine.name
                if multitask_enabled and not hybrid_enabled
                else student_engine.name
            ),
            realtime_model_sha256=student_detector.engine_sha256,
            capture_detector=teacher_detector.method,
            realtime_detector=student_detector.method,
            crack_model_filename=(
                args.capture_hrsegnet_crack_engine.name
                if capture_hrsegnet_enabled
                else None
            ),
            crack_model_sha256=(
                None
                if capture_crack_detector is None
                else capture_crack_engine_sha256
            ),
            crack_detector=(
                None if capture_crack_detector is None else capture_crack_detector.method
            ),
            crack_probability_threshold=(
                args.capture_hrsegnet_crack_probability_threshold
            ),
            capture_crack_min_component_pixels=(
                args.capture_hrsegnet_crack_min_component_pixels
            ),
            realtime_crack_model_filename=(
                args.realtime_multitask_engine.name
                if multitask_enabled
                else (
                    args.realtime_hrsegnet_crack_engine.name
                    if hrsegnet_enabled
                    else (
                        None
                        if args.realtime_crack_engine is None
                        else args.realtime_crack_engine.name
                    )
                )
            ),
            realtime_crack_model_sha256=(
                multitask_engine_sha256
                if multitask_enabled
                else (
                    hrsegnet_engine_sha256
                    if hrsegnet_enabled
                    else (
                        None
                        if realtime_crack_detector is None
                        else realtime_crack_detector.engine_sha256
                    )
                )
            ),
            realtime_crack_detector=(
                (
                    realtime_crack_detector.crack_method
                    if hybrid_enabled
                    else student_detector.crack_method
                )
                if multitask_enabled
                else (
                    None
                    if realtime_crack_detector is None
                    else realtime_crack_detector.method
                )
            ),
            realtime_crack_probability_threshold=(
                args.realtime_hrsegnet_crack_probability_threshold
                if hrsegnet_enabled
                else args.realtime_crack_threshold
            ),
            realtime_crack_min_component_pixels=(
                args.realtime_hrsegnet_crack_min_component_pixels
                if hrsegnet_enabled
                else args.realtime_crack_min_component_pixels
            ),
        )
    except Exception as exc:
        print(f"Could not open inspection report {args.report}: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 2

    uart = None
    realtime_actuator_arbiter = None
    if not args.no_uart:
        try:
            uart = serial.Serial(
                args.serial_port,
                args.baud_rate,
                timeout=0,
                write_timeout=UART_WRITE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            print(f"Could not open UART {args.serial_port}: {exc}", file=sys.stderr)
            close_runtime_resources(
                student_detector=student_detector,
                teacher_detector=teacher_detector,
                capture_crack_detector=capture_crack_detector,
                realtime_crack_detector=realtime_crack_detector,
                obstacle_detector=obstacle_detector,
                camera=camera,
                top_camera=top_camera,
            )
            return 1

    if dual_camera_enabled:
        front_cleaner_controller = CleanerController(
            uart,
            FRONT_ACTUATOR_COMMANDS,
        )
        front_pump_controller = PumpController(uart, FRONT_ACTUATOR_COMMANDS)
        cleaner_controller = CleanerController(uart, SIDE_ACTUATOR_COMMANDS)
        pump_controller = PumpController(uart, SIDE_ACTUATOR_COMMANDS)
        actuator_pairs = (
            (front_cleaner_controller, front_pump_controller),
            (cleaner_controller, pump_controller),
        )
    else:
        front_cleaner_controller = None
        front_pump_controller = None
        cleaner_controller = CleanerController(uart)
        pump_controller = PumpController(uart)
        actuator_pairs = ((cleaner_controller, pump_controller),)
    analysis_worker = None
    display_controller = None
    live_preview_publisher = None

    try:
        analysis_worker = CaptureAnalysisWorker(
            teacher_detector,
            workbook,
            crack_detector=capture_crack_detector,
            obstacle_detector_factory=(
                lambda: ObstacleDetector(args.obstacle_engine,
                                         obstacle_engine_sha256,
                                         args.obstacle_confidence_threshold)
            ) if dual_camera_enabled else None,
        )
        analysis_worker.start()
        print(f"Capture detector: {teacher_detector.method}")
        print(f"Realtime detector: {student_detector.method}")
        if dual_camera_enabled:
            print(f"Side camera: {args.side_camera_device}")
            print(f"Top camera: {args.top_camera_device}")
            print(f"Top obstacle detector: {obstacle_detector.method}")
        if args.no_crack:
            print(
                "WARNING: --no-crack TEST MODE; crack detection and the "
                "fixed y=540:720 crack safety interlock is bypassed."
            )
        elif capture_crack_detector is None:
            print(
                "Crack detectors: disabled; provide both capture and realtime "
                "HrSegNet TensorRT engines."
            )
        else:
            print(f"Capture crack detector: {capture_crack_detector.method}")
            print(
                "Capture HrSegNet baseline: threshold="
                f"{args.capture_hrsegnet_crack_probability_threshold:g}, "
                "min_component_pixels="
                f"{args.capture_hrsegnet_crack_min_component_pixels}; "
                "operator-selected and not field-calibrated."
            )
            print(
                "Realtime crack detector: "
                f"{(realtime_crack_detector.crack_method if hybrid_enabled else student_detector.crack_method) if multitask_enabled else realtime_crack_detector.method}"
            )
            if hrsegnet_enabled:
                print(
                    "Realtime HrSegNet baseline: threshold="
                    f"{args.realtime_hrsegnet_crack_probability_threshold:g}, "
                    "min_component_pixels="
                    f"{args.realtime_hrsegnet_crack_min_component_pixels}."
                )
        print(
            "Realtime control inputs: "
            f"rust={warmup_water_roi.shape[1]}x{warmup_water_roi.shape[0]} "
            f"(full width, y={REALTIME_RUST_ROI_TOP}:{REALTIME_RUST_ROI_BOTTOM}), "
            f"crack={warmup_crack_roi.shape[1]}x{warmup_crack_roi.shape[0]} "
            f"(full width, y={REALTIME_CRACK_ROI_TOP}:{REALTIME_CRACK_ROI_BOTTOM})"
        )
        print(f"Inspection report: {workbook.path}")
        if uart is None:
            print("UART trigger: disabled (--no-uart); terminal start will not be sent.")
        else:
            print(f"UART trigger: {args.serial_port} at {args.baud_rate} baud")
            print("Type start and press Enter to send START to the STM.")
        if headless:
            headless_source = (
                "automatic: no DISPLAY/WAYLAND_DISPLAY"
                if getattr(args, "headless_auto", False)
                else "explicit --headless"
            )
            print(
                "Display mode: headless ("
                f"{headless_source}); preview, Q/Esc, and manual S capture are "
                "disabled. STM CAMERA_CAPTURE, inference, UART control, capture "
                "analysis, and reports remain active."
            )
        else:
            print("Display mode: GUI")
            if db_test_manual_capture:
                print(
                    f"DB test manual capture: press S {2 * AUTOMATIC_RAIL_SECTION_TARGET} times "
                    f"(1-{AUTOMATIC_RAIL_SECTION_TARGET} before cleaning, "
                    f"{AUTOMATIC_RAIL_SECTION_TARGET + 1}-{2 * AUTOMATIC_RAIL_SECTION_TARGET} after cleaning). "
                    "The run uploads to Firebase and exits automatically after the last capture."
                )
            else:
                print(
                    "Press S to save a capture in CAPTURE_SCAN mode, "
                    "or Q/Esc to close the camera window."
                )
        uart_buffer = bytearray()
        captured_until = 0.0
        captured_display = None
        mode = CAPTURE_SCAN_MODE
        initial_rail_section_count = 0
        rescan_rail_section_count = 0
        realtime_control_ready = False
        front_control_ready = False
        side_control_ready = False
        realtime_inference = AlternatingRealtimeInference(
            student_detector,
            realtime_crack_detector,
            multitask_enabled=multitask_enabled,
            hybrid_enabled=hybrid_enabled,
            no_crack=args.no_crack,
        )
        realtime_control_history = RealtimeControlHistory(no_crack=args.no_crack)
        top_realtime_inference = (
            AlternatingTopRealtimeInference(
                obstacle_detector,
                realtime_crack_detector,
            )
            if dual_camera_enabled
            else None
        )
        front_control_history = (
            FrontControlHistory() if dual_camera_enabled else None
        )
        front_command_hold = (
            RealtimeActuatorDecisionHold() if dual_camera_enabled else None
        )
        side_command_hold = (
            RealtimeActuatorDecisionHold() if dual_camera_enabled else None
        )
        realtime_display_cache = RealtimeDisplayCache()
        top_obstacle_display = None
        top_crack_display = None
        last_camera_sequence = warmup_camera_frame.sequence
        last_top_camera_sequence = (
            warmup_top_camera_frame.sequence if dual_camera_enabled else None
        )
        start_command_sent = False
        last_link_ok_sent = None
        start_acknowledged = False
        terminal_input_active = True
        if not headless:
            display_controller = open_latest_frame_display()
            try:
                live_preview_publisher = FirebaseLivePreviewPublisher.from_environment()
            except FirebaseUploadError as exc:
                print(f"Firebase live preview disabled: {exc}", file=sys.stderr)
            if live_preview_publisher is not None:
                print("Firebase live preview: GUI snapshot upload every 5 seconds.")
        print(f"Mode: {mode}")
    except Exception as exc:
        print(f"Could not initialize runtime state: {exc}", file=sys.stderr)
        for cleaner, pump in actuator_pairs:
            for controller in (cleaner, pump):
                try:
                    controller.force_off()
                except RuntimeError as control_exc:
                    print(control_exc, file=sys.stderr)
        if analysis_worker is not None:
            worker_failure = analysis_worker.shutdown()
            if worker_failure is not None:
                print(worker_failure, file=sys.stderr)
        close_runtime_resources(
            display=display_controller,
            uart=uart,
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    analysis_shutdown_failure = None
    realtime_actuator_shutdown_failure = None
    uart_safe_to_close = True
    worker_failure_reported = None
    mission_completed = False
    dashboard_finalized = False
    dashboard_manifest_path = None
    db_test_publish_allowed = not db_test_manual_capture
    try:
        while True:
            if realtime_actuator_arbiter is not None:
                try:
                    realtime_actuator_arbiter.raise_if_failed()
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
                    return 1
            try:
                if (
                    display_controller is not None
                    and display_controller.check_status()
                ):
                    break
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1
            worker_failure = analysis_worker.failure_message()
            if worker_failure is not None:
                print(worker_failure, file=sys.stderr)
                worker_failure_reported = worker_failure
                return 1

            try:
                cache_expired_while_waiting = False
                side_frame_available = True
                top_frame_available = not dual_camera_enabled
                if mode == REALTIME_MODE:
                    if dual_camera_enabled:
                        frame_read = read_dual_realtime_camera_frames(
                            camera,
                            top_camera,
                            after_side_sequence=last_camera_sequence,
                            after_top_sequence=last_top_camera_sequence,
                            side_inference=realtime_inference,
                            top_inference=top_realtime_inference,
                            side_cleaner_controller=cleaner_controller,
                            side_pump_controller=pump_controller,
                            front_cleaner_controller=front_cleaner_controller,
                            front_pump_controller=front_pump_controller,
                            actuator_arbiter=realtime_actuator_arbiter,
                        )
                        side_frame_available = frame_read.side_frame is not None
                        top_frame_available = frame_read.top_frame is not None
                        if side_frame_available:
                            camera_frame = frame_read.side_frame
                        if top_frame_available:
                            top_camera_frame = frame_read.top_frame
                        if frame_read.side_error is not None:
                            print(
                                f"Side camera realtime frame unavailable: "
                                f"{frame_read.side_error}",
                                file=sys.stderr,
                            )
                        if frame_read.top_error is not None:
                            print(
                                f"Top camera realtime frame unavailable: "
                                f"{frame_read.top_error}",
                                file=sys.stderr,
                            )
                    else:
                        (
                            camera_frame,
                            cache_expired_while_waiting,
                        ) = read_realtime_camera_frame(
                            camera,
                            after_sequence=last_camera_sequence,
                            inference=realtime_inference,
                            cleaner_controller=cleaner_controller,
                            pump_controller=pump_controller,
                        )
                else:
                    camera_frame = camera.read_latest(
                        after_sequence=last_camera_sequence,
                        timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                    )
                    if dual_camera_enabled:
                        top_camera_frame = top_camera.read_latest(
                            after_sequence=last_top_camera_sequence,
                            timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                        )
                        top_frame_available = True
            except (RuntimeError, TimeoutError) as exc:
                if mode == REALTIME_MODE:
                    if realtime_actuator_arbiter is not None:
                        realtime_actuator_arbiter.clear_all()
                    else:
                        force_cleaning_pairs_safe_off(*actuator_pairs)
                print(f"Camera frame stream failed: {exc}", file=sys.stderr)
                return 1
            if dual_camera_enabled and mode == REALTIME_MODE:
                if frame_read.side_cache_expired:
                    realtime_inference.expire_cache()
                    realtime_control_history.prune()
                if frame_read.top_cache_expired:
                    top_realtime_inference.expire_cache()
                    front_control_history.prune()
            elif cache_expired_while_waiting:
                realtime_control_history.reset()
                realtime_control_ready = False
            if side_frame_available:
                last_camera_sequence = camera_frame.sequence
            if dual_camera_enabled:
                if top_frame_available:
                    last_top_camera_sequence = top_camera_frame.sequence
            frame = camera_frame.frame

            terminal_command = None
            if terminal_input_active:
                terminal_command, terminal_input_active = read_terminal_command()
            if terminal_command is not None:
                if terminal_command != "start":
                    print("Unknown terminal command ignored; type start.")
                elif uart is None:
                    print("START not sent because UART is disabled (--no-uart).")
                elif start_command_sent:
                    print("START ignored because it was already sent to the STM.")
                elif mode != CAPTURE_SCAN_MODE or initial_rail_section_count != 0:
                    print(
                        f"START ignored because inspection is already in progress "
                        f"(mode={mode}, rail_sections={initial_rail_section_count}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET})."
                    )
                else:
                    try:
                        written = uart.write(START_COMMAND)
                    except (serial.SerialException, OSError) as exc:
                        print(f"Could not send START to the STM: {exc}", file=sys.stderr)
                        return 1
                    if written != len(START_COMMAND):
                        print(
                            f"Could not send the complete START command "
                            f"({written}/{len(START_COMMAND)} bytes).",
                            file=sys.stderr,
                        )
                        return 1
                    start_command_sent = True
                    print("UART TX: START")

            uart_messages = []
            captured_this_iteration = False
            realtime_started_this_iteration = False
            if uart is not None and start_command_sent:
                # STM legacy 통신 감시(주행 중 1 s)용 keepalive. 이 루프가 멈추면 STM도 멈춘다.
                link_now = time.monotonic()
                if (last_link_ok_sent is None
                        or link_now - last_link_ok_sent >= LEGACY_LINK_INTERVAL_SECONDS):
                    try:
                        written = uart.write(LINK_OK_COMMAND)
                    except (serial.SerialException, OSError) as exc:
                        print(f"Could not send LINK_OK to the STM: {exc}", file=sys.stderr)
                        return 1
                    if written != len(LINK_OK_COMMAND):
                        print("Could not send the complete LINK_OK keepalive.", file=sys.stderr)
                        return 1
                    last_link_ok_sent = link_now
            if uart is not None:
                try:
                    uart_messages = read_uart_messages(uart, uart_buffer)
                except (serial.SerialException, OSError) as exc:
                    print(f"UART connection failed: {exc}", file=sys.stderr)
                    return 1

            if (
                REALTIME_START_TRIGGER in uart_messages
                and RESCAN_RETURN_START_TRIGGER in uart_messages
            ):
                print(
                    f"Timing/protocol error: STM {REALTIME_START_TRIGGER} and "
                    f"{RESCAN_RETURN_START_TRIGGER} arrived in the same UART read "
                    "batch; no "
                    "realtime frame could be inspected.",
                    file=sys.stderr,
                )
                return 1

            if sum(message == CAPTURE_TRIGGER for message in uart_messages) > 1:
                print(
                    f"Multiple {CAPTURE_TRIGGER} messages arrived before one "
                    "frame could be saved. Stopping to avoid duplicate captures.",
                    file=sys.stderr,
                )
                return 1

            for message in uart_messages:
                print(f"UART: {message}")
                if message in MOTION_TRIGGERS and not start_acknowledged:
                    print(
                        f"Protocol error: STM {message} arrived before STARTED "
                        "acknowledgement; possible old firmware or automatic start.",
                        file=sys.stderr,
                    )
                    return 1
                if message == STARTED_TRIGGER:
                    if not start_command_sent:
                        print(
                            f"Unsolicited STM {STARTED_TRIGGER} received; inspection "
                            "state was not changed.",
                            file=sys.stderr,
                        )
                    elif start_acknowledged:
                        print(f"Duplicate STM {STARTED_TRIGGER} ignored.")
                    else:
                        start_acknowledged = True
                        print(
                            "STM STARTED acknowledged; waiting for encoder "
                            "Rail Sections."
                        )
                elif message == CAPTURE_TRIGGER:
                    if mode == CAPTURE_SCAN_MODE:
                        phase = INITIAL_PHASE
                        phase_sequence = initial_rail_section_count + 1
                        phase_rail_section_count = initial_rail_section_count
                        phase_label = "Automatic"
                    elif mode == RESCAN_MODE:
                        phase = RESCAN_PHASE
                        phase_sequence = rescan_rail_section_count + 1
                        phase_rail_section_count = rescan_rail_section_count
                        phase_label = "Rescan"
                    else:
                        print(
                            f"Protocol error: STM {CAPTURE_TRIGGER} arrived in "
                            f"{mode} mode; no capture was acknowledged.",
                            file=sys.stderr,
                        )
                        return 1
                    if phase_rail_section_count >= AUTOMATIC_RAIL_SECTION_TARGET:
                        print(
                            f"Protocol error: extra STM {CAPTURE_TRIGGER} after "
                            f"{AUTOMATIC_RAIL_SECTION_TARGET} {phase} Rail Sections.",
                            file=sys.stderr,
                        )
                        return 1
                    try:
                        top_capture_camera_frame = None
                        if dual_camera_enabled:
                            (
                                capture_camera_frame,
                                top_capture_camera_frame,
                            ) = read_stopped_capture_pair(camera, top_camera)
                            last_top_camera_sequence = (
                                top_capture_camera_frame.sequence
                            )
                        else:
                            capture_camera_frame = read_stopped_capture_frame(
                                camera,
                                after_sequence=last_camera_sequence,
                            )
                        last_camera_sequence = capture_camera_frame.sequence
                        capture_frame = capture_camera_frame.frame
                        shared_captured_at = datetime.now()
                        if dual_camera_enabled:
                            queue_capture_for_analysis(
                                capture_frame,
                                phase,
                                phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                                camera_role=SIDE_CAMERA_ROLE,
                                captured_at=shared_captured_at,
                                frame_read_completed_at=(
                                    capture_camera_frame.read_completed_at
                                ),
                            )
                            queue_capture_for_analysis(
                                top_capture_camera_frame.frame,
                                phase,
                                phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                                camera_role=TOP_CAMERA_ROLE,
                                captured_at=shared_captured_at,
                                frame_read_completed_at=(
                                    top_capture_camera_frame.read_completed_at
                                ),
                            )
                        else:
                            # Preserve the legacy single-camera call contract.
                            # Several downstream integrations wrap this function
                            # positionally and do not know the dual-camera metadata.
                            queue_capture_for_analysis(
                                capture_frame,
                                phase,
                                phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                            )
                        send_capture_ok(uart)
                    except (OSError, RuntimeError, ValueError) as exc:
                        print(exc, file=sys.stderr)
                        return 1
                    if phase == INITIAL_PHASE:
                        initial_rail_section_count += 1
                    else:
                        rescan_rail_section_count += 1
                    if display_controller is not None:
                        captured_display = (
                            stack_dual_camera_displays(
                                capture_frame,
                                top_capture_camera_frame.frame,
                            )
                            if dual_camera_enabled
                            else capture_frame.copy()
                        )
                        captured_until = time.monotonic() + CAPTURE_NOTICE_SECONDS
                    captured_this_iteration = True
                    phase_rail_section_count += 1
                    print(
                        f"{phase_label} Rail Sections: "
                        f"{phase_rail_section_count}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET}"
                    )
                    if phase_rail_section_count >= AUTOMATIC_RAIL_SECTION_TARGET:
                        next_trigger = (
                            RETURN_START_TRIGGER
                            if phase == INITIAL_PHASE
                            else RESCAN_DONE_TRIGGER
                        )
                        print(
                            f"{phase_label} Rail Section scan complete; waiting for STM "
                            f"{next_trigger}."
                        )
                elif message == RETURN_START_TRIGGER:
                    if mode != CAPTURE_SCAN_MODE:
                        print(
                            f"Out-of-order STM {RETURN_START_TRIGGER} ignored in "
                            f"{mode} mode.",
                            file=sys.stderr,
                        )
                        continue
                    if (
                        initial_rail_section_count
                        != AUTOMATIC_RAIL_SECTION_TARGET
                    ):
                        print(
                            f"Out-of-order STM {RETURN_START_TRIGGER} ignored: "
                            "initial Rail Sections are "
                            f"{initial_rail_section_count}/"
                            f"{AUTOMATIC_RAIL_SECTION_TARGET}.",
                            file=sys.stderr,
                        )
                        continue
                    mode = RETURN_MODE
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {RETURN_START_TRIGGER})")
                elif message == REALTIME_START_TRIGGER:
                    if mode != RETURN_MODE:
                        print(
                            f"Out-of-order STM {REALTIME_START_TRIGGER} ignored in "
                            f"{mode} mode.",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        prepare_realtime_transition(
                            cleaner_controller,
                            pump_controller,
                            analysis_worker,
                            additional_pairs=(
                                (
                                    front_cleaner_controller,
                                    front_pump_controller,
                                ),
                            ) if dual_camera_enabled else (),
                        )
                    except RuntimeError as exc:
                        print(f"REALTIME transition blocked: {exc}", file=sys.stderr)
                        return 1
                    mode = REALTIME_MODE
                    realtime_started_this_iteration = True
                    realtime_inference.reset()
                    realtime_control_history.reset()
                    if dual_camera_enabled:
                        top_realtime_inference.reset()
                        front_control_history.reset()
                        front_command_hold.clear()
                        side_command_hold.clear()
                        if uart is not None:
                            realtime_actuator_arbiter = RealtimeActuatorArbiter(uart)
                            realtime_actuator_arbiter.start()
                    realtime_display_cache.reset()
                    realtime_control_ready = False
                    front_control_ready = False
                    side_control_ready = False
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    if args.no_crack:
                        print(
                            "REALTIME TEST MODE: --no-crack bypass is active; "
                            "the crack safety interlock is NOT protecting the "
                            "cleaner or pump."
                        )
                    elif realtime_crack_detector is None and not multitask_enabled:
                        print(
                            "Cleaning locked OFF: crack detector is disabled, so "
                            "the y=540:720 no-crack condition cannot be verified."
                        )
                    print(f"Mode: {mode} (STM {REALTIME_START_TRIGGER})")
                elif message == RESCAN_RETURN_START_TRIGGER:
                    if mode != REALTIME_MODE:
                        print(
                            f"Protocol error: STM {RESCAN_RETURN_START_TRIGGER} "
                            f"arrived in {mode} mode.",
                            file=sys.stderr,
                        )
                        return 1
                    try:
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.close()
                            realtime_actuator_arbiter = None
                        if dual_camera_enabled:
                            front_command_hold.clear()
                            side_command_hold.clear()
                        for cleaner, pump in actuator_pairs:
                            cleaner.force_off()
                            pump.force_off()
                    except RuntimeError as exc:
                        print(f"Cleaning actuator control failed: {exc}", file=sys.stderr)
                        return 1
                    mode = RESCAN_RETURN_MODE
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {RESCAN_RETURN_START_TRIGGER})")
                elif message == RESCAN_START_TRIGGER:
                    if mode != RESCAN_RETURN_MODE or rescan_rail_section_count != 0:
                        print(
                            f"Protocol error: STM {RESCAN_START_TRIGGER} arrived "
                            f"in {mode} mode with {rescan_rail_section_count} rescan "
                            "Rail Sections.",
                            file=sys.stderr,
                        )
                        return 1
                    mode = RESCAN_MODE
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {RESCAN_START_TRIGGER})")
                elif message == RESCAN_DONE_TRIGGER:
                    if (
                        mode != RESCAN_MODE
                        or rescan_rail_section_count
                        != AUTOMATIC_RAIL_SECTION_TARGET
                    ):
                        print(
                            f"Protocol error: STM {RESCAN_DONE_TRIGGER} arrived in "
                            f"{mode} mode with {rescan_rail_section_count}/"
                            f"{AUTOMATIC_RAIL_SECTION_TARGET} rescan Rail Sections.",
                            file=sys.stderr,
                        )
                        return 1
                    mode = RESCAN_DONE_MODE
                    print(f"Mode: {mode} (STM {RESCAN_DONE_TRIGGER}); waiting for DONE")
                elif message == DONE_TRIGGER:
                    if mode != RESCAN_DONE_MODE:
                        print(
                            f"Protocol error: STM {DONE_TRIGGER} arrived in {mode} "
                            f"mode before {RESCAN_DONE_TRIGGER}.",
                            file=sys.stderr,
                        )
                        return 1
                    force_cleaning_pairs_safe_off(*actuator_pairs)
                    if not analysis_worker.wait_until_idle(
                        CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS
                    ):
                        print(
                            "DONE blocked: capture analysis did not become idle "
                            "within "
                            f"{CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS:.1f} seconds; "
                            "cleaning remains OFF and the mission is stopping.",
                            file=sys.stderr,
                        )
                        return 1
                    worker_failure = analysis_worker.failure_message()
                    if worker_failure is not None:
                        print(worker_failure, file=sys.stderr)
                        worker_failure_reported = worker_failure
                        return 1
                    mode = COMPLETE_MODE
                    mission_completed = True
                    try:
                        dashboard_manifest_path = finalize_dashboard_run(
                            workbook=workbook,
                            output_directory=CAPTURE_DIRECTORY,
                            status="complete",
                        )
                        dashboard_finalized = dashboard_manifest_path is not None
                    except Exception as exc:
                        print(
                            "Dashboard mission finalization failed; XLSX and "
                            f"captured images remain valid: {exc}",
                            file=sys.stderr,
                        )
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {DONE_TRIGGER})")

            if display_controller is None and mission_completed:
                print("Headless mission complete; exiting after STM DONE.")
                break

            if mode == REALTIME_MODE and realtime_started_this_iteration:
                try:
                    if dual_camera_enabled:
                        frame_read = read_dual_realtime_camera_frames(
                            camera,
                            top_camera,
                            after_side_sequence=last_camera_sequence,
                            after_top_sequence=last_top_camera_sequence,
                            side_inference=realtime_inference,
                            top_inference=top_realtime_inference,
                            side_cleaner_controller=cleaner_controller,
                            side_pump_controller=pump_controller,
                            front_cleaner_controller=front_cleaner_controller,
                            front_pump_controller=front_pump_controller,
                            actuator_arbiter=realtime_actuator_arbiter,
                        )
                        side_frame_available = frame_read.side_frame is not None
                        top_frame_available = frame_read.top_frame is not None
                        if side_frame_available:
                            camera_frame = frame_read.side_frame
                            last_camera_sequence = camera_frame.sequence
                        if top_frame_available:
                            top_camera_frame = frame_read.top_frame
                            last_top_camera_sequence = top_camera_frame.sequence
                        if frame_read.side_cache_expired:
                            realtime_inference.expire_cache()
                            realtime_control_history.prune()
                        if frame_read.top_cache_expired:
                            top_realtime_inference.expire_cache()
                            front_control_history.prune()
                    else:
                        camera_frame = camera.read_latest(
                            after_sequence=last_camera_sequence,
                            timeout=REALTIME_FRAME_TIMEOUT_SECONDS,
                        )
                except (RuntimeError, TimeoutError) as exc:
                    if realtime_actuator_arbiter is not None:
                        realtime_actuator_arbiter.clear_all()
                    else:
                        force_cleaning_pairs_safe_off(*actuator_pairs)
                    print(
                        f"Fresh frame after REALTIME_START failed: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                if not dual_camera_enabled:
                    last_camera_sequence = camera_frame.sequence
                frame = camera_frame.frame

            if display_controller is None and mode != REALTIME_MODE:
                continue

            if mode == REALTIME_MODE:
                captured_display = None
                captured_until = 0.0
                try:
                    realtime_outcome = None
                    top_realtime_outcome = None
                    side_waiting = (
                        dual_camera_enabled and frame_read.side_waiting
                    )
                    top_waiting = (
                        dual_camera_enabled and frame_read.top_waiting
                    )
                    side_inputs_fresh = (
                        side_frame_available
                        and realtime_role_inputs_are_fresh(
                            camera_frame,
                            realtime_inference,
                        )
                    )
                    if side_inputs_fresh:
                        water_roi, crack_roi = extract_realtime_control_rois(frame)
                        realtime_outcome = realtime_inference.process(
                            camera_frame,
                            water_roi,
                            crack_roi,
                        )
                    if side_inputs_fresh:
                        realtime_control_history.update(realtime_outcome)
                    elif not side_waiting:
                        realtime_outcome = None
                        realtime_control_history.prune()
                        realtime_inference.expire_cache()
                        realtime_display_cache.reset()

                    if dual_camera_enabled:
                        top_inputs_fresh = (
                            top_frame_available
                            and realtime_role_inputs_are_fresh(
                                top_camera_frame,
                                top_realtime_inference,
                            )
                        )
                        if top_inputs_fresh:
                            _, top_crack_roi = extract_realtime_control_rois(
                                top_camera_frame.frame
                            )
                            top_realtime_outcome = top_realtime_inference.process(
                                top_camera_frame,
                                top_camera_frame.frame,
                                top_crack_roi,
                            )
                        if top_inputs_fresh:
                            front_control_history.update(top_realtime_outcome)
                        elif not top_waiting:
                            top_realtime_outcome = None
                            front_control_history.prune()
                            top_realtime_inference.expire_cache()
                            top_obstacle_display = None
                            top_crack_display = None
                    if realtime_actuator_arbiter is not None:
                        realtime_actuator_arbiter.raise_if_failed()
                    if display_controller is not None:
                        realtime_display_cache.update(realtime_outcome)
                        if (
                            dual_camera_enabled
                            and top_realtime_outcome is not None
                        ):
                            if (
                                top_realtime_outcome.display_obstacle_result
                                is not None
                            ):
                                top_obstacle_display = (
                                    top_realtime_outcome.display_obstacle_result
                                )
                            if top_realtime_outcome.display_crack_result is not None:
                                top_crack_display = (
                                    top_realtime_outcome.display_crack_result
                                )
                except (RuntimeError, ValueError) as exc:
                    try:
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.clear_all()
                        else:
                            force_cleaning_pairs_safe_off(*actuator_pairs)
                    except RuntimeError as control_exc:
                        print(control_exc, file=sys.stderr)
                    print(f"Realtime inference failed: {exc}", file=sys.stderr)
                    return 1
                try:
                    if (
                        display_controller is not None
                        and display_controller.check_status()
                    ):
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.clear_all()
                        else:
                            force_cleaning_pairs_safe_off(*actuator_pairs)
                        break
                    if dual_camera_enabled:
                        side_has_new_decision = (
                            realtime_outcome is not None
                            and realtime_control_history.ready
                        )
                        side_immediate_off = side_outcome_requires_immediate_off(
                            realtime_outcome,
                            history_ready=realtime_control_history.ready,
                        )
                        if frame_read.side_error is not None or side_immediate_off:
                            if frame_read.side_error is not None:
                                realtime_control_history.reset()
                                realtime_inference.reset()
                                realtime_display_cache.reset()
                            side_command_hold.clear()
                            clear_realtime_actuator_role(
                                cleaner_controller,
                                pump_controller,
                                role="side",
                                arbiter=realtime_actuator_arbiter,
                            )
                            side_ready = False
                        elif side_has_new_decision:
                            if not side_control_ready:
                                print(
                                    "Side realtime control ready: four fresh rust "
                                    "and crack results are available."
                                )
                            publish_realtime_actuator_decision(
                                cleaner_controller,
                                pump_controller,
                                role="side",
                                history=realtime_control_history,
                                decision=realtime_control_history.decision(),
                                hold=side_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                            side_ready = True
                        else:
                            side_ready = maintain_waiting_role_control(
                                cleaner_controller,
                                pump_controller,
                                history=realtime_control_history,
                                inference=realtime_inference,
                                role="side",
                                hold=side_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                        if side_control_ready and not side_ready:
                            print(
                                "Side realtime control fail-closed: camera/error, "
                                "hazard, or 1.2 s command hold expired."
                            )
                        side_control_ready = side_ready

                        front_has_new_decision = (
                            top_realtime_outcome is not None
                            and front_control_history.ready
                        )
                        front_immediate_off = front_outcome_requires_immediate_off(
                            top_realtime_outcome
                        )
                        if frame_read.top_error is not None or front_immediate_off:
                            if frame_read.top_error is not None:
                                front_control_history.reset()
                                top_realtime_inference.reset()
                                top_obstacle_display = None
                                top_crack_display = None
                            front_command_hold.clear()
                            clear_realtime_actuator_role(
                                front_cleaner_controller,
                                front_pump_controller,
                                role="front",
                                arbiter=realtime_actuator_arbiter,
                            )
                            front_ready = False
                        elif front_has_new_decision:
                            if not front_control_ready:
                                print(
                                    "Front realtime control ready: four fresh "
                                    "obstacle and crack results are available."
                                )
                            publish_realtime_actuator_decision(
                                front_cleaner_controller,
                                front_pump_controller,
                                role="front",
                                history=front_control_history,
                                decision=front_control_history.decision(),
                                hold=front_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                            front_ready = True
                        else:
                            front_ready = maintain_waiting_role_control(
                                front_cleaner_controller,
                                front_pump_controller,
                                history=front_control_history,
                                inference=top_realtime_inference,
                                role="front",
                                hold=front_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                        if front_control_ready and not front_ready:
                            print(
                                "Front realtime control fail-closed: camera/error, "
                                "hazard, or 1.2 s command hold expired."
                            )
                        front_control_ready = front_ready
                    elif (
                        realtime_outcome is not None
                        and realtime_control_history.ready
                    ):
                        if not realtime_control_ready:
                            print(
                                "Realtime control ready: four fresh inference "
                                "results from each enabled model are available."
                            )
                        realtime_control_ready = True
                        update_cleaning_actuators(
                            cleaner_controller,
                            pump_controller,
                            decision=realtime_control_history.decision(),
                        )
                    else:
                        if realtime_control_ready:
                            print(
                                "Realtime control fail-closed: camera frame or "
                                "cached inference result exceeded 800 ms."
                            )
                        force_cleaning_safe_off(
                            cleaner_controller,
                            pump_controller,
                        )
                        realtime_control_ready = False
                except (RuntimeError, ValueError) as exc:
                    try:
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.clear_all()
                        else:
                            force_cleaning_pairs_safe_off(*actuator_pairs)
                    except RuntimeError as safe_off_exc:
                        print(safe_off_exc, file=sys.stderr)
                    print(f"Cleaning actuator control failed: {exc}", file=sys.stderr)
                    return 1
                if display_controller is None:
                    continue
                try:
                    realtime_annotated = annotate_realtime_control_results(
                        frame,
                        (
                            realtime_outcome.display_rust_result
                            if realtime_outcome is not None
                            else None
                        ),
                        (
                            realtime_outcome.display_crack_result
                            if realtime_outcome is not None
                            else None
                        ),
                    )
                    display = draw_realtime_roi_guide(
                        frame,
                        realtime_annotated,
                    )
                    display = draw_realtime_result_summary(
                        display,
                        realtime_display_cache.rust_result,
                        realtime_display_cache.crack_result,
                        crack_bypassed=args.no_crack,
                    )
                    if dual_camera_enabled:
                        top_display = annotate_top_realtime_results(
                            top_camera_frame.frame,
                            top_obstacle_display,
                            top_crack_display,
                        )
                        top_display = draw_realtime_roi_guide(
                            top_camera_frame.frame,
                            top_display,
                            primary_roi_label="OBSTACLE CONTROL",
                        )
                        display = stack_dual_camera_displays(display, top_display)
                except (RuntimeError, ValueError) as exc:
                    print(f"Realtime display rendering failed: {exc}", file=sys.stderr)
                    return 1
            elif captured_this_iteration and captured_display is not None:
                display = captured_display
            elif captured_display is not None and time.monotonic() < captured_until:
                display = captured_display
            else:
                captured_display = None
                try:
                    display = draw_roi_guide(frame)
                    if dual_camera_enabled:
                        top_display = draw_roi_guide(top_camera_frame.frame)
                        display = stack_dual_camera_displays(display, top_display)
                except ValueError as exc:
                    print(f"Could not draw ROI preview: {exc}", file=sys.stderr)
                    return 1
            if mode == CAPTURE_SCAN_MODE:
                if db_test_manual_capture:
                    db_phase = (
                        "BEFORE_CLEANING"
                        if initial_rail_section_count < AUTOMATIC_RAIL_SECTION_TARGET
                        else "AFTER_CLEANING"
                    )
                    db_total = initial_rail_section_count + rescan_rail_section_count
                    mode_label = (
                        f"DB_TEST phase={db_phase} total={db_total}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET * 2}"
                    )
                else:
                    mode_label = (
                        f"mode={mode} rail_sections="
                        f"{initial_rail_section_count}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET}"
                    )
            elif mode == RESCAN_MODE:
                mode_label = (
                    f"mode={mode} rail_sections="
                    f"{rescan_rail_section_count}/"
                    f"{AUTOMATIC_RAIL_SECTION_TARGET}"
                )
            else:
                mode_label = f"mode={mode}"
            (label_width, label_height), label_baseline = cv2.getTextSize(
                mode_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2,
            )
            label_x = max(12, display.shape[1] - label_width - 24)
            label_y = max(
                label_height + 12,
                display.shape[0] - (100 if mode == REALTIME_MODE else 24),
            )
            cv2.rectangle(
                display,
                (label_x - 8, label_y - label_height - 8),
                (label_x + label_width + 8, label_y + label_baseline + 8),
                (20, 35, 50),
                -1,
            )
            cv2.putText(
                display,
                mode_label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (60, 220, 90) if mode == REALTIME_MODE else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            try:
                if live_preview_publisher is not None:
                    try:
                        live_preview_publisher.submit(display)
                    except (FirebaseUploadError, ValueError) as exc:
                        print(f"Firebase live preview disabled: {exc}", file=sys.stderr)
                        live_preview_publisher.close()
                        live_preview_publisher = None
                if not display_controller.submit(display):
                    break
                key = display_controller.poll_key()
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1
            if key in (ord("s"), ord("S")):
                if mode != CAPTURE_SCAN_MODE:
                    print(f"Manual capture ignored while {mode} mode is active.")
                    continue
                if captured_this_iteration or captured_display is not None:
                    print(
                        "Manual capture ignored while the previous capture result "
                        "is displayed."
                    )
                    continue
                try:
                    if db_test_manual_capture:
                        manual_phase, manual_phase_sequence = db_test_capture_phase(
                            initial_rail_section_count,
                            rescan_rail_section_count,
                        )
                    else:
                        manual_phase = MANUAL_PHASE
                        manual_phase_sequence = None
                    manual_top_frame = None
                    if dual_camera_enabled:
                        side_manual, top_manual = read_stopped_capture_pair(
                            camera,
                            top_camera,
                        )
                        last_camera_sequence = side_manual.sequence
                        last_top_camera_sequence = top_manual.sequence
                        shared_captured_at = datetime.now()
                        if db_test_manual_capture:
                            queue_dual_capture_for_analysis(
                                side_manual.frame,
                                top_manual.frame,
                                manual_phase,
                                manual_phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                                captured_at=shared_captured_at,
                                side_frame_read_completed_at=(
                                    side_manual.read_completed_at
                                ),
                                top_frame_read_completed_at=(
                                    top_manual.read_completed_at
                                ),
                            )
                        else:
                            queue_capture_for_analysis(
                                side_manual.frame,
                                manual_phase,
                                manual_phase_sequence,
                                "manual",
                                analysis_worker,
                                camera_role=SIDE_CAMERA_ROLE,
                                captured_at=shared_captured_at,
                                frame_read_completed_at=side_manual.read_completed_at,
                            )
                            queue_capture_for_analysis(
                                top_manual.frame,
                                manual_phase,
                                manual_phase_sequence,
                                "manual",
                                analysis_worker,
                                camera_role=TOP_CAMERA_ROLE,
                                captured_at=shared_captured_at,
                                frame_read_completed_at=top_manual.read_completed_at,
                            )
                        frame = side_manual.frame
                        manual_top_frame = top_manual.frame
                    else:
                        queue_capture_for_analysis(
                            frame,
                            manual_phase,
                            manual_phase_sequence,
                            CAPTURE_TRIGGER if db_test_manual_capture else "manual",
                            analysis_worker,
                        )
                except (OSError, RuntimeError, ValueError) as exc:
                    print(exc, file=sys.stderr)
                    return 1
                captured_display = (
                    stack_dual_camera_displays(frame, manual_top_frame)
                    if dual_camera_enabled
                    else frame.copy()
                )
                captured_until = time.monotonic() + CAPTURE_NOTICE_SECONDS
                if db_test_manual_capture:
                    if manual_phase == INITIAL_PHASE:
                        initial_rail_section_count += 1
                    else:
                        rescan_rail_section_count += 1
                    db_total = initial_rail_section_count + rescan_rail_section_count
                    print(
                        f"DB test capture accepted: phase={manual_phase} "
                        f"phase_progress={manual_phase_sequence}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET}, total={db_total}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET * 2}."
                    )
                    if db_total == AUTOMATIC_RAIL_SECTION_TARGET:
                        print(
                            "청소 전 촬영 4회 완료. 청소를 수행한 뒤 S를 눌러 "
                            "청소 후 촬영 4회를 진행하세요."
                        )
                    if db_total == AUTOMATIC_RAIL_SECTION_TARGET * 2:
                        if not analysis_worker.wait_until_idle(
                            CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS
                        ):
                            print(
                                "DB test finalization blocked: capture analysis did "
                                "not become idle before timeout.",
                                file=sys.stderr,
                            )
                            return 1
                        worker_failure = analysis_worker.failure_message()
                        if worker_failure is not None:
                            print(worker_failure, file=sys.stderr)
                            worker_failure_reported = worker_failure
                            return 1
                        try:
                            candidate_manifest_path = finalize_dashboard_run(
                                workbook=workbook,
                                output_directory=CAPTURE_DIRECTORY,
                                status="complete",
                            )
                            if candidate_manifest_path is not None:
                                validate_db_test_dashboard_manifest(
                                    candidate_manifest_path
                                )
                        except Exception as exc:
                            dashboard_manifest_path = None
                            print(
                                "Dashboard DB test finalization failed; XLSX and "
                                f"captured images remain valid: {exc}",
                                file=sys.stderr,
                            )
                            return 1
                        if candidate_manifest_path is None:
                            dashboard_manifest_path = None
                            print(
                                "Dashboard DB test finalization did not produce a "
                                "manifest.",
                                file=sys.stderr,
                            )
                            return 1
                        dashboard_manifest_path = candidate_manifest_path
                        dashboard_finalized = True
                        mission_completed = True
                        mode = COMPLETE_MODE
                        db_test_publish_allowed = True
                        print(
                            "DB test capture 8/8 complete; finalized dashboard "
                            f"manifest: {dashboard_manifest_path}"
                        )
                        break
    finally:
        if live_preview_publisher is not None:
            live_preview_publisher.close()
        if realtime_actuator_arbiter is not None:
            try:
                realtime_actuator_arbiter.close()
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                realtime_actuator_shutdown_failure = str(exc)
                uart_safe_to_close = (
                    not realtime_actuator_arbiter.worker_is_alive()
                )
        if uart_safe_to_close:
            for cleaner, pump in actuator_pairs:
                for controller in (cleaner, pump):
                    try:
                        controller.force_off()
                    except RuntimeError as exc:
                        print(exc, file=sys.stderr)
                        if realtime_actuator_shutdown_failure is None:
                            realtime_actuator_shutdown_failure = str(exc)
        analysis_shutdown_failure = analysis_worker.shutdown()
        if (
            analysis_shutdown_failure is not None
            and analysis_shutdown_failure != worker_failure_reported
        ):
            print(analysis_shutdown_failure, file=sys.stderr)
        if not dashboard_finalized:
            dashboard_status = None
            dashboard_failure_reason = None
            if mission_completed:
                dashboard_status = "complete"
            elif (
                start_command_sent
                or start_acknowledged
                or initial_rail_section_count > 0
                or rescan_rail_section_count > 0
                or mode != CAPTURE_SCAN_MODE
            ):
                dashboard_status = "failed"
                dashboard_failure_reason = (
                    analysis_shutdown_failure
                    or realtime_actuator_shutdown_failure
                    or worker_failure_reported
                    or f"Runtime ended before STM DONE completion (mode={mode})."
                )
            if dashboard_status is not None:
                try:
                    dashboard_manifest_path = finalize_dashboard_run(
                        workbook=workbook,
                        output_directory=CAPTURE_DIRECTORY,
                        status=dashboard_status,
                        failure_reason=dashboard_failure_reason,
                    )
                except Exception as exc:
                    print(
                        "Dashboard mission finalization failed; XLSX and captured "
                        f"images remain valid: {exc}",
                        file=sys.stderr,
                    )
        close_runtime_resources(
            display=display_controller,
            uart=uart if uart_safe_to_close else None,
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        firebase_upload_safe = (
            uart_safe_to_close
            and realtime_actuator_shutdown_failure is None
        )
        firebase_upload_completed = publish_dashboard_manifest(
            dashboard_manifest_for_publish(
                dashboard_manifest_path,
                db_test_manual_capture=db_test_manual_capture,
                db_test_publish_allowed=db_test_publish_allowed,
            ),
            firebase_upload_safe=firebase_upload_safe,
            upload_required=firebase_upload_required,
        )

    if (
        analysis_shutdown_failure is not None
        or realtime_actuator_shutdown_failure is not None
        or not firebase_upload_completed
    ):
        return 1
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--legacy"]:
        del sys.argv[1]
        raise SystemExit(main())
    from node_runtime.ros.launcher import main as launch_ros
    raise SystemExit(launch_ros())
