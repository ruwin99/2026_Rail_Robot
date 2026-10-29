from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
import socket
import stat
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_CONFIG_PATH = Path("~/.config/rail_robot/firebase.env")
DEFAULT_PROJECT_ID = "eswcontest-rail-robot"
DEFAULT_STORAGE_BUCKET = "eswcontest-rail-robot.firebasestorage.app"
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_REQUEST_ATTEMPTS = 3
LIVE_PREVIEW_OBJECT_KEY = "live/jetson/gui.jpg"
LIVE_PREVIEW_INTERVAL_SECONDS = 5.0
LIVE_PREVIEW_JPEG_QUALITY = 75
RETRYABLE_HTTP_STATUS = frozenset((408, 429, 500, 502, 503, 504))
TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
FALSE_VALUES = frozenset(("0", "false", "no", "off"))


class FirebaseUploadError(RuntimeError):
    """Raised when a finalized local dashboard run cannot be published."""


class FirebaseLivePreviewPublisher:
    """Upload a bounded-rate GUI JPEG without blocking camera/control loops."""

    def __init__(
        self,
        config: FirebaseUploadConfig,
        *,
        interval_seconds: float = LIVE_PREVIEW_INTERVAL_SECONDS,
        request_function: RequestFunction | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Live preview interval must be positive.")
        self._config = config
        self._interval_seconds = float(interval_seconds)
        self._request_function = request_function or _request_with_retry
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firebase-preview")
        self._pending: Future | None = None
        self._last_submitted_at = float("-inf")

    @classmethod
    def from_environment(cls, *, environment=None) -> FirebaseLivePreviewPublisher | None:
        config = load_firebase_upload_config(environment=environment)
        return None if config is None else cls(config)

    def submit(self, frame) -> bool:
        """Queue the newest valid BGR GUI image if the interval and worker allow."""

        import cv2
        import numpy as np

        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError("Live preview must be a uint8 HxWx3 BGR frame.")
        now = time.monotonic()
        if now - self._last_submitted_at < self._interval_seconds:
            return False
        if self._pending is not None and not self._pending.done():
            return False
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            (cv2.IMWRITE_JPEG_QUALITY, LIVE_PREVIEW_JPEG_QUALITY),
        )
        if not ok:
            raise FirebaseUploadError("Could not JPEG-encode the live GUI preview.")
        payload = encoded.tobytes()
        if not payload or len(payload) > MAX_IMAGE_BYTES:
            raise FirebaseUploadError("Live GUI preview exceeds the upload limit.")
        self._last_submitted_at = now
        self._pending = self._pool.submit(
            self._upload,
            payload,
            hashlib.sha256(payload).hexdigest(),
        )
        return True

    def _upload(self, payload: bytes, sha256: str) -> None:
        id_token = _authenticate(self._config, self._request_function)
        _upload_storage_object(
            config=self._config,
            id_token=id_token,
            object_key=LIVE_PREVIEW_OBJECT_KEY,
            media_type="image/jpeg",
            payload=payload,
            custom_metadata={
                "source": "jetson-live-preview",
                "artifactType": "gui_preview",
                "sha256": sha256,
            },
            request_function=self._request_function,
        )

    def poll(self) -> bool | None:
        """Report a finished preview upload, propagating its error to the owner."""
        if self._pending is None or not self._pending.done():
            return None
        finished, self._pending = self._pending, None
        finished.result()
        return True

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class FirebaseUploadConfig:
    api_key: str = field(repr=False)
    expected_uid: str
    project_id: str = DEFAULT_PROJECT_ID
    storage_bucket: str = DEFAULT_STORAGE_BUCKET
    email: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Firebase Web API key is required.")
        if not self.expected_uid.strip():
            raise ValueError("The dedicated Jetson Firebase UID is required.")
        if not self.project_id.strip() or not self.storage_bucket.strip():
            raise ValueError("Firebase project ID and Storage bucket are required.")
        has_refresh = bool(self.refresh_token)
        has_password_pair = bool(self.email) and bool(self.password)
        if has_refresh == has_password_pair:
            raise ValueError(
                "Configure exactly one Firebase login method: refresh token or "
                "email/password."
            )
        if self.timeout_seconds <= 0:
            raise ValueError("Firebase request timeout must be positive.")


@dataclass(frozen=True)
class FirebaseUploadResult:
    run_id: str
    artifact_count: int
    uploaded_bytes: int
    firestore_document: str
    manifest_object_key: str


RequestFunction = Callable[[str, str, dict[str, str], bytes, float, str], dict]


def _parse_boolean(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise FirebaseUploadError(
        f"{name} must be one of: 1/0, true/false, yes/no, on/off."
    )


def _read_config_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    if os.name == "posix":
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise FirebaseUploadError(
                f"Firebase credential file is too permissive: {path}. "
                "Run chmod 600 on it."
            )

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FirebaseUploadError(
            f"Could not read Firebase credential file: {path}."
        ) from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise FirebaseUploadError(
                f"Invalid Firebase credential line {line_number} in {path}."
            )
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name.startswith("RAIL_ROBOT_FIREBASE_"):
            raise FirebaseUploadError(
                f"Unsupported setting {name!r} on line {line_number} in {path}."
            )
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def load_firebase_upload_config(
    *,
    environment: dict[str, str] | None = None,
) -> FirebaseUploadConfig | None:
    """Load private Jetson credentials without sourcing executable shell code."""

    env = os.environ if environment is None else environment
    configured_path = env.get("RAIL_ROBOT_FIREBASE_CONFIG_FILE")
    config_path = Path(configured_path or DEFAULT_CONFIG_PATH).expanduser()
    file_values = _read_config_file(config_path)
    values = {**file_values}
    values.update(
        {
            name: value
            for name, value in env.items()
            if name.startswith("RAIL_ROBOT_FIREBASE_")
            and name != "RAIL_ROBOT_FIREBASE_CONFIG_FILE"
        }
    )

    enabled_default = "1" if file_values else "0"
    enabled = _parse_boolean(
        values.get("RAIL_ROBOT_FIREBASE_UPLOAD", enabled_default),
        name="RAIL_ROBOT_FIREBASE_UPLOAD",
    )
    if not enabled:
        return None

    timeout_text = values.get(
        "RAIL_ROBOT_FIREBASE_TIMEOUT_SECONDS",
        str(DEFAULT_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(timeout_text)
    except ValueError as exc:
        raise FirebaseUploadError(
            "RAIL_ROBOT_FIREBASE_TIMEOUT_SECONDS must be numeric."
        ) from exc

    try:
        return FirebaseUploadConfig(
            api_key=values.get("RAIL_ROBOT_FIREBASE_API_KEY", ""),
            expected_uid=values.get("RAIL_ROBOT_FIREBASE_UID", ""),
            project_id=values.get(
                "RAIL_ROBOT_FIREBASE_PROJECT_ID", DEFAULT_PROJECT_ID
            ),
            storage_bucket=values.get(
                "RAIL_ROBOT_FIREBASE_STORAGE_BUCKET", DEFAULT_STORAGE_BUCKET
            ),
            email=values.get("RAIL_ROBOT_FIREBASE_EMAIL") or None,
            password=values.get("RAIL_ROBOT_FIREBASE_PASSWORD") or None,
            refresh_token=values.get("RAIL_ROBOT_FIREBASE_REFRESH_TOKEN") or None,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        raise FirebaseUploadError(f"Invalid Firebase upload configuration: {exc}") from exc


def _error_detail(payload: bytes) -> str:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message[:300]
    if isinstance(error, str):
        return error[:300]
    return ""


def _request_with_retry(
    url: str,
    method: str,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: float,
    operation: str,
) -> dict:
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            if not payload:
                return {}
            parsed = json.loads(payload.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise FirebaseUploadError(
                    f"{operation} returned an invalid JSON response."
                )
            return parsed
        except HTTPError as exc:
            detail = _error_detail(exc.read(4096))
            retryable = exc.code in RETRYABLE_HTTP_STATUS
            if not retryable or attempt == MAX_REQUEST_ATTEMPTS:
                suffix = f" ({detail})" if detail else ""
                raise FirebaseUploadError(
                    f"{operation} failed with HTTP {exc.code}{suffix}."
                ) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise FirebaseUploadError(
                    f"{operation} failed after {MAX_REQUEST_ATTEMPTS} attempts: "
                    "network unavailable or timed out."
                ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FirebaseUploadError(
                f"{operation} returned an invalid JSON response."
            ) from exc
        time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _authenticate(
    config: FirebaseUploadConfig,
    request_function: RequestFunction,
) -> str:
    if config.refresh_token:
        body = urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": config.refresh_token,
            }
        ).encode("utf-8")
        response = request_function(
            "https://securetoken.googleapis.com/v1/token?"
            + urlencode({"key": config.api_key}),
            "POST",
            {"Content-Type": "application/x-www-form-urlencoded"},
            body,
            config.timeout_seconds,
            "Firebase token refresh",
        )
        id_token = response.get("id_token")
        local_id = response.get("user_id")
    else:
        body = json.dumps(
            {
                "email": config.email,
                "password": config.password,
                "returnSecureToken": True,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        response = request_function(
            "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?"
            + urlencode({"key": config.api_key}),
            "POST",
            {"Content-Type": "application/json"},
            body,
            config.timeout_seconds,
            "Firebase device sign-in",
        )
        id_token = response.get("idToken")
        local_id = response.get("localId")

    if not isinstance(id_token, str) or not id_token:
        raise FirebaseUploadError("Firebase authentication returned no ID token.")
    if local_id != config.expected_uid:
        raise FirebaseUploadError(
            "Firebase authenticated UID does not match the pinned Jetson uploader UID."
        )
    return id_token


def _artifact_path(artifact_root: Path, object_key: str) -> Path:
    key = PurePosixPath(object_key)
    if key.is_absolute() or not key.parts or ".." in key.parts:
        raise FirebaseUploadError(f"Unsafe dashboard object key: {object_key!r}.")
    candidate = artifact_root.joinpath(*key.parts).resolve(strict=False)
    root = artifact_root.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FirebaseUploadError(
            f"Dashboard artifact escapes the local output directory: {object_key!r}."
        ) from exc
    if not candidate.is_file():
        raise FirebaseUploadError(f"Dashboard artifact is missing: {object_key}.")
    return candidate


def _validate_artifact_contract(
    artifact: dict,
    *,
    run_id: str,
) -> tuple[str, str]:
    artifact_type = artifact.get("artifact_type")
    object_key = artifact.get("object_key")
    media_type = artifact.get("media_type")
    expected = {
        "raw": ("captures/", "image/jpeg"),
        "analyzed": ("captures/", "image/jpeg"),
        "rust_preview": (f"dashboard/media/{run_id}/", "image/png"),
        "crack_preview": (f"dashboard/media/{run_id}/", "image/png"),
        "obstacle_preview": (f"dashboard/media/{run_id}/", "image/png"),
    }
    contract = expected.get(artifact_type)
    if contract is None:
        raise FirebaseUploadError(
            f"Unsupported dashboard artifact type: {artifact_type!r}."
        )
    key_prefix, required_media_type = contract
    if not isinstance(object_key, str) or not object_key.startswith(key_prefix):
        raise FirebaseUploadError(
            f"Dashboard {artifact_type} object key is outside its upload prefix."
        )
    if media_type != required_media_type:
        raise FirebaseUploadError(
            f"Dashboard {artifact_type} media type must be {required_media_type}."
        )
    return object_key, media_type


def _validate_image_signature(
    payload: bytes,
    *,
    media_type: str,
    object_key: str,
) -> None:
    signatures = {
        "image/jpeg": (b"\xff\xd8\xff",),
        "image/png": (b"\x89PNG\r\n\x1a\n",),
    }
    if not any(payload.startswith(signature) for signature in signatures[media_type]):
        raise FirebaseUploadError(
            f"Dashboard artifact bytes do not match {media_type}: {object_key}."
        )


def _multipart_body(
    *,
    object_key: str,
    media_type: str,
    payload: bytes,
    custom_metadata: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"railrobot-{uuid.uuid4().hex}"
    metadata = {
        "name": object_key,
        "contentType": media_type,
        "metadata": custom_metadata,
    }
    preamble = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n\r\n"
        + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        + f"\r\n--{boundary}\r\n"
        + f"Content-Type: {media_type}\r\n\r\n"
    ).encode("utf-8")
    ending = f"\r\n--{boundary}--".encode("ascii")
    return preamble + payload + ending, boundary


def _upload_storage_object(
    *,
    config: FirebaseUploadConfig,
    id_token: str,
    object_key: str,
    media_type: str,
    payload: bytes,
    custom_metadata: dict[str, str],
    request_function: RequestFunction,
) -> None:
    body, boundary = _multipart_body(
        object_key=object_key,
        media_type=media_type,
        payload=payload,
        custom_metadata=custom_metadata,
    )
    bucket = quote(config.storage_bucket, safe="")
    url = (
        f"https://firebasestorage.googleapis.com/v0/b/{bucket}/o?"
        + urlencode({"name": object_key})
    )
    request_function(
        url,
        "POST",
        {
            "Authorization": f"Firebase {id_token}",
            "Content-Type": f"multipart/related; boundary={boundary}",
            "X-Goog-Upload-Protocol": "multipart",
        },
        body,
        config.timeout_seconds,
        f"Firebase Storage upload for {PurePosixPath(object_key).name}",
    )


def _firestore_value(value) -> dict:
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FirebaseUploadError("Dashboard JSON contains a non-finite number.")
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_firestore_value(item) for item in value]}}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise FirebaseUploadError("Dashboard JSON object keys must be strings.")
        return {
            "mapValue": {
                "fields": {key: _firestore_value(item) for key, item in value.items()}
            }
        }
    raise FirebaseUploadError(
        f"Dashboard JSON contains unsupported value type: {type(value).__name__}."
    )


def upload_dashboard_manifest(
    manifest_path: str | Path,
    config: FirebaseUploadConfig,
    *,
    request_function: RequestFunction = _request_with_retry,
) -> FirebaseUploadResult:
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    if manifest_path.parent.name != "runs" or manifest_path.parent.parent.name != "dashboard":
        raise FirebaseUploadError(
            "Dashboard manifest must be under outputs/dashboard/runs/."
        )
    artifact_root = manifest_path.parent.parent.parent
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FirebaseUploadError(
            f"Could not read dashboard manifest: {manifest_path.name}."
        ) from exc

    runs = document.get("runs") if isinstance(document, dict) else None
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(runs, list) or len(runs) != 1:
        raise FirebaseUploadError("Dashboard manifest must contain exactly one run.")
    run_id = runs[0].get("run_id") if isinstance(runs[0], dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise FirebaseUploadError("Dashboard manifest run_id is invalid.")
    if runs[0].get("status") not in ("complete", "failed"):
        raise FirebaseUploadError(
            "Dashboard manifest is still in progress and cannot be published."
        )
    if not isinstance(artifacts, list):
        raise FirebaseUploadError("Dashboard manifest artifacts are invalid.")

    id_token = _authenticate(config, request_function)
    uploaded_bytes = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise FirebaseUploadError("Dashboard manifest contains an invalid artifact.")
        object_key, media_type = _validate_artifact_contract(
            artifact,
            run_id=run_id,
        )
        expected_sha256 = artifact.get("sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise FirebaseUploadError("Dashboard artifact metadata is incomplete.")
        path = _artifact_path(artifact_root, object_key)
        if path.stat().st_size >= MAX_IMAGE_BYTES:
            raise FirebaseUploadError(
                f"Dashboard artifact exceeds the 15 MiB upload limit: {object_key}."
            )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
            raise FirebaseUploadError(
                f"Dashboard artifact SHA-256 mismatch: {object_key}."
            )
        _validate_image_signature(
            payload,
            media_type=media_type,
            object_key=object_key,
        )
        _upload_storage_object(
            config=config,
            id_token=id_token,
            object_key=object_key,
            media_type=media_type,
            payload=payload,
            custom_metadata={
                "runId": run_id,
                "captureId": str(artifact.get("capture_id", "")),
                "artifactType": str(artifact.get("artifact_type", "")),
                "sha256": expected_sha256.lower(),
                "source": "jetson-runtime",
            },
            request_function=request_function,
        )
        uploaded_bytes += len(payload)

    manifest_payload = manifest_path.read_bytes()
    if len(manifest_payload) >= MAX_MANIFEST_BYTES:
        raise FirebaseUploadError("Dashboard manifest exceeds the 1 MiB upload limit.")
    manifest_object_key = f"dashboard/manifests/{run_id}.json"
    _upload_storage_object(
        config=config,
        id_token=id_token,
        object_key=manifest_object_key,
        media_type="application/json",
        payload=manifest_payload,
        custom_metadata={
            "runId": run_id,
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "source": "jetson-runtime",
        },
        request_function=request_function,
    )
    uploaded_bytes += len(manifest_payload)

    imported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    firestore_fields = {
        "payload": _firestore_value(document),
        "source": {"stringValue": "jetson-runtime"},
        "uploader_uid": {"stringValue": config.expected_uid},
        "imported_at": {"timestampValue": imported_at},
    }
    encoded_run_id = quote(run_id, safe="")
    document_path = f"inspection_exports/{run_id}"
    firestore_url = (
        "https://firestore.googleapis.com/v1/projects/"
        f"{quote(config.project_id, safe='')}/databases/(default)/documents/"
        f"inspection_exports/{encoded_run_id}"
    )
    request_function(
        firestore_url,
        "PATCH",
        {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json",
        },
        json.dumps(
            {"fields": firestore_fields},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
        config.timeout_seconds,
        f"Firestore publish for {run_id}",
    )

    return FirebaseUploadResult(
        run_id=run_id,
        artifact_count=len(artifacts),
        uploaded_bytes=uploaded_bytes,
        firestore_document=document_path,
        manifest_object_key=manifest_object_key,
    )


def upload_dashboard_manifest_from_environment(
    manifest_path: str | Path,
    *,
    environment: dict[str, str] | None = None,
    request_function: RequestFunction = _request_with_retry,
) -> FirebaseUploadResult | None:
    config = load_firebase_upload_config(environment=environment)
    if config is None:
        return None
    return upload_dashboard_manifest(
        manifest_path,
        config,
        request_function=request_function,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload finalized local rail-inspection JSON and images to Firebase."
    )
    parser.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        help=(
            "finalized outputs/dashboard/runs/*.json paths; when omitted, "
            "all local run manifests are retried"
        ),
    )
    args = parser.parse_args()
    manifests = args.manifests or sorted(
        Path("outputs/dashboard/runs").glob("*.json")
    )
    if not manifests:
        print("No local dashboard run manifests were found.")
        return 0

    try:
        config = load_firebase_upload_config()
        if config is None:
            raise FirebaseUploadError(
                "Firebase upload is disabled or its credential file is missing."
            )
        for manifest_path in manifests:
            result = upload_dashboard_manifest(manifest_path, config)
            print(
                "Firebase upload complete: "
                f"run_id={result.run_id}, artifacts={result.artifact_count}, "
                f"bytes={result.uploaded_bytes}"
            )
    except (FirebaseUploadError, OSError) as exc:
        print(f"Firebase upload failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
