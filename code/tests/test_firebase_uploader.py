from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from firebase_uploader import (  # noqa: E402
    FirebaseLivePreviewPublisher,
    FirebaseUploadConfig,
    FirebaseUploadError,
    LIVE_PREVIEW_OBJECT_KEY,
    load_firebase_upload_config,
    upload_dashboard_manifest,
)


class FirebaseUploaderTests(unittest.TestCase):
    def test_live_preview_uses_a_fixed_object_key_and_required_metadata(self) -> None:
        calls: list[tuple[str, str, dict[str, str], bytes, str]] = []

        def fake_request(
            url: str,
            method: str,
            headers: dict[str, str],
            body: bytes,
            _timeout_seconds: float,
            operation: str,
        ) -> dict:
            calls.append((url, method, headers, body, operation))
            if "signInWithPassword" in url:
                return {"idToken": "id-token", "localId": "jetson-device-uid"}
            return {}

        config = FirebaseUploadConfig(
            api_key="public-web-key",
            project_id="eswcontest-rail-robot",
            storage_bucket="eswcontest-rail-robot.firebasestorage.app",
            email="device@example.invalid",
            password="private-password",
            expected_uid="jetson-device-uid",
        )
        payload = b"\xff\xd8\xffjpeg-preview"
        publisher = FirebaseLivePreviewPublisher(config, request_function=fake_request)
        try:
            publisher._upload(payload, hashlib.sha256(payload).hexdigest())
        finally:
            publisher.close()

        self.assertEqual(len(calls), 2)
        upload_url, method, headers, body, operation = calls[1]
        self.assertEqual(method, "POST")
        self.assertEqual(operation, "Firebase Storage upload for gui.jpg")
        self.assertIn("live%2Fjetson%2Fgui.jpg", upload_url)
        self.assertEqual(headers["Authorization"], "Firebase id-token")
        self.assertIn(b'"source":"jetson-live-preview"', body)
        self.assertIn(b'"artifactType":"gui_preview"', body)

    def test_missing_config_leaves_upload_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.env"
            config = load_firebase_upload_config(
                environment={"RAIL_ROBOT_FIREBASE_CONFIG_FILE": str(missing)}
            )

        self.assertIsNone(config)

    def test_environment_loads_pinned_device_identity(self) -> None:
        config = load_firebase_upload_config(
            environment={
                "RAIL_ROBOT_FIREBASE_UPLOAD": "1",
                "RAIL_ROBOT_FIREBASE_API_KEY": "public-web-key",
                "RAIL_ROBOT_FIREBASE_UID": "jetson-device-uid",
                "RAIL_ROBOT_FIREBASE_EMAIL": "device@example.invalid",
                "RAIL_ROBOT_FIREBASE_PASSWORD": "private-password",
            }
        )

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.expected_uid, "jetson-device-uid")
        self.assertEqual(config.project_id, "eswcontest-rail-robot")
        self.assertEqual(
            config.storage_bucket,
            "eswcontest-rail-robot.firebasestorage.app",
        )
        rendered = repr(config)
        self.assertNotIn("public-web-key", rendered)
        self.assertNotIn("device@example.invalid", rendered)
        self.assertNotIn("private-password", rendered)

    def test_uploads_artifacts_then_manifest_then_firestore_document(self) -> None:
        calls: list[tuple[str, str, dict[str, str], bytes, str]] = []

        def fake_request(
            url: str,
            method: str,
            headers: dict[str, str],
            body: bytes,
            _timeout_seconds: float,
            operation: str,
        ) -> dict:
            calls.append((url, method, headers, body, operation))
            if "signInWithPassword" in url:
                return {"idToken": "id-token", "localId": "jetson-device-uid"}
            return {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = Path(temporary_directory) / "outputs"
            media = outputs / "dashboard" / "media" / "run_demo"
            runs = outputs / "dashboard" / "runs"
            media.mkdir(parents=True)
            runs.mkdir(parents=True)
            artifact_path = media / "cap_demo_crack.png"
            artifact_payload = b"\x89PNG\r\n\x1a\npng-payload"
            artifact_path.write_bytes(artifact_payload)
            object_key = "dashboard/media/run_demo/cap_demo_crack.png"
            manifest = {
                "schema_version": 1,
                "exported_at_utc": "2026-09-03T00:00:00Z",
                "runs": [{"run_id": "run_demo", "status": "complete"}],
                "model_provenance": [],
                "captures": [],
                "analyses": [],
                "artifacts": [
                    {
                        "artifact_id": "cap_demo:crack_preview",
                        "capture_id": "cap_demo",
                        "artifact_type": "crack_preview",
                        "object_key": object_key,
                        "public_url": None,
                        "media_type": "image/png",
                        "width": 2,
                        "height": 2,
                        "sha256": hashlib.sha256(artifact_payload).hexdigest(),
                    }
                ],
                "run_summaries": [],
            }
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            config = FirebaseUploadConfig(
                api_key="public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="private-password",
            )

            result = upload_dashboard_manifest(
                manifest_path,
                config,
                request_function=fake_request,
            )

        self.assertEqual(result.run_id, "run_demo")
        self.assertEqual(result.artifact_count, 1)
        self.assertEqual(
            result.uploaded_bytes,
            len(artifact_payload) + len(json.dumps(manifest, ensure_ascii=False).encode("utf-8")),
        )
        self.assertEqual(len(calls), 4)
        self.assertIn("signInWithPassword", calls[0][0])
        self.assertIn("dashboard%2Fmedia%2Frun_demo", calls[1][0])
        self.assertEqual(calls[1][2]["Authorization"], "Firebase id-token")
        self.assertIn(artifact_payload, calls[1][3])
        self.assertIn("dashboard%2Fmanifests%2Frun_demo.json", calls[2][0])
        self.assertEqual(calls[3][1], "PATCH")
        self.assertEqual(calls[3][2]["Authorization"], "Bearer id-token")
        firestore_body = json.loads(calls[3][3].decode("utf-8"))
        fields = firestore_body["fields"]
        self.assertEqual(fields["source"], {"stringValue": "jetson-runtime"})
        self.assertEqual(
            fields["uploader_uid"], {"stringValue": "jetson-device-uid"}
        )
        self.assertEqual(
            fields["payload"]["mapValue"]["fields"]["runs"]["arrayValue"]
            ["values"][0]["mapValue"]["fields"]["run_id"],
            {"stringValue": "run_demo"},
        )

    def test_rejects_unpinned_authenticated_uid_before_upload(self) -> None:
        def fake_request(
            _url: str,
            _method: str,
            _headers: dict[str, str],
            _body: bytes,
            _timeout_seconds: float,
            _operation: str,
        ) -> dict:
            return {"idToken": "id-token", "localId": "wrong-uid"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = Path(temporary_directory) / "outputs" / "dashboard" / "runs"
            runs.mkdir(parents=True)
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [{"run_id": "run_demo", "status": "complete"}],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            config = FirebaseUploadConfig(
                api_key="public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="private-password",
            )

            with self.assertRaisesRegex(FirebaseUploadError, "pinned"):
                upload_dashboard_manifest(
                    manifest_path,
                    config,
                    request_function=fake_request,
                )

    def test_rejects_changed_artifact_bytes_before_storage_upload(self) -> None:
        calls: list[str] = []

        def fake_request(
            url: str,
            _method: str,
            _headers: dict[str, str],
            _body: bytes,
            _timeout_seconds: float,
            _operation: str,
        ) -> dict:
            calls.append(url)
            return {"idToken": "id-token", "localId": "jetson-device-uid"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = Path(temporary_directory) / "outputs"
            media = outputs / "dashboard" / "media" / "run_demo"
            runs = outputs / "dashboard" / "runs"
            media.mkdir(parents=True)
            runs.mkdir(parents=True)
            object_key = "dashboard/media/run_demo/cap_demo_crack.png"
            (media / "cap_demo_crack.png").write_bytes(
                b"\x89PNG\r\n\x1a\nchanged"
            )
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [{"run_id": "run_demo", "status": "complete"}],
                        "artifacts": [
                            {
                                "artifact_type": "crack_preview",
                                "object_key": object_key,
                                "media_type": "image/png",
                                "sha256": hashlib.sha256(b"original").hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = FirebaseUploadConfig(
                api_key="public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="private-password",
            )

            with self.assertRaisesRegex(FirebaseUploadError, "SHA-256 mismatch"):
                upload_dashboard_manifest(
                    manifest_path,
                    config,
                    request_function=fake_request,
                )

        self.assertEqual(len(calls), 1, "only authentication may occur before rejection")


if __name__ == "__main__":
    unittest.main()
