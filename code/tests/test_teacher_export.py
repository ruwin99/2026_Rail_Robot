from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import export_capture_teacher_onnx as exporter  # noqa: E402


class _FakeTorch:
    def __init__(self) -> None:
        self.load = mock.Mock(return_value=object())


class LegacyPickleSafetyTests(unittest.TestCase):
    def test_hash_mismatch_never_calls_unsafe_torch_load(self) -> None:
        fake_torch = _FakeTorch()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "untrusted.pt"
            checkpoint.write_bytes(b"not the pinned checkpoint")

            with self.assertRaisesRegex(exporter.ExportError, "Refusing to unpickle"):
                exporter.load_verified_legacy_pickle(fake_torch, checkpoint)

        fake_torch.load.assert_not_called()

    def test_matching_hash_loads_verified_memory_on_cpu_once(self) -> None:
        payload = b"test-only legacy pickle bytes"
        expected = hashlib.sha256(payload).hexdigest()
        fake_torch = _FakeTorch()
        stream_positions: list[int] = []
        loaded_payloads: list[bytes] = []

        def record_stream_position(stream: object, **_kwargs: object) -> object:
            self.assertIsInstance(stream, io.BytesIO)
            stream_positions.append(stream.tell())  # type: ignore[attr-defined]
            loaded_payloads.append(stream.read())  # type: ignore[attr-defined]
            return object()

        fake_torch.load.side_effect = record_stream_position
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fixture.pt"
            checkpoint.write_bytes(payload)
            with mock.patch.object(exporter, "EXPECTED_RAW_SHA256", expected):
                loaded, actual, size = exporter.load_verified_legacy_pickle(
                    fake_torch,
                    checkpoint,
                )

        self.assertIsNotNone(loaded)
        self.assertEqual(actual, expected)
        self.assertEqual(size, len(payload))
        self.assertEqual(stream_positions, [0])
        self.assertEqual(loaded_payloads, [payload])
        fake_torch.load.assert_called_once()
        stream = fake_torch.load.call_args.args[0]
        self.assertTrue(stream.closed)
        self.assertEqual(fake_torch.load.call_args.kwargs["map_location"], "cpu")
        self.assertIs(fake_torch.load.call_args.kwargs["weights_only"], False)


class CliContractTests(unittest.TestCase):
    def test_defaults_are_fixed_and_help_discloses_pickle_risk(self) -> None:
        parser = exporter.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.raw_checkpoint, exporter.RAW_CHECKPOINT_RELATIVE)
        self.assertEqual(exporter.OPSET_VERSION, 18)
        self.assertEqual(
            args.onnx_output,
            Path(
                "models/virginia_tech_cssd/export/"
                "weighted_ce_r101_fp32_opset18.onnx"
            ),
        )
        self.assertEqual(args.ort, "auto")
        self.assertIn("legacy whole-model .pt pickle", parser.format_help())
        self.assertNotIn("expected-sha256", parser.format_help())

    def test_relative_paths_resolve_from_project_root(self) -> None:
        resolved = exporter._resolve_from_project(Path("models/example.pt"))

        self.assertEqual(
            resolved,
            (exporter.PROJECT_ROOT / "models/example.pt").resolve(),
        )

    def test_outputs_cannot_alias_raw_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = (Path(directory) / "raw.pt").resolve()
            checkpoint.write_bytes(b"fixture")
            other = (Path(directory) / "other.onnx").resolve()
            metadata = (Path(directory) / "metadata.json").resolve()

            with self.assertRaisesRegex(exporter.ExportError, "raw legacy"):
                exporter._validate_destinations(
                    checkpoint,
                    (checkpoint, other, metadata),
                    overwrite=True,
                )

    def test_official_factory_explicitly_disables_pretrained_backbone(self) -> None:
        factory = mock.Mock(return_value=object())

        model = exporter.build_teacher(factory)

        self.assertIsNotNone(model)
        factory.assert_called_once_with(
            num_classes=4,
            output_stride=8,
            pretrained_backbone=False,
        )

    def test_required_dependency_error_is_actionable(self) -> None:
        with mock.patch.object(
            exporter.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named 'missing'"),
        ):
            with self.assertRaisesRegex(
                exporter.ExportError,
                "Required dependency 'missing' is unavailable.*Install it",
            ):
                exporter._required_import("missing", "Install it.")

    def test_output_set_restores_previous_files_after_partial_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destinations = tuple(root / f"output-{index}" for index in range(3))
            temporaries = tuple(root / f"temporary-{index}" for index in range(3))
            for index, destination in enumerate(destinations):
                destination.write_bytes(f"old-{index}".encode())
            for index, temporary in enumerate(temporaries):
                temporary.write_bytes(f"new-{index}".encode())

            real_replace = exporter.os.replace
            replace_calls = 0

            def fail_during_second_commit(source: Path, destination: Path) -> None:
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 5:
                    raise OSError("simulated locked output")
                real_replace(source, destination)

            with (
                mock.patch.object(
                    exporter.os,
                    "replace",
                    side_effect=fail_during_second_commit,
                ),
                self.assertRaisesRegex(exporter.ExportError, "Previous outputs"),
            ):
                exporter._commit_output_set(
                    tuple(zip(temporaries, destinations)),
                    overwrite=True,
                )

            for index, destination in enumerate(destinations):
                self.assertEqual(destination.read_bytes(), f"old-{index}".encode())

    def test_output_lock_rejects_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "teacher.onnx"

            with exporter._exclusive_output_locks((destination,)):
                with self.assertRaisesRegex(exporter.ExportError, "Another exporter"):
                    with exporter._exclusive_output_locks((destination,)):
                        self.fail("a second writer acquired the same output lock")


if __name__ == "__main__":
    unittest.main()
