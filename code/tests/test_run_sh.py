from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


RUN_SH = Path(__file__).resolve().parents[1] / "scripts" / "run.sh"
DUAL_CAMERA_TEST_SH = Path(__file__).resolve().parents[1] / "scripts" / "dual_camera_test.sh"
FRONT_CAPTURE_TEST_SH = (
    Path(__file__).resolve().parents[1] / "scripts" / "front_capture_test.sh"
)
YOLO_DEBUG_SH = Path(__file__).resolve().parents[1] / "scripts" / "yolo_debug.sh"


class RunScriptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = RUN_SH.read_text(encoding="utf-8")
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        cls.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")

    def _execute(
        self,
        env_updates: dict[str, str] | None = None,
        *user_arguments: str,
        launcher: Path = RUN_SH,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
        if self.bash is None:
            self.skipTest("bash is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory) / "code"
            scripts = project_root / "scripts"
            venv_bin = project_root / ".venv" / "bin"
            scripts.mkdir(parents=True)
            venv_bin.mkdir(parents=True)
            script_path = scripts / "run.sh"
            script_path.write_bytes(RUN_SH.read_bytes())
            if DUAL_CAMERA_TEST_SH.is_file():
                (scripts / "dual_camera_test.sh").write_bytes(
                    DUAL_CAMERA_TEST_SH.read_bytes()
                )
            if FRONT_CAPTURE_TEST_SH.is_file():
                (scripts / "front_capture_test.sh").write_bytes(
                    FRONT_CAPTURE_TEST_SH.read_bytes()
                )
            launcher_path = scripts / launcher.name
            if launcher != RUN_SH and not launcher_path.exists():
                launcher_path.write_bytes(launcher.read_bytes())
            requirement = scripts / "requirement.sh"
            requirement.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            python_stub = venv_bin / "python"
            python_stub.write_text(
                "#!/usr/bin/env bash\n"
                'STUB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." '
                '&& pwd)"\n'
                'printf \'%s\\n\' "$@" > "${STUB_ROOT}/argv.txt"\n'
                'printf \'%s\\n\' "${NVIDIA_TF32_OVERRIDE-}" > '
                '"${STUB_ROOT}/tf32.txt"\n',
                encoding="utf-8",
            )
            os.chmod(requirement, 0o755)
            os.chmod(python_stub, 0o755)

            env = os.environ.copy()
            for name in tuple(env):
                if name.startswith("RAIL_ROBOT_") or name == "NVIDIA_TF32_OVERRIDE":
                    env.pop(name)
            env.update(
                {
                    "RAIL_ROBOT_CAPTURE_RUST_ENGINE": "capture-rust.plan",
                    "RAIL_ROBOT_REALTIME_RUST_ENGINE": "realtime-rust.plan",
                    "RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256": "a" * 64,
                    "RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256": "b" * 64,
                    "RAIL_ROBOT_REALTIME_CRACK_ENGINE": "realtime-crack.plan",
                    "RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256": "d" * 64,
                    "RAIL_ROBOT_REALTIME_CRACK_THRESHOLD": "0.67",
                    "RAIL_ROBOT_REALTIME_CRACK_MIN_COMPONENT_PIXELS": "34",
                }
            )
            if env_updates:
                env.update(env_updates)
            completed = subprocess.run(
                [self.bash, str(launcher_path),
                 *(["--legacy"] if launcher == RUN_SH else []), *user_arguments],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
            argv_path = project_root / "argv.txt"
            argv = argv_path.read_text(encoding="utf-8").splitlines() if argv_path.exists() else []
            tf32_path = project_root / "tf32.txt"
            tf32 = tf32_path.read_text(encoding="utf-8").strip() if tf32_path.exists() else ""
            return completed, argv, tf32

    def test_raw_script_has_lf_shebang_and_no_carriage_returns(self) -> None:
        raw = RUN_SH.read_bytes()

        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", raw)

    def test_headless_flag_is_forwarded_to_runtime(self) -> None:
        completed, argv, _tf32 = self._execute(None, "--headless")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--headless"), 1)

    def test_no_crack_option_is_rejected(self) -> None:
        completed, argv, _tf32 = self._execute(None, "--no-crack")

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(argv, [])
        self.assertIn("--no-crack has been removed", completed.stderr)

    def test_dual_camera_shortcut_is_no_uart_realtime_with_pinned_obstacle(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--headless",
            launcher=DUAL_CAMERA_TEST_SH,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--realtime-test"), 1)
        self.assertNotIn("--realtime-test-uart", argv)
        self.assertIn("--headless", argv)
        self.assertIn("--side-camera-device", argv)
        self.assertIn("--top-camera-device", argv)
        self.assertIn("--obstacle-engine", argv)
        self.assertIn("--obstacle-engine-sha256", argv)
        self.assertEqual(
            argv[argv.index("--obstacle-engine") + 1],
            "/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/"
            "plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-"
            "int-h736-fp16-notf32-trt10.3.plan",
        )
        self.assertIn(
            "cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c",
            argv,
        )

    def test_dual_camera_shortcut_enables_uart_only_when_explicit(self) -> None:
        argument_sets = (
            ("--uart",),
            ("--headless", "--uart"),
            ("--uart", "--headless"),
        )
        for user_arguments in argument_sets:
            with self.subTest(user_arguments=user_arguments):
                completed, argv, _tf32 = self._execute(
                    None,
                    *user_arguments,
                    launcher=DUAL_CAMERA_TEST_SH,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(argv.count("--realtime-test-uart"), 1)
                self.assertNotIn("--uart", argv)
                if "--headless" in user_arguments:
                    self.assertEqual(argv.count("--headless"), 1)

    def test_dual_camera_debug_is_headless_bounded_mode_without_uart(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--debug",
            launcher=DUAL_CAMERA_TEST_SH,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--dual-debug-benchmark"), 1)
        self.assertEqual(argv.count("--headless"), 1)
        self.assertNotIn("--realtime-test-uart", argv)
        self.assertNotIn("--dual-timing", argv)

    def test_dual_camera_debug_rejects_uart_and_timing(self) -> None:
        for user_arguments in (("--debug", "--uart"), ("--timing", "--debug")):
            with self.subTest(user_arguments=user_arguments):
                completed, argv, _tf32 = self._execute(
                    None,
                    *user_arguments,
                    launcher=DUAL_CAMERA_TEST_SH,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(argv, [])

    def test_dual_camera_rust_debug_is_headless_bounded_mode_without_uart(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--rust-debug",
            launcher=DUAL_CAMERA_TEST_SH,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--dual-rust-debug-benchmark"), 1)
        self.assertEqual(argv.count("--headless"), 1)
        self.assertNotIn("--dual-debug-benchmark", argv)
        self.assertNotIn("--realtime-test-uart", argv)
        self.assertNotIn("--dual-timing", argv)

    def test_dual_camera_rust_debug_accepts_underscore_alias(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--rust_debug",
            launcher=DUAL_CAMERA_TEST_SH,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--dual-rust-debug-benchmark"), 1)
        self.assertEqual(argv.count("--headless"), 1)

    def test_dual_camera_rust_debug_rejects_unsafe_combinations(self) -> None:
        for user_arguments in (
            ("--rust-debug", "--uart"),
            ("--timing", "--rust-debug"),
            ("--debug", "--rust-debug"),
        ):
            with self.subTest(user_arguments=user_arguments):
                completed, argv, _tf32 = self._execute(
                    None,
                    *user_arguments,
                    launcher=DUAL_CAMERA_TEST_SH,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(argv, [])

    def test_dual_camera_shortcut_rejects_unknown_or_duplicate_flags(self) -> None:
        argument_sets = (
            ("--unknown",),
            ("--uart", "--uart"),
            ("--headless", "--headless"),
            ("--debug", "--debug"),
            ("--rust-debug", "--rust-debug"),
        )
        for user_arguments in argument_sets:
            with self.subTest(user_arguments=user_arguments):
                completed, argv, _tf32 = self._execute(
                    None,
                    *user_arguments,
                    launcher=DUAL_CAMERA_TEST_SH,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("Usage:", completed.stderr)
                self.assertEqual(argv, [])

    def test_passes_four_explicit_engine_options(self) -> None:
        for option in (
            "--teacher-engine",
            "--student-engine",
            "--capture-hrsegnet-crack-engine",
            "--realtime-hrsegnet-crack-engine",
        ):
            with self.subTest(option=option):
                self.assertIn(option, self.script)

    def test_defaults_identify_the_native_static_shapes(self) -> None:
        for filename in (
            "corrosion-capture-r101-os8-w1280-h720-fp32.plan",
            "capture-rust-r101-os8-hardneg-v6-w1280-h720-fp32-notf32.plan",
            "realtime-rust-mnv2-os16-finetuned-w1280-h240-fp16-super25w-7d61dbcf.plan",
            "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan",
            "hrsegnet-b32-steelcrack-aug400-realtime-w1280-h128-fp16.plan",
            "obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-"
            "int-h736-fp16-notf32-trt10.3.plan",
        ):
            with self.subTest(filename=filename):
                self.assertIn(filename, self.script)

    def test_tf32_override_is_process_wide_and_fail_closed(self) -> None:
        self.assertIn('export NVIDIA_TF32_OVERRIDE=0', self.script)
        self.assertIn('"${NVIDIA_TF32_OVERRIDE}" != "0"', self.script)

    def test_legacy_shared_crack_engine_is_rejected(self) -> None:
        self.assertIn("RAIL_ROBOT_CRACK_ENGINE is no longer supported", self.script)

    def test_normal_operation_pins_dual_cameras_and_all_models(self) -> None:
        completed, argv, tf32 = self._execute(None)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(argv[0].replace("\\", "/").endswith("/jetson_code/run.py"))
        expected_pairs = {
            "--teacher-engine": "capture-rust.plan",
            "--student-engine": (
                "/home/bitdol/models/plans_new/"
                "realtime-rust-mnv2-os16-finetuned-w1280-h240-fp16-super25w-7d61dbcf.plan"
            ),
            "--student-engine-sha256": (
                "e5093573ce43ec5f4c37b5e4de5321bd9af9dfacbcbb3cdc28f51bdebe0997cc"
            ),
            "--capture-hrsegnet-crack-engine": (
                "/home/bitdol/models/hrsegnet_b32_20260816/plans/"
                "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
            ),
            "--capture-hrsegnet-crack-engine-sha256": (
                "01c2dc7e4467e8888eda7eea68670ba6598ad4f7b664e57d420bda76e293111a"
            ),
            "--capture-hrsegnet-crack-probability-threshold": "0.55",
            "--capture-hrsegnet-crack-min-component-pixels": "20",
            "--realtime-hrsegnet-crack-engine": (
                "/home/bitdol/models/hrsegnet_b32_steelcrack_aug400_20260924/plans/"
                "hrsegnet-b32-steelcrack-aug400-realtime-w1280-h128-fp16.plan"
            ),
            "--realtime-hrsegnet-crack-engine-sha256": (
                "964306b153b8ae9580851bb9e93157c621fbb397c57e8e5681ea3c4e71f72ce6"
            ),
            "--realtime-hrsegnet-crack-probability-threshold": "0.55",
            "--realtime-hrsegnet-crack-min-component-pixels": "20",
            "--side-camera-device": (
                "/dev/v4l/by-path/"
                "platform-3610000.usb-usb-0:2.1:1.0-video-index0"
            ),
            "--top-camera-device": (
                "/dev/v4l/by-path/"
                "platform-3610000.usb-usb-0:2.3:1.0-video-index0"
            ),
            "--obstacle-engine": (
                "/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/"
                "plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-"
                "int-h736-fp16-notf32-trt10.3.plan"
            ),
            "--obstacle-engine-sha256": (
                "cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c"
            ),
            "--obstacle-confidence-threshold": "0.424",
        }
        for option, value in expected_pairs.items():
            with self.subTest(option=option):
                index = argv.index(option)
                self.assertEqual(argv[index + 1], value)
        for option in (
            "--side-camera-device",
            "--top-camera-device",
            "--obstacle-engine",
            "--obstacle-engine-sha256",
            "--obstacle-confidence-threshold",
        ):
            self.assertEqual(argv.count(option), 1)
        self.assertNotIn("--camera-index", argv)
        self.assertNotIn("--realtime-crack-engine", argv)
        self.assertNotIn("--realtime-multitask-engine", argv)
        self.assertIn("Ignoring legacy BGCrack/multitask", completed.stderr)
        self.assertEqual(tf32, "0")

    def test_normal_operation_rejects_camera_and_obstacle_overrides(self) -> None:
        override_arguments = (
            ("--camera-index", "3"),
            ("--side-camera-device", "/dev/video0"),
            ("--top-camera-device", "/dev/video2"),
            ("--obstacle-engine", "other.plan"),
            ("--obstacle-engine-sha256", "f" * 64),
            ("--obstacle-confidence-threshold", "0.9"),
        )
        for arguments in override_arguments:
            with self.subTest(arguments=arguments):
                completed, argv, _tf32 = self._execute(None, *arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("pinned SIDE/TOP cameras", completed.stderr)
                self.assertEqual(argv, [])

    def test_capture_test_passes_only_capture_model_options(self) -> None:
        completed, argv, tf32 = self._execute(
            {
                # These intentionally incomplete/conflicting realtime settings
                # must not block a capture-only test.
                "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE": "unused.plan",
                "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256": "",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE": "unused-hrseg.plan",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256": "",
            },
            "--capture-test",
            "--camera-index",
            "2",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected_pairs = {
            "--teacher-engine": (
                "/home/bitdol/models/capture_r101_hardneg_v6_20260821/plans/"
                "capture-rust-r101-os8-hardneg-v6-w1280-h720-fp32-notf32.plan"
            ),
            "--teacher-engine-sha256": (
                "7c442abf7049bcc56d093e0dc7dd2a347caaee74dc8e20006b63ee032b6f22d4"
            ),
            "--capture-hrsegnet-crack-engine": (
                "/home/bitdol/models/hrsegnet_b32_20260816/plans/"
                "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
            ),
            "--capture-hrsegnet-crack-engine-sha256": (
                "01c2dc7e4467e8888eda7eea68670ba6598ad4f7b664e57d420bda76e293111a"
            ),
            "--capture-hrsegnet-crack-probability-threshold": "0.55",
            "--capture-hrsegnet-crack-min-component-pixels": "20",
        }
        for option, value in expected_pairs.items():
            with self.subTest(option=option):
                index = argv.index(option)
                self.assertEqual(argv[index + 1], value)
        for realtime_option in (
            "--student-engine",
            "--realtime-crack-engine",
            "--realtime-multitask-engine",
            "--realtime-crack-threshold",
            "--realtime-crack-min-component-pixels",
        ):
            with self.subTest(realtime_option=realtime_option):
                self.assertNotIn(realtime_option, argv)
        self.assertEqual(argv[-3:], ["--capture-test", "--camera-index", "2"])
        self.assertNotIn("--side-camera-device", argv)
        self.assertNotIn("--top-camera-device", argv)
        self.assertNotIn("--obstacle-engine", argv)
        self.assertEqual(tf32, "0")
        self.assertIn("Capture test rust engine (pinned)", completed.stderr)
        self.assertIn(
            "Ignoring capture rust environment overrides",
            completed.stderr,
        )

    def test_unapproved_hrsegnet_env_is_ignored_for_pinned_baseline(self) -> None:
        completed, argv, tf32 = self._execute(
            {
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE": "",
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256": "",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE": "hrsegnet.plan",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256": "7" * 64,
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD": "0.73",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS": "31",
            },
            "--realtime-test",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        expected_pairs = {
            "--realtime-hrsegnet-crack-engine": (
                "/home/bitdol/models/hrsegnet_b32_steelcrack_aug400_20260924/plans/"
                "hrsegnet-b32-steelcrack-aug400-realtime-w1280-h128-fp16.plan"
            ),
            "--realtime-hrsegnet-crack-engine-sha256": (
                "964306b153b8ae9580851bb9e93157c621fbb397c57e8e5681ea3c4e71f72ce6"
            ),
            "--realtime-hrsegnet-crack-probability-threshold": "0.55",
            "--realtime-hrsegnet-crack-min-component-pixels": "20",
        }
        for option, value in expected_pairs.items():
            self.assertEqual(argv[argv.index(option) + 1], value)
        self.assertNotIn("--realtime-crack-engine", argv)
        self.assertNotIn("--realtime-multitask-engine", argv)
        self.assertNotIn("--side-camera-device", argv)
        self.assertNotIn("--top-camera-device", argv)
        self.assertNotIn("--obstacle-engine", argv)
        self.assertIn("Ignoring unapproved realtime Rust/HrSegNet", completed.stderr)
        self.assertEqual(tf32, "0")

    def test_front_capture_shortcut_uses_only_pinned_front_camera(self) -> None:
        raw = FRONT_CAPTURE_TEST_SH.read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", raw)

        completed, argv, _tf32 = self._execute(
            None,
            launcher=FRONT_CAPTURE_TEST_SH,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--front-capture-test"), 1)
        self.assertNotIn("--capture-test", argv)
        self.assertEqual(argv.count("--capture-camera-device"), 1)
        self.assertEqual(
            argv[argv.index("--capture-camera-device") + 1],
            "/dev/v4l/by-path/"
            "platform-3610000.usb-usb-0:2.3:1.0-video-index0",
        )
        self.assertNotIn("--side-camera-device", argv)
        self.assertNotIn("--top-camera-device", argv)
        self.assertNotIn("--teacher-engine", argv)
        self.assertNotIn("--student-engine", argv)
        expected_models = {
            "--capture-hrsegnet-crack-engine": (
                "/home/bitdol/models/hrsegnet_b32_20260816/plans/"
                "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
            ),
            "--capture-hrsegnet-crack-engine-sha256": (
                "01c2dc7e4467e8888eda7eea68670ba6598ad4f7b664e57d420bda76e293111a"
            ),
            "--obstacle-engine": (
                "/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/"
                "plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-"
                "int-h736-fp16-notf32-trt10.3.plan"
            ),
            "--obstacle-engine-sha256": (
                "cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c"
            ),
        }
        for option, value in expected_models.items():
            self.assertEqual(argv[argv.index(option) + 1], value)

    def test_front_capture_shortcut_rejects_arguments(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--camera-index",
            "2",
            launcher=FRONT_CAPTURE_TEST_SH,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage:", completed.stderr)
        self.assertEqual(argv, [])

    def test_yolo_debug_shortcut_pins_only_top_camera_and_obstacle_model(self) -> None:
        raw = YOLO_DEBUG_SH.read_bytes()
        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", raw)

        completed, argv, tf32 = self._execute(None, launcher=YOLO_DEBUG_SH)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv.count("--yolo-debug-benchmark"), 1)
        self.assertEqual(argv.count("--top-camera-device"), 1)
        self.assertEqual(
            argv[argv.index("--top-camera-device") + 1],
            "/dev/v4l/by-path/"
            "platform-3610000.usb-usb-0:2.3:1.0-video-index0",
        )
        self.assertEqual(
            argv[argv.index("--obstacle-engine") + 1],
            "/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/"
            "plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-"
            "int-h736-fp16-notf32-trt10.3.plan",
        )
        self.assertEqual(
            argv[argv.index("--obstacle-engine-sha256") + 1],
            "cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c",
        )
        self.assertIn("--headless", argv)
        self.assertNotIn("--side-camera-device", argv)
        self.assertNotIn("--student-engine", argv)
        self.assertNotIn("--realtime-hrsegnet-crack-engine", argv)
        self.assertEqual(tf32, "0")

    def test_yolo_debug_shortcut_rejects_arguments(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--headless",
            launcher=YOLO_DEBUG_SH,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage:", completed.stderr)
        self.assertEqual(argv, [])

    def test_front_capture_mode_rejects_camera_or_model_overrides(self) -> None:
        for arguments in (
            ("--camera-index", "2"),
            ("--capture-camera-device", "/dev/video2"),
            ("--obstacle-engine", "other.plan"),
            ("--teacher-engine", "unused.plan"),
            ("--realtime-hrsegnet-crack-engine", "other.plan"),
        ):
            with self.subTest(arguments=arguments):
                completed, argv, _tf32 = self._execute(
                    None,
                    "--front-capture-test",
                    *arguments,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(argv, [])

    def test_legacy_capture_bgcrack_env_is_ignored_for_pinned_capture_hrsegnet(self) -> None:
        completed, argv, _tf32 = self._execute(
            {
                "RAIL_ROBOT_CAPTURE_CRACK_ENGINE": "legacy-bgcrack.plan",
                "RAIL_ROBOT_CAPTURE_CRACK_ENGINE_SHA256": "c" * 64,
                "RAIL_ROBOT_CAPTURE_CRACK_THRESHOLD": "0.9",
                "RAIL_ROBOT_CAPTURE_CRACK_MIN_COMPONENT_PIXELS": "999",
            },
            "--capture-test",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            argv[argv.index("--capture-hrsegnet-crack-engine") + 1],
            "/home/bitdol/models/hrsegnet_b32_20260816/plans/"
            "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan",
        )
        self.assertNotIn("--capture-crack-engine", argv)
        self.assertIn("Ignoring legacy capture BGCrack", completed.stderr)

    def test_direct_capture_model_cli_override_is_rejected(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--capture-test",
            "--capture-hrsegnet-crack-engine",
            "other.plan",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(argv, [])
        self.assertIn("Capture model CLI overrides are disabled", completed.stderr)

    def test_direct_teacher_engine_cli_override_is_rejected(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--capture-test",
            "--teacher-engine",
            "other.plan",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(argv, [])
        self.assertIn(
            "Teacher engine CLI overrides are disabled for --capture-test",
            completed.stderr,
        )

    def test_normal_operation_keeps_existing_teacher_cli_override(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--teacher-engine",
            "other.plan",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        teacher_indices = [
            index for index, value in enumerate(argv) if value == "--teacher-engine"
        ]
        self.assertEqual(len(teacher_indices), 2)
        self.assertEqual(argv[teacher_indices[-1] + 1], "other.plan")

    def test_direct_realtime_model_cli_override_is_rejected(self) -> None:
        options = (
            "--realtime-test",
            "--realtime-hrsegnet-crack-engine",
            "cli-hrsegnet.plan",
            "--realtime-hrsegnet-crack-engine-sha256",
            "8" * 64,
            "--realtime-hrsegnet-crack-probability-threshold",
            "0.8",
            "--realtime-hrsegnet-crack-min-component-pixels",
            "40",
        )
        completed, argv, _tf32 = self._execute(
            {
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE": "",
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256": "",
            },
            *options,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(argv, [])
        self.assertIn("Realtime model CLI overrides are disabled", completed.stderr)

    def test_hrsegnet_is_selected_for_normal_operation(self) -> None:
        completed, argv, _tf32 = self._execute(
            {
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE": "",
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256": "",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE": "hrsegnet.plan",
                "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256": "7" * 64,
            }
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--capture-hrsegnet-crack-engine", argv)
        self.assertIn("--realtime-hrsegnet-crack-engine", argv)
        self.assertNotIn("--realtime-crack-engine", argv)

    def test_training_mode_bypasses_engine_configuration(self) -> None:
        completed, argv, tf32 = self._execute(
            {
                "RAIL_ROBOT_CRACK_ENGINE": "legacy-unused.plan",
                "NVIDIA_TF32_OVERRIDE": "1",
            },
            "--training",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(argv), 1)
        self.assertTrue(argv[0].replace("\\", "/").endswith("/jetson_code/training_capture.py"))
        self.assertEqual(tf32, "1")

    def test_training_mode_rejects_other_options(self) -> None:
        completed, argv, _tf32 = self._execute(
            None,
            "--training",
            "--camera-index",
            "1",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--training must be used alone", completed.stderr)
        self.assertEqual(argv, [])

    def test_stale_realtime_env_cannot_replace_pinned_baseline(self) -> None:
        completed, argv, _tf32 = self._execute(
            {
                "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE": "optimized.plan",
                "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256": "e" * 64,
                "RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE": "optimized-rust.plan",
                "RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256": "f" * 64,
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE": "old-crack.plan",
                "RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256": "d" * 64,
            }
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--capture-hrsegnet-crack-engine", argv)
        self.assertIn("--student-engine", argv)
        self.assertIn("--realtime-hrsegnet-crack-engine", argv)
        self.assertNotIn("--optimized-student-engine", argv)
        self.assertNotIn("--realtime-multitask-engine", argv)
        self.assertNotIn("--realtime-crack-engine", argv)
        self.assertIn("Ignoring legacy BGCrack/multitask", completed.stderr)

    def test_legacy_shared_crack_settings_fail_before_execution(self) -> None:
        legacy_names = (
            "RAIL_ROBOT_CRACK_ENGINE",
            "RAIL_ROBOT_CRACK_ENGINE_SHA256",
            "RAIL_ROBOT_CRACK_THRESHOLD",
            "RAIL_ROBOT_CRACK_MIN_COMPONENT_PIXELS",
        )
        for name in legacy_names:
            with self.subTest(name=name):
                completed, argv, _tf32 = self._execute({name: "legacy"})
                self.assertEqual(completed.returncode, 2)
                self.assertIn("no longer supported", completed.stderr)
                self.assertEqual(argv, [])


if __name__ == "__main__":
    unittest.main()
