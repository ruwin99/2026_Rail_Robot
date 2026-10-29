from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


DB_TEST_SH = Path(__file__).resolve().parents[1] / "scripts" / "db_test.sh"


class DbTestScriptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        cls.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")

    def _execute(
        self,
        *user_arguments: str,
        create_config: bool = True,
        custom_config: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], dict[str, str]]:
        if self.bash is None:
            self.skipTest("bash is unavailable")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project_root = root / "code"
            scripts = project_root / "scripts"
            scripts.mkdir(parents=True)
            (scripts / "db_test.sh").write_bytes(DB_TEST_SH.read_bytes())
            (scripts / "run.sh").write_text(
                "#!/usr/bin/env bash\n"
                ': > "${PROJECT_ROOT_FOR_TEST}/argv.txt"\n'
                "for argument in \"$@\"; do\n"
                '    printf \'%s\\n\' "${argument}" '
                '>> "${PROJECT_ROOT_FOR_TEST}/argv.txt"\n'
                "done\n"
                'printf \'%s\\n\' "${RAIL_ROBOT_FIREBASE_UPLOAD-}" '
                '> "${PROJECT_ROOT_FOR_TEST}/upload.txt"\n'
                'printf \'%s\\n\' "${RAIL_ROBOT_FIREBASE_CONFIG_FILE-}" '
                '> "${PROJECT_ROOT_FOR_TEST}/config-path.txt"\n'
                'printf \'%s\\n\' "${RAIL_ROBOT_FIREBASE_UPLOAD_REQUIRED-}" '
                '> "${PROJECT_ROOT_FOR_TEST}/upload-required.txt"\n'
                'printf \'%s\\n\' "${RAIL_ROBOT_DB_TEST_MANUAL_CAPTURE-}" '
                '> "${PROJECT_ROOT_FOR_TEST}/manual-db-test.txt"\n',
                encoding="utf-8",
            )

            home = root / "home"
            default_config = home / ".config" / "rail_robot" / "firebase.env"
            selected_config = (
                root / "private" / "firebase.env"
                if custom_config
                else default_config
            )
            if create_config:
                selected_config.parent.mkdir(parents=True)
                selected_config.write_text(
                    "RAIL_ROBOT_FIREBASE_UPLOAD=0\n",
                    encoding="utf-8",
                )

            env = os.environ.copy()
            for name in tuple(env):
                if name.startswith("RAIL_ROBOT_FIREBASE_"):
                    env.pop(name)
            env.update(
                {
                    "HOME": home.as_posix(),
                    "PROJECT_ROOT_FOR_TEST": project_root.as_posix(),
                }
            )
            if custom_config:
                env["RAIL_ROBOT_FIREBASE_CONFIG_FILE"] = selected_config.as_posix()

            completed = subprocess.run(
                [self.bash, str(scripts / "db_test.sh"), *user_arguments],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
            argv_path = project_root / "argv.txt"
            argv = argv_path.read_text(encoding="utf-8").splitlines() if argv_path.exists() else []
            captured_environment = {}
            for name, filename in (
                ("RAIL_ROBOT_FIREBASE_UPLOAD", "upload.txt"),
                ("RAIL_ROBOT_FIREBASE_CONFIG_FILE", "config-path.txt"),
                (
                    "RAIL_ROBOT_FIREBASE_UPLOAD_REQUIRED",
                    "upload-required.txt",
                ),
                ("RAIL_ROBOT_DB_TEST_MANUAL_CAPTURE", "manual-db-test.txt"),
            ):
                path = project_root / filename
                if path.exists():
                    captured_environment[name] = path.read_text(encoding="utf-8").strip()
            return completed, argv, captured_environment

    def test_raw_script_has_lf_shebang_and_no_carriage_returns(self) -> None:
        raw = DB_TEST_SH.read_bytes()

        self.assertTrue(raw.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", raw)

    def test_forces_upload_and_forwards_normal_arguments(self) -> None:
        completed, argv, captured_environment = self._execute(
            "--report",
            "outputs/db_test.xlsx",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            argv,
            ["--legacy", "--report", "outputs/db_test.xlsx"],
        )
        self.assertEqual(captured_environment["RAIL_ROBOT_FIREBASE_UPLOAD"], "1")
        self.assertEqual(
            captured_environment["RAIL_ROBOT_FIREBASE_UPLOAD_REQUIRED"],
            "1",
        )
        self.assertEqual(
            captured_environment["RAIL_ROBOT_DB_TEST_MANUAL_CAPTURE"],
            "1",
        )
        self.assertTrue(
            captured_environment["RAIL_ROBOT_FIREBASE_CONFIG_FILE"].endswith(
                "/.config/rail_robot/firebase.env"
            )
        )

    def test_accepts_existing_custom_config_path(self) -> None:
        completed, argv, captured_environment = self._execute(custom_config=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(argv, ["--legacy"])
        self.assertTrue(
            captured_environment["RAIL_ROBOT_FIREBASE_CONFIG_FILE"].endswith(
                "/private/firebase.env"
            )
        )

    def test_missing_config_stops_before_normal_mission(self) -> None:
        completed, argv, captured_environment = self._execute(create_config=False)

        self.assertEqual(completed.returncode, 1)
        self.assertIn("Firebase config not found", completed.stderr)
        self.assertEqual(argv, [])
        self.assertEqual(captured_environment, {})

    def test_rejects_modes_without_database_capture_output(self) -> None:
        for argument in (
            "--capture-test",
            "--capture-test=1",
            "--realtime-test",
            "--realtime-test=1",
            "--training",
            "--training=1",
            "--no-uart",
            "--no-uart=1",
            "--headless",
            "--headless=1",
        ):
            with self.subTest(argument=argument):
                completed, argv, captured_environment = self._execute(argument)

                self.assertEqual(completed.returncode, 2)
                self.assertIn("interactive GUI initial+rescan capture flow", completed.stderr)
                self.assertEqual(argv, [])
                self.assertEqual(captured_environment, {})


if __name__ == "__main__":
    unittest.main()
