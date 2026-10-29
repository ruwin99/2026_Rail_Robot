"""음성 작업자는 ROS나 실제 사운드카드 없이도 시작 승인·취소 계약을 지킨다."""
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.audio_alerts import AudioAlerts


def until(predicate, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("audio worker did not reach expected state")


class FakeProcess:
    def __init__(self, args, *, immediate=False, **kwargs):
        self.args = args
        self.returncode = 0 if immediate else None
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for cue in ("mission_started", "mission_completed", "obstacle_detected"):
            with wave.open(str(Path(self.directory.name) / (cue + ".wav")), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\0\0" * 160)
        self.children = []

    def backend(self, *, immediate=False, **kwargs):
        def spawn(args, **options):
            child = FakeProcess(args, immediate=immediate, **options)
            self.children.append(child)
            return child
        backend = AudioAlerts(self.directory.name, popen=spawn, which=lambda _: "/usr/bin/aplay",
                              repeat_gap_sec=0.02, **kwargs)
        self.addCleanup(backend.close)
        return backend

    def request(self, cue="mission_started", **kwargs):
        return {"request_id": str(uuid.uuid4()), "cue": cue, "expires_at": 110.0, **kwargs}

    def results(self, backend, count=1):
        found = []
        def poll():
            found.extend(backend.poll_results())
            return len(found) >= count
        until(poll)
        return found

    def test_start_success_only_after_child_finishes_and_duplicate_reuses_result(self):
        backend = self.backend()
        request = self.request()
        backend.submit(request, 100.0)
        until(lambda: len(self.children) == 1)
        backend.submit(request, 100.0)
        self.assertEqual(backend.poll_results(), [])
        self.children[0].returncode = 0
        first = self.results(backend)[0]
        self.assertTrue(first["success"])
        backend.submit(request, 100.0)
        self.assertEqual(self.results(backend), [first])
        self.assertEqual(len(self.children), 1)

    def test_obstacle_interrupts_start_and_repeats_until_clear(self):
        backend = self.backend()
        backend.submit(self.request(), 100.0)
        until(lambda: len(self.children) == 1)
        backend.set_obstacle(True)
        self.assertFalse(self.results(backend)[0]["success"])
        until(lambda: len(self.children) == 2)
        self.assertTrue(self.children[0].terminated)
        self.assertIn("obstacle_detected.wav", self.children[1].args[-1])
        backend.set_obstacle(True)
        time.sleep(0.03)
        self.assertEqual(len(self.children), 2)
        self.children[1].returncode = 0
        until(lambda: len(self.children) == 3)
        backend.set_obstacle(False)
        until(lambda: self.children[2].terminated)
        time.sleep(0.05)
        self.assertEqual(len(self.children), 3)

    def test_completion_interleaves_at_warning_boundary_then_warning_resumes(self):
        backend = self.backend()
        backend.set_obstacle(True)
        until(lambda: len(self.children) == 1)
        backend.submit(self.request("mission_completed"), 100.0)
        self.assertEqual(backend.poll_results(), [])
        time.sleep(0.03)
        self.assertEqual(len(self.children), 1)
        self.assertFalse(self.children[0].terminated)
        self.children[0].returncode = 0
        until(lambda: len(self.children) == 2)
        self.assertIn("mission_completed.wav", self.children[1].args[-1])
        self.children[1].returncode = 0
        self.assertTrue(self.results(backend)[0]["success"])
        until(lambda: len(self.children) == 3)
        self.assertIn("obstacle_detected.wav", self.children[2].args[-1])
        self.assertTrue(backend.status()["obstacle_held"])

    def test_timeout_failure_and_shutdown_reap_child(self):
        backend = self.backend(playback_timeout_sec=0.06)
        backend.submit(self.request(), 100.0)
        self.assertFalse(self.results(backend)[0]["success"])
        self.assertTrue(self.children[0].terminated)
        backend.set_obstacle(True)
        until(lambda: len(self.children) >= 2)
        backend.close()
        self.assertTrue(self.children[-1].terminated)
        self.assertFalse(backend.status()["worker_alive"])

    def test_fault_cancels_active_and_queued_and_never_restarts_warning(self):
        backend = self.backend()
        backend.submit(self.request(), 100.0)
        until(lambda: len(self.children) == 1)
        backend.submit(self.request("mission_completed"), 100.0)
        backend.cancel_all("runtime fault")
        self.assertTrue(all(not item["success"] for item in self.results(backend, 2)))
        backend.set_obstacle(True)
        time.sleep(0.03)
        self.assertEqual(len(self.children), 1)

    def test_expired_and_malformed_requests_cannot_spawn(self):
        backend = self.backend()
        backend.submit(self.request(expires_at=99), 100.0)
        self.assertFalse(self.results(backend)[0]["success"])
        for request in (self.request(cue="../../evil"), self.request(request_id="abc"),
                        self.request(expires_at=float("nan")), self.request(extra="path")):
            with self.assertRaises(ValueError):
                backend.submit(request, 100.0)
        self.assertEqual(self.children, [])

    def test_missing_wav_and_missing_player_fail_readiness_without_dead_worker(self):
        backend = self.backend()
        Path(self.directory.name, "mission_started.wav").unlink()
        self.assertFalse(backend.readiness()[0])
        backend.submit(self.request(), 100.0)
        self.assertFalse(self.results(backend)[0]["success"])
        self.assertTrue(backend.status()["worker_alive"])
        missing = AudioAlerts(self.directory.name, which=lambda _: None)
        self.addCleanup(missing.close)
        self.assertFalse(missing.readiness()[0])

    def test_request_queue_is_bounded_and_pending_duplicate_does_not_grow(self):
        backend = self.backend(queue_limit=2)
        backend.set_obstacle(True)
        until(lambda: len(self.children) == 1)
        first = self.request("mission_completed")
        backend.submit(first, 100.0)
        backend.submit(first, 100.0)
        backend.submit(self.request("mission_completed"), 100.0)
        backend.submit(self.request("mission_completed"), 100.0)
        self.assertFalse(self.results(backend)[0]["success"])
        self.assertLessEqual(backend.status()["pending"], 2)

    def test_nonzero_exit_never_acknowledges_mission_start(self):
        backend = self.backend()
        backend.submit(self.request(), 100.0)
        until(lambda: len(self.children) == 1)
        self.children[0].returncode = 1
        result = self.results(backend)[0]
        self.assertFalse(result["success"])
        self.assertIn("code 1", result["error"])

    def test_expired_completion_never_plays_after_obstacle_clears(self):
        backend = self.backend()
        backend.set_obstacle(True)
        until(lambda: len(self.children) == 1)
        backend.submit(self.request("mission_completed", expires_at=100.04), 100.0)
        time.sleep(0.06)
        backend.set_obstacle(False)
        self.assertFalse(self.results(backend)[0]["success"])
        self.assertEqual(len(self.children), 1)

    def test_full_dedupe_cache_rejects_new_work_without_replaying_live_ids(self):
        backend = self.backend(immediate=True, queue_limit=1, cache_limit=1)
        first = self.request()
        backend.submit(first, 100.0)
        self.assertTrue(self.results(backend)[0]["success"])
        backend.submit(self.request(), 100.0)
        self.assertFalse(self.results(backend)[0]["success"])
        backend.submit(first, 100.0)
        self.assertTrue(self.results(backend)[0]["success"])
        self.assertEqual(len(self.children), 1)

    def test_reaper_escalates_to_kill_and_waits(self):
        child = FakeProcess(["aplay"])
        child.terminate = lambda: setattr(child, "terminated", True)
        AudioAlerts._reap(child)
        self.assertTrue(child.terminated)
        self.assertTrue(child.killed)
        self.assertEqual(child.returncode, -9)

    def test_truncated_wav_rejected_without_launching_player(self):
        path = Path(self.directory.name, "mission_started.wav")
        path.write_bytes(path.read_bytes()[:-4])
        backend = self.backend()
        self.assertFalse(backend.readiness()[0])
        backend.submit(self.request(), 100.0)
        self.assertFalse(self.results(backend)[0]["success"])
        self.assertEqual(self.children, [])

    def test_obstacle_does_not_truncate_completion_already_playing(self):
        backend = self.backend()
        backend.submit(self.request("mission_completed"), 100.0)
        until(lambda: len(self.children) == 1)
        backend.set_obstacle(True)
        self.assertFalse(backend.readiness()[0])
        time.sleep(0.03)
        self.assertFalse(self.children[0].terminated)
        self.assertEqual(len(self.children), 1)
        self.children[0].returncode = 0
        self.assertTrue(self.results(backend)[0]["success"])
        until(lambda: len(self.children) == 2)
        self.assertIn("obstacle_detected.wav", self.children[1].args[-1])

    def test_failed_process_cleanup_latches_fault_and_stops_new_children(self):
        backend = self.backend()
        backend.submit(self.request(), 100.0)
        until(lambda: len(self.children) == 1)
        self.children[0].terminate = lambda: (_ for _ in ()).throw(OSError("cannot terminate"))
        backend.set_obstacle(True)
        self.assertFalse(self.results(backend)[0]["success"])
        time.sleep(0.08)
        self.assertEqual(len(self.children), 1)
        self.assertFalse(backend.readiness()[0])
        self.assertIn("cleanup", backend.status()["error"])
        self.children[0].terminate = lambda: setattr(self.children[0], "returncode", -15)


if __name__ == "__main__":
    unittest.main()
