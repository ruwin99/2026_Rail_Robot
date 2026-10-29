"""고정 WAV 세 종류를 재생하는 bounded worker. ROS·UART·모터를 직접 호출하지 않는다."""
from __future__ import annotations

from collections import OrderedDict, deque
from pathlib import Path
import math
import shutil
import subprocess
import threading
import time
import uuid
import wave


FILES = {cue: cue + ".wav" for cue in
         ("mission_started", "mission_completed", "obstacle_detected")}
REQUEST_CUES = frozenset(("mission_started", "mission_completed"))


class AudioAlerts:
    """한 worker·한 aplay 프로세스만 사용하고, 완료 후에만 success를 만든다.

    요청 만료 시각은 수신 시 ROS 시각에서 monotonic 기한으로 변환한다.
    장애물 상태는 큐 항목이 아니라 현재 상태이므로 true 반복이 소리를 중첩하지 않는다.
    """
    def __init__(self, audio_directory, audio_device="default", playback_timeout_sec=30.0,
                 repeat_gap_sec=0.5, *, queue_limit=8, cache_limit=128,
                 popen=subprocess.Popen, which=shutil.which):
        for name, value, minimum in (("playback_timeout_sec", playback_timeout_sec, 0.0),
                                     ("repeat_gap_sec", repeat_gap_sec, -0.000001)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= minimum:
                raise ValueError(f"{name} must be a finite {'positive' if minimum == 0 else 'nonnegative'} number")
        if not isinstance(audio_device, str) or not audio_device.strip() or "\x00" in audio_device:
            raise ValueError("audio_device must be a nonempty ALSA device name")
        if type(queue_limit) is not int or queue_limit < 1 or type(cache_limit) is not int or cache_limit < queue_limit:
            raise ValueError("Invalid audio queue/cache capacity")
        self.directory = Path(audio_directory).expanduser().resolve()
        self.device = audio_device
        self.timeout = float(playback_timeout_sec)
        self.repeat_gap = float(repeat_gap_sec)
        self._queue_limit, self._cache_limit = queue_limit, cache_limit
        self._popen, self._which = popen, which
        self._condition = threading.Condition(threading.RLock())
        self._pending = OrderedDict()
        self._cache = OrderedDict()
        self._results = deque(maxlen=cache_limit)
        self._active = None
        self._cancel = None
        self._obstacle = False
        self._closed = False
        self._fault = ""
        self._last_error = ""
        self._unreaped = None
        self._next_warning = 0.0
        self._worker = threading.Thread(target=self._run, name="rail-audio", daemon=True)
        self._worker.start()

    def _file(self, cue):
        path = self.directory / FILES[cue]
        # 경로는 설정 디렉터리와 고정 이름에서만 결정한다. 토픽으로 파일·명령을 받지 않는다.
        with wave.open(str(path), "rb") as audio:
            if (audio.getcomptype() != "NONE" or audio.getnchannels() not in (1, 2)
                    or audio.getsampwidth() not in (1, 2, 3, 4) or audio.getframerate() <= 0
                    or audio.getnframes() <= 0):
                raise ValueError(f"{path.name}: nonempty mono/stereo PCM WAV required")
            duration = audio.getnframes() / audio.getframerate()
            if duration > self.timeout:
                raise ValueError(f"{path.name}: WAV duration exceeds playback_timeout_sec")
            # 헤더만 있고 본문이 잘린 WAV는 시작 승인 음성으로 인정하지 않는다.
            frame_bytes = audio.getnchannels() * audio.getsampwidth()
            audio.setpos(audio.getnframes() - 1)
            if len(audio.readframes(1)) != frame_bytes:
                raise ValueError(f"{path.name}: truncated PCM data")
        return path

    def readiness(self):
        with self._condition:
            if self._closed or self._fault:
                return False, self._fault or "Audio worker is closed"
            if self._obstacle:
                return False, "Obstacle hold is active"
        try:
            if not self._which("aplay"):
                raise ValueError("aplay not found; install alsa-utils")
            for cue in FILES:
                self._file(cue)
        except (OSError, ValueError, EOFError, wave.Error) as exc:
            return False, str(exc)
        return True, "Three PCM WAV files and aplay are available; device playback is checked when requested"

    def submit(self, request, ros_now):
        if not isinstance(request, dict) or set(request) != {"request_id", "cue", "expires_at"}:
            raise ValueError("Expected request_id, cue and expires_at only")
        request_id, cue, expiry = request["request_id"], request["cue"], request["expires_at"]
        if not isinstance(request_id, str) or len(request_id) != 36:
            raise ValueError("request_id must be a canonical UUID")
        try:
            if str(uuid.UUID(request_id)) != request_id.lower():
                raise ValueError("request_id must be a canonical UUID")
        except (ValueError, AttributeError) as exc:
            raise ValueError("request_id must be a canonical UUID") from exc
        if not isinstance(cue, str) or cue not in REQUEST_CUES:
            raise ValueError("Unsupported audio cue")
        if (isinstance(expiry, bool) or not isinstance(expiry, (int, float))
                or not math.isfinite(expiry) or not math.isfinite(ros_now)):
            raise ValueError("expires_at and ROS time must be finite seconds")
        now = time.monotonic()
        item = dict(request_id=request_id, cue=cue, deadline=now + expiry - ros_now)
        with self._condition:
            if request_id in self._cache:
                previous, _ = self._cache[request_id]
                if previous["cue"] != cue:
                    raise ValueError("request_id cannot be reused for another cue")
                self._results.append(dict(previous))
                return
            if request_id in self._pending:
                if self._pending[request_id]["cue"] != cue:
                    raise ValueError("request_id cannot be reused for another cue")
                return
            # 완료 ID는 원래 기한까지 보존한다. 캐시가 차면 새 재생을 거절하여
            # 아직 유효한 ID를 축출한 뒤 동일 시작 음성을 다시 재생하는 일을 막는다.
            for identity, (_, deadline) in list(self._cache.items()):
                if deadline < now:
                    del self._cache[identity]
            if self._closed or self._fault:
                error = self._fault or "Audio worker is closed"
            elif expiry <= ros_now:
                error = "Audio request expired"
            elif expiry - ros_now > 300.0:
                error = "Audio request lifetime exceeds 300 seconds"
            elif len(self._pending) >= self._queue_limit or len(self._cache) + len(self._pending) >= self._cache_limit:
                error = "Audio request queue/cache is full"
            elif cue == "mission_started" and self._obstacle:
                error = "Obstacle hold prevents mission start announcement"
            else:
                self._pending[request_id] = item
                self._condition.notify_all()
                return
            self._finish(item, False, error)

    def _finish(self, item, success, error):
        result = dict(request_id=item["request_id"], cue=item["cue"],
                      success=bool(success), error=str(error))
        self._pending.pop(item["request_id"], None)
        if len(self._cache) < self._cache_limit or item["request_id"] in self._cache:
            self._cache[item["request_id"]] = (result, item["deadline"])
        self._results.append(dict(result))
        if error:
            self._last_error = str(error)

    def set_obstacle(self, held):
        if type(held) is not bool:
            raise ValueError("Obstacle state must be boolean")
        with self._condition:
            if self._closed or self._fault or held == self._obstacle:
                return
            self._obstacle = held
            if held:
                self._next_warning = 0.0
                # 시작 안내 도중 정지가 들어오면 승인하지 않는다. 이미 재생 중인
                # 완료 안내는 한 번 끝까지 읽는다. 이때 경고는 남은 클립 시간만큼
                # (최대 playback_timeout_sec) 지연되지만 STM 정지는 이미 이루어졌다.
                if self._active is not None and self._active["cue"] == "mission_started":
                    self._cancel.set()
                for item in list(self._pending.values()):
                    if item["cue"] == "mission_started" and item is not self._active:
                        self._finish(item, False, "Obstacle hold interrupted mission start announcement")
            elif self._active is not None and self._active["cue"] == "obstacle_detected":
                self._cancel.set()
            self._condition.notify_all()

    def cancel_all(self, reason="Audio cancelled"):
        with self._condition:
            self._fault = str(reason) or "Audio cancelled"
            self._obstacle = False
            if self._cancel is not None:
                self._cancel.set()
            for item in list(self._pending.values()):
                if item is not self._active:
                    self._finish(item, False, self._fault)
            self._condition.notify_all()

    def poll_results(self):
        with self._condition:
            results = list(self._results)
            self._results.clear()
            return results

    def status(self):
        with self._condition:
            return dict(active_cue=self._active["cue"] if self._active else "",
                        obstacle_held=self._obstacle, pending=len(self._pending),
                        error=self._fault or self._last_error, closed=self._closed,
                        worker_alive=self._worker.is_alive())

    @staticmethod
    def _reap(child):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=0.5)
        else:
            child.wait(timeout=0.5)

    def _play(self, item, cancel):
        child = None
        try:
            path = self._file(item["cue"])
            executable = self._which("aplay")
            if not executable:
                raise ValueError("aplay not found; install alsa-utils")
            if cancel.is_set():
                return False, "Audio playback cancelled"
            child = self._popen([executable, "-q", "-D", self.device, str(path)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, shell=False)
            deadline = min(item["deadline"], time.monotonic() + self.timeout)
            while True:
                if cancel.is_set():
                    return False, "Audio playback cancelled"
                if time.monotonic() >= deadline:
                    return False, "Audio playback timed out or request expired"
                code = child.poll()
                if code is not None:
                    return (True, "") if code == 0 else (False, f"aplay exited with code {code}")
                cancel.wait(0.02)
        except (OSError, ValueError, EOFError, wave.Error) as exc:
            return False, str(exc)
        finally:
            if child is not None:
                try:
                    self._reap(child)
                except Exception as exc:
                    # 이전 자식의 종료를 확인하지 못하면 새 aplay를 절대 만들지 않는다.
                    # 핸들을 보존하여 최종 shutdown에서 다시 terminate/kill/wait한다.
                    with self._condition:
                        self._unreaped = child
                        self.cancel_all(f"Audio process cleanup failed: {exc}")
                    raise

    def _run(self):
        while True:
            with self._condition:
                if self._closed:
                    return
                now = time.monotonic()
                for pending in list(self._pending.values()):
                    if now >= pending["deadline"]:
                        self._finish(pending, False, "Audio request expired while queued")
                item = None
                # 장애물이 오래 남아도 완료 안내가 영원히 밀리지 않게 한다.
                # 현재 경고 클립은 끝까지 읽고 다음 경계에서 완료를 한 번 읽은 뒤
                # HOLD가 유지되면 경고 반복으로 돌아간다. 시작 안내는 HOLD에서 거절된다.
                if not self._fault and self._obstacle:
                    item = next((pending for pending in self._pending.values()
                                 if pending["cue"] == "mission_completed"), None)
                if item is None and not self._fault and self._obstacle and now >= self._next_warning:
                    item = dict(cue="obstacle_detected", deadline=now + self.timeout)
                elif not self._fault and not self._obstacle and self._pending:
                    item = next(iter(self._pending.values()))
                if item is None:
                    self._condition.wait(0.05)
                    continue
                self._active = item
                self._cancel = cancel = threading.Event()
            try:
                success, error = self._play(item, cancel)
            except Exception as exc:
                # 예상치 못한 player 예외도 worker를 죽이지 않고 요청 실패로 회수한다.
                success, error = False, f"Audio worker error: {type(exc).__name__}: {exc}"
            with self._condition:
                if cancel.is_set() or self._fault or self._closed:
                    success, error = False, self._fault or "Audio playback cancelled"
                if item["cue"] == "obstacle_detected":
                    self._next_warning = time.monotonic() + max(self.repeat_gap, 1.0 if error else 0.0)
                    if error and not cancel.is_set():
                        self._last_error = error
                else:
                    self._finish(item, success, error)
                self._active = self._cancel = None
                self._condition.notify_all()

    def close(self):
        with self._condition:
            if not self._closed:
                self.cancel_all("Audio worker is closing")
                self._closed = True
                self._condition.notify_all()
        self._worker.join(timeout=3.0)
        if self._worker.is_alive():
            raise RuntimeError("Audio worker did not stop within 3 seconds")
        if self._unreaped is not None:
            self._reap(self._unreaped)
            self._unreaped = None
