"""Small source-clock metrics, independent of ROS and GUI render frequency."""
from collections import deque
import math


class FrameRateMeter:
    """Estimate acquisition FPS from source sequence deltas over one second.

    Sequence gaps include frames skipped by the publisher. They must not turn
    camera acquisition FPS into topic delivery or inference throughput.
    """

    def __init__(self, window_seconds=1.0, stale_seconds=1.0):
        self.window_seconds = window_seconds
        self.stale_seconds = stale_seconds
        self._stream = None
        self._samples = deque()
        self._received_at = None

    def record(self, stream_id, sequence, source_seconds, now):
        if not stream_id or sequence < 0 or not all(map(math.isfinite, (source_seconds, now))):
            return
        if self._stream != stream_id:
            self._samples.clear()
            self._stream = stream_id
        if self._samples:
            last_sequence, last_stamp = self._samples[-1]
            if sequence == last_sequence:
                return
            if sequence < last_sequence or source_seconds <= last_stamp:
                self._samples.clear()
        self._samples.append((sequence, source_seconds))
        self._received_at = now
        while len(self._samples) > 2 and source_seconds - self._samples[1][1] >= self.window_seconds:
            self._samples.popleft()

    def fps(self, now):
        if self._received_at is None or not 0 <= now - self._received_at <= self.stale_seconds:
            return None
        if len(self._samples) < 2:
            return None
        first_sequence, first_stamp = self._samples[0]
        last_sequence, last_stamp = self._samples[-1]
        return (last_sequence - first_sequence) / (last_stamp - first_stamp)
