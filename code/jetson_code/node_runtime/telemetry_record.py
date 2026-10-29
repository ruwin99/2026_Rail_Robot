"""읽기 전용 텔레메트리 기록. 콜백은 제한된 큐만 사용하고 파일은 한 worker가 쓴다."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid


def _bounded_json(value, limit):
    # 큰 문자열/배열을 JSON으로 복제하기 전에 구조 크기를 제한한다.
    remaining = limit
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        remaining -= 8
        if depth > 12 or remaining < 0:
            raise ValueError('Telemetry exceeds structural limit')
        if isinstance(item, str):
            remaining -= len(item)
        elif isinstance(item, dict):
            if len(item) > 128:
                raise ValueError('Too many telemetry fields')
            pending.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif isinstance(item, (list, tuple)):
            if len(item) > 256:
                raise ValueError('Telemetry array exceeds limit')
            pending.extend((v, depth + 1) for v in item)
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise ValueError('Unsupported telemetry value')
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8') + b'\n'
    if len(data) > limit:
        raise ValueError('Telemetry exceeds byte limit')
    return data


class TelemetryWriter:
    """One uniquely named JSONL file; no image payload, rotation, or unbounded history."""

    def __init__(self, directory, *, topics, namespace='/', queue_capacity=256,
                 message_max_bytes=65536, max_bytes=256 * 1024 * 1024):
        if not 1 <= queue_capacity <= 4096 or not 1024 <= message_max_bytes <= 65536:
            raise ValueError('Invalid recording queue/message limits')
        if not 65536 < max_bytes <= 1024 * 1024 * 1024 or len(topics) > 32:
            raise ValueError('Invalid recording disk/topic limits')
        self.topics = dict(topics)
        self.message_max_bytes, self.max_bytes = message_max_bytes, max_bytes
        self._lock = threading.Lock()
        self._queue = queue.Queue(queue_capacity)
        self._stopping = threading.Event()
        self._accepting = True
        self.error = ''
        self.completed = self.limit_reached = False
        self._bytes = self._samples = 0
        self._counts = dict.fromkeys(self.topics, 0)
        self._written_counts = dict.fromkeys(self.topics, 0)
        self._loss = dict(queue_dropped=0, oversize_dropped=0, unknown_topic_dropped=0,
                          size_limit_dropped=0, serialization_errors=0)
        self._segment = None
        self._previous_state = None
        self._last_state_receipt_ns = None
        self.started_wall_ns = time.time_ns()
        self.started_monotonic_ns = time.monotonic_ns()
        utc = datetime.now(timezone.utc)
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / ('rail_record_' + utc.strftime('%Y%m%dT%H%M%S.%fZ_') + uuid.uuid4().hex[:12] + '.jsonl')
        self._file = self.path.open('xb')
        self._metadata = dict(event='metadata', schema_version=1, utc=utc.isoformat(),
            receipt_wall_ns=self.started_wall_ns, receipt_monotonic_ns=self.started_monotonic_ns,
            ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'), namespace=namespace, topics=self.topics,
            physical_mapping={'left_*': 'physical RIGHT ENC1/TIM3', 'right_*': 'physical LEFT ENC2/TIM4'},
            pi_internal_fields_available=True,
            pi_sample_semantics='Latest 10ms STM control sample reported at the 100ms diagnostic cadence.',
            counter_warning='Wrapped counters are raw; no RPM, distance or speed ratio is inferred.',
            loss_accounting='Counters only measure recorder-side loss after receipt; DDS/UART loss is not measurable here.',
            segment_semantics='Observed valid applied-direction samples; endpoints are receipt times, not physical motion times.',
            qos={'reliability': 'best_effort', 'durability': 'volatile', 'depth': 50},
            queue_capacity=queue_capacity, message_max_bytes=message_max_bytes, max_bytes=max_bytes,
            privacy='Explicit robot-motion/status allowlist; images and WebRTC/Firebase payloads excluded.')
        self._thread = threading.Thread(target=self._run, name='rail-record-writer', daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._file.close()
            self.path.unlink(missing_ok=True)
            raise

    def note_serialization_error(self):
        with self._lock:
            self._loss['serialization_errors'] += 1

    def submit(self, topic, data, *, wall_ns=None, monotonic_ns=None):
        wall_ns = time.time_ns() if wall_ns is None else wall_ns
        monotonic_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        with self._lock:
            if not self._accepting or self.error:
                return False
            if topic not in self.topics:
                self._loss['unknown_topic_dropped'] += 1
                return False
            self._counts[topic] += 1
        record = dict(event='sample', topic=topic, receipt_wall_ns=wall_ns,
                      receipt_monotonic_ns=monotonic_ns, data=data)
        try:
            encoded = _bounded_json(record, self.message_max_bytes)
        except (ValueError, TypeError, OverflowError, RecursionError):
            with self._lock:
                self._loss['oversize_dropped'] += 1
            return False
        with self._lock:
            if not self._accepting or self.error:
                return False
            try:
                # immutable bytes freeze the sample before the callback releases its message.
                self._queue.put_nowait(encoded)
                return True
            except queue.Full:
                self._loss['queue_dropped'] += 1
                return False

    def _write_record(self, record):
        data = _bounded_json(record, max(self.message_max_bytes, 32768))
        if self._bytes + len(data) > self.max_bytes:
            raise OSError('Recording disk limit exhausted while finalizing')
        if self._file.write(data) != len(data):
            raise OSError('Incomplete recording write')
        self._bytes += len(data)

    def _end_segment(self, record, reason, unknown_end):
        if self._segment is not None:
            self._segment.update(end_receipt_wall_ns=record['receipt_wall_ns'],
                end_receipt_monotonic_ns=record['receipt_monotonic_ns'],
                end_reason=reason, unknown_end=unknown_end)
            self._write_record(self._segment)
            self._segment = None

    def _observe_segment(self, record):
        if not record['topic'].endswith('/debug/encoder/state'):
            return
        data = record['data']
        now = record['receipt_monotonic_ns']
        if self._last_state_receipt_ns is not None and now - self._last_state_receipt_ns > 600000000:
            self._end_segment(record, 'feedback_gap', True)
            self._previous_state = None
        self._last_state_receipt_ns = now
        session = data.get('session_id')
        valid = data.get('feedback_valid') is True and bool(session)
        direction = data.get('applied_direction') if valid else None
        moving = direction in ('forward', 'reverse') and data.get('applied_speed_percent', 0) > 0
        key = (session, direction) if moving else None
        if self._segment and key != (self._segment['session_id'], self._segment['direction']):
            observed_stop = valid and direction == 'stop' and session == self._segment['session_id']
            self._end_segment(record, 'observed_stop' if observed_stop else 'changed_or_invalid_feedback', not observed_stop)
        if moving and self._segment is None:
            self._segment = dict(event='segment', session_id=session, direction=direction,
                start_receipt_wall_ns=record['receipt_wall_ns'],
                start_receipt_monotonic_ns=record['receipt_monotonic_ns'],
                unknown_start=self._previous_state != (session, 'stop'))
        self._previous_state = (session, direction) if valid else None

    def _run(self):
        try:
            self._write_record(self._metadata)
            self._file.flush()
            last_flush = time.monotonic()
            while not self._stopping.is_set() or not self._queue.empty():
                try:
                    encoded = self._queue.get(timeout=0.1)
                except queue.Empty:
                    self._file.flush()
                    continue
                record = json.loads(encoded)
                # Reserve room for the final segment and loss/absence summary, never rotate.
                if self.limit_reached or self._bytes + len(encoded) + 65536 > self.max_bytes:
                    self.limit_reached = True
                    with self._lock:
                        self._loss['size_limit_dropped'] += 1
                    continue
                self._write_record(record)
                self._samples += 1
                self._written_counts[record['topic']] += 1
                self._observe_segment(record)
                if time.monotonic() - last_flush >= 1:
                    self._file.flush()
                    last_flush = time.monotonic()
            end = dict(receipt_wall_ns=time.time_ns(), receipt_monotonic_ns=time.monotonic_ns())
            self._end_segment(end, 'recording_ended', True)
            with self._lock:
                summary = dict(event='summary', **end, samples_written=self._samples,
                    received_by_topic=dict(self._counts), written_by_topic=dict(self._written_counts),
                    topics_without_samples=[topic for topic, count in self._counts.items() if not count],
                    **self._loss, limit_reached=self.limit_reached, writer_error=self.error)
            self._write_record(summary)
            self._file.flush()
            os.fsync(self._file.fileno())
            self.completed = not bool(self.error)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'[:1000]
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self.error = f'File close failed: {exc}'[:1000]
                self.completed = False

    def stop_accepting(self):
        with self._lock:
            self._accepting = False

    def close(self, timeout=5.0):
        self.stop_accepting()
        self._stopping.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.error = 'Recording writer did not finish within shutdown timeout'
            self.completed = False
        return self.completed and not self.error

    def abort(self):
        """Only for failed node construction; remove this newly created incomplete run."""
        self.close()
        if not self._thread.is_alive():
            self.path.unlink(missing_ok=True)
