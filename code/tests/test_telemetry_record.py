"""Bounded real-file recorder tests; no ROS or hardware."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson_code'))
from node_runtime.telemetry_record import TelemetryWriter

TOPICS = {'/debug/encoder/state': 'rail_robot_interfaces/msg/EncoderState',
          '/runtime/fault': 'std_msgs/msg/String'}


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def writer(self, **kwargs):
        writer = TelemetryWriter(self.temp.name, topics=TOPICS, namespace='/', **kwargs)
        self.addCleanup(writer.close)
        return writer

    def read(self, writer):
        return [json.loads(line) for line in writer.path.read_text(encoding='utf-8').splitlines()]

    def test_empty_run_unique_file_and_idempotent_close(self):
        first, second = self.writer(), self.writer()
        self.assertNotEqual(first.path, second.path)
        self.assertTrue(first.close())
        self.assertTrue(first.close())
        rows = self.read(first)
        self.assertEqual([r['event'] for r in rows], ['metadata', 'summary'])
        self.assertEqual(rows[-1]['samples_written'], 0)
        self.assertEqual(set(rows[-1]['topics_without_samples']), set(TOPICS))
        self.assertTrue(rows[0]['pi_internal_fields_available'])
        self.assertEqual(rows[0]['pi_sample_semantics'],
                         'Latest 10ms STM control sample reported at the 100ms diagnostic cadence.')

    def test_order_raw_fields_and_drain(self):
        writer = self.writer(queue_capacity=64)
        for n in range(20):
            self.assertTrue(writer.submit('/debug/encoder/state', {'sequence': n,
                'header': {'stamp': {'sec': 1, 'nanosec': n}}, 'left_raw_count': n}))
        self.assertTrue(writer.close())
        rows = self.read(writer)
        samples = [r for r in rows if r['event'] == 'sample']
        self.assertEqual([r['data']['sequence'] for r in samples], list(range(20)))
        self.assertEqual(samples[7]['data']['header']['stamp']['nanosec'], 7)
        self.assertIn('receipt_wall_ns', samples[0])
        self.assertIn('receipt_monotonic_ns', samples[0])
        self.assertEqual(rows[-1]['samples_written'], 20)

    def test_bounded_queue_reports_loss_without_blocking(self):
        entered, release = threading.Event(), threading.Event()
        original = TelemetryWriter._write_record
        def delayed(writer, record):
            if record['event'] == 'metadata':
                entered.set()
                release.wait(3)
            return original(writer, record)
        with patch.object(TelemetryWriter, '_write_record', delayed):
            writer = self.writer(queue_capacity=1)
            self.assertTrue(entered.wait(1))
            try:
                self.assertTrue(writer.submit('/runtime/fault', {'data': 'one'}))
                self.assertFalse(writer.submit('/runtime/fault', {'data': 'two'}))
            finally:
                release.set()
            self.assertTrue(writer.close())
        self.assertEqual(self.read(writer)[-1]['queue_dropped'], 1)

    def test_oversize_and_unknown_topics_are_reported_not_written(self):
        writer = self.writer(message_max_bytes=1024)
        self.assertFalse(writer.submit('/runtime/fault', {'data': 'x' * 5000}))
        self.assertFalse(writer.submit('/camera/side', {'data': 'image'}))
        self.assertTrue(writer.close())
        summary = self.read(writer)[-1]
        self.assertEqual(summary['oversize_dropped'], 1)
        self.assertEqual(summary['unknown_topic_dropped'], 1)

    def test_disk_error_never_claims_success(self):
        with patch.object(TelemetryWriter, '_write_record', side_effect=OSError('disk full')):
            writer = self.writer()
            self.assertFalse(writer.close())
        self.assertIn('disk full', writer.error)
        self.assertFalse(writer.completed)

    def test_size_limit_keeps_one_bounded_file_with_summary(self):
        writer = self.writer(max_bytes=70000, message_max_bytes=4096, queue_capacity=32)
        for _ in range(20):
            writer.submit('/runtime/fault', {'data': 'x' * 3000})
        self.assertTrue(writer.close())
        self.assertTrue(writer.limit_reached)
        self.assertLessEqual(writer.path.stat().st_size, 70000)
        self.assertEqual(len(list(Path(self.temp.name).glob('*.jsonl'))), 1)
        self.assertGreater(self.read(writer)[-1]['size_limit_dropped'], 0)

    def test_segment_bounds_are_observations_not_invented_motion_times(self):
        writer = self.writer()
        for direction in ('forward', 'forward', 'stop'):
            writer.submit('/debug/encoder/state', dict(session_id='s1', feedback_valid=True,
                applied_direction=direction, applied_speed_percent=50 if direction == 'forward' else 0))
        self.assertTrue(writer.close())
        segments = [r for r in self.read(writer) if r['event'] == 'segment']
        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0]['unknown_start'])
        self.assertFalse(segments[0]['unknown_end'])
        self.assertEqual(segments[0]['end_reason'], 'observed_stop')

    def test_gap_invalid_feedback_and_session_change_have_unknown_boundaries(self):
        writer = self.writer()
        for tick, session, valid in ((0, 's1', True), (1000000000, 's1', True),
                                     (1100000000, 's1', False), (1200000000, 's2', True)):
            writer.submit('/debug/encoder/state', dict(session_id=session, feedback_valid=valid,
                applied_direction='forward', applied_speed_percent=50), monotonic_ns=tick)
        self.assertTrue(writer.close())
        segments = [r for r in self.read(writer) if r['event'] == 'segment']
        self.assertEqual([r['end_reason'] for r in segments],
                         ['feedback_gap', 'changed_or_invalid_feedback', 'recording_ended'])
        self.assertTrue(all(r['unknown_start'] and r['unknown_end'] for r in segments))

    def test_diagnostics_json_string_is_preserved_without_cloud_payloads(self):
        topic = '/debug/encoder/diagnostics'
        writer = TelemetryWriter(self.temp.name, topics={topic: 'std_msgs/msg/String'})
        self.addCleanup(writer.close)
        raw = '{"reason":"RUNNING","right_trim_percent":99.2,"historical":false}'
        self.assertTrue(writer.submit(topic, {'data': raw}))
        self.assertTrue(writer.close())
        sample = next(r for r in self.read(writer) if r['event'] == 'sample')
        self.assertEqual(sample['data']['data'], raw)
        self.assertEqual(json.loads(sample['data']['data'])['right_trim_percent'], 99.2)
