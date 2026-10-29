"""Actual recorder callbacks with ROS doubles; no DDS/Jetson access."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson_code'))


class FakeNode:
    fail_subscription = False
    def __init__(self, name, **kwargs):
        self.name, self.subscriptions, self.timers = name, [], []
    def declare_parameter(self, name, value):
        return NS(value=value)
    def resolve_topic_name(self, topic):
        return '/bench/' + topic
    def get_namespace(self):
        return '/bench'
    def create_subscription(self, msg_type, topic, callback, qos):
        if self.fail_subscription:
            raise RuntimeError('subscription unavailable')
        self.subscriptions.append((msg_type, topic, callback, qos))
        return object()
    def destroy_node(self):
        self.destroyed = True
        return True


def load_record():
    modules = {name: ModuleType(name) for name in ('rclpy', 'rclpy.node', 'rclpy.qos',
        'rosidl_runtime_py', 'rosidl_runtime_py.utilities', 'rosidl_runtime_py.convert')}
    modules['rclpy.node'].Node = FakeNode
    qos = modules['rclpy.qos']
    qos.QoSProfile = lambda **kwargs: NS(**kwargs)
    qos.ReliabilityPolicy = NS(BEST_EFFORT='best_effort')
    qos.DurabilityPolicy = NS(VOLATILE='volatile')
    qos.HistoryPolicy = NS(KEEP_LAST='keep_last')
    modules['rosidl_runtime_py.utilities'].get_message = lambda name: name
    modules['rosidl_runtime_py.convert'].message_to_ordereddict = lambda message: vars(message)
    path = Path(__file__).resolve().parents[1] / 'jetson_code/node_runtime/ros/record_node.py'
    spec = importlib.util.spec_from_file_location('node_runtime.ros.isolated_record', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class RecordNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, RAIL_ROBOT_HOME=self.temp.name)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.module = load_record()

    def node(self):
        node = self.module.RecordNode()
        self.addCleanup(node.destroy_node)
        return node

    def test_relative_topic_types_qos_and_no_publishers(self):
        node = self.node()
        self.assertEqual(node.name, 'rail_record')
        self.assertEqual(len(node.subscriptions), len(self.module.TOPICS))
        for msg_type, topic, _, qos in node.subscriptions:
            self.assertFalse(topic.startswith('/'))
            self.assertEqual(msg_type, self.module.TOPICS[topic])
            self.assertEqual((qos.depth, qos.reliability, qos.durability), (50, 'best_effort', 'volatile'))
            self.assertNotIn('image', topic)
            self.assertNotIn('webrtc', topic)
            self.assertNotIn('firebase', topic)

    def test_runtime_fault_keeps_recording_and_namespace_stamp_preserved(self):
        node = self.node()
        callbacks = {topic: callback for _, topic, callback, _ in node.subscriptions}
        callbacks['runtime/fault'](NS(data='DRIVE_STALL'))
        self.assertFalse(node.shutdown_requested())
        self.assertFalse(node._failure)
        callbacks['debug/encoder/state'](NS(header={'stamp': {'sec': 3, 'nanosec': 7}},
            session_id='abc', feedback_valid=True, applied_direction='stop', applied_speed_percent=0))
        node.destroy_node()
        rows = [json.loads(line) for line in node.writer.path.read_text(encoding='utf-8').splitlines()]
        samples = [row for row in rows if row['event'] == 'sample']
        self.assertEqual(samples[0]['topic'], '/bench/runtime/fault')
        self.assertEqual(samples[1]['data']['header']['stamp']['nanosec'], 7)
        self.assertEqual(rows[0]['namespace'], '/bench')
        self.assertEqual(rows[-1]['samples_written'], 2)

    def test_construction_failure_removes_new_partial_file(self):
        with patch.object(FakeNode, 'fail_subscription', True):
            with self.assertRaisesRegex(RuntimeError, 'subscription unavailable'):
                self.module.RecordNode()
        self.assertEqual(list(Path(self.temp.name).rglob('*.jsonl')), [])

    def test_prepare_shutdown_rejects_late_callbacks_and_finalizes_empty_run(self):
        node = self.node()
        node.prepare_shutdown()
        node._receive('/bench/runtime/fault', NS(data='late'))
        node.destroy_node()
        rows = [json.loads(line) for line in node.writer.path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(rows[-1]['samples_written'], 0)
