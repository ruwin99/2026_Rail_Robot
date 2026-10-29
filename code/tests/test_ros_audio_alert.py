"""ROS adapter tests run with isolated doubles; no installed ROS or aplay needed."""
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class RosAudioTests(unittest.TestCase):
    def test_topics_readiness_errors_and_fault_cleanup(self):
        source = r'''
from test_ros_robotics import install_fake_ros
install_fake_ros()
import std_msgs.msg
from types import SimpleNamespace
std_msgs.msg = __import__('sys').modules['std_msgs.msg']
std_msgs.msg.Bool = SimpleNamespace
import json, uuid
from node_runtime.ros.audio_alert_node import AudioAlertNode
from unittest.mock import Mock
backend = Mock()
backend.readiness.return_value = (True, 'ready')
backend.poll_results.return_value = []
backend.status.return_value = {'pending':0}
node = AudioAlertNode(backend_factory=lambda **kwargs: backend)
assert set(node.subscriptions) == {'tts/request','tts/debug','tts/obstacle_state','runtime/fault'}
assert node.subscriptions['tts/obstacle_state'].qos.durability == 'transient_local'
assert 'runtime/fault' not in node.publishers
request = {'request_id':str(uuid.uuid4()), 'cue':'mission_started','expires_at':110.0}
node.subscriptions['tts/request'].callback(SimpleNamespace(data=json.dumps(request)))
backend.submit.assert_called_once_with(request, 100.0)
backend.submit.reset_mock()
node.subscriptions['tts/debug'].callback(SimpleNamespace(data='mission_started'))
debug_request, debug_time = backend.submit.call_args.args
assert debug_request['cue'] == 'mission_started'
assert str(uuid.UUID(debug_request['request_id'])) == debug_request['request_id']
assert 100.0 < debug_request['expires_at'] <= 140.0
assert debug_time == 100.0
backend.readiness.assert_not_called()
assert set(node.publishers) == {'tts/result', 'tts/status'}
backend.submit.reset_mock()
node.subscriptions['tts/debug'].callback(SimpleNamespace(data='mission_completed'))
assert backend.submit.call_args.args[0]['cue'] == 'mission_completed'
backend.submit.reset_mock()
for invalid in ('start', 'done', 'obstacle_detected', '', 'x' * 1025):
    node.subscriptions['tts/debug'].callback(SimpleNamespace(data=invalid))
backend.submit.assert_not_called()
assert node.log_messages
result = dict(request_id=request['request_id'],cue='mission_started',success=True,error='')
backend.poll_results.return_value=[result]
node._tick()
assert json.loads(node.publishers['tts/result'].messages[-1].data)==result
node.subscriptions['tts/obstacle_state'].callback(SimpleNamespace(data=True))
backend.set_obstacle.assert_called_once_with(True)
response=node.services['tts/ready'].callback(None,SimpleNamespace())
assert response.success
node.subscriptions['tts/request'].callback(SimpleNamespace(data='not json'))
assert node.log_messages
node.subscriptions['runtime/fault'].callback(SimpleNamespace(data='camera fault'))
backend.cancel_all.assert_called_once()
node.prepare_shutdown()
node.destroy_node()
backend.close.assert_called_once()
'''
        result = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(source)],
                                cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
