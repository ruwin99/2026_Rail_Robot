"""Exercise real ROS adapter with isolated ROS doubles, never opening hardware."""
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson_code'))


class SensorNodeTests(unittest.TestCase):
    def test_default_rr2_node_led_status_and_worker_owned_shutdown(self):
        script = r'''
import json, threading, time
from types import SimpleNamespace
from test_ros_robotics import install_fake_ros, FakeNode, FakeUart
install_fake_ros()
FakeNode.count_publishers = lambda self, topic: 0
FakeNode.resolve_service_name = lambda self, name: '/' + name
FakeNode.get_service_names_and_types = lambda self: []
from node_runtime.rr2_protocol import Rr2Frame, decode_rr2, encode_rr2
from node_runtime.ros.sensor_debug_node import SensorDebugNode
threads, frames, serials = set(), [], []
class Serial(FakeUart):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        threads.add(threading.get_ident())
        serials.append(self)
    @property
    def in_waiting(self):
        threads.add(threading.get_ident())
        return super().in_waiting
    def read(self, size):
        threads.add(threading.get_ident())
        return super().read(size)
    def write(self, data):
        threads.add(threading.get_ident())
        frame = decode_rr2(data)
        frames.append(frame)
        if frame.kind == 'HELLO':
            self.pending.extend(encode_rr2(Rr2Frame('READY', frame.session, None,
                ('2','SENSOR_DEBUG'))))
            self.pending.extend(encode_rr2(Rr2Frame('STATUS', frame.session, 1,
                ('ULTRASONIC','100','180','1','1','0','1','0','0','0','ECHO'))))
        elif frame.kind in ('CMD','PING'):
            opcode = frame.payload[0] if frame.kind == 'CMD' else 'PING'
            self.pending.extend(encode_rr2(Rr2Frame('ACK', frame.session, frame.sequence,
                (opcode,'ACCEPTED','0'))))
        return super().write(data)
    def close(self):
        threads.add(threading.get_ident())
        super().close()
node = SensorDebugNode(uart_protocol='v2', uart_factory=Serial,
    exclusive_device=lambda uart: threads.add(threading.get_ident()))
for _ in range(50):
    node._tick()
    if node.session.feedback is not None: break
    time.sleep(.005)
assert node.session.ready and node.session.feedback['distance_mm'] == 180
state = json.loads(node.publishers['debug/ultrasonic/state'].messages[-1].data)
assert state['feedback_valid'] and state['detected']
prior = len(frames)
node._on_led(SimpleNamespace(data='green'))
assert len(frames) == prior
node._tick()
for _ in range(50):
    if any(f.kind == 'CMD' and f.payload == ('LED','breathe','0','255','0') for f in frames): break
    node._tick()
    time.sleep(.005)
assert any(f.kind == 'CMD' and f.payload == ('LED','breathe','0','255','0') for f in frames)
node.destroy_node()
assert serials[0].closed and len(threads) == 1
assert any(f.kind == 'CMD' and f.payload == ('STOP',) for f in frames)
assert not any(b'DBG1' in data for data in serials[0].writes)
'''
        tests = Path(__file__).resolve().parent
        source = f'import sys; sys.path[:0] = {[str(tests), str(tests.parent / "jetson_code")]!r}\n' + script
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rr2_sensor_session_keeps_lease_with_stop_without_restarting_led_effect(self):
        from concurrent.futures import Future
        from node_runtime.sensor_debug import Rr2SensorDebugSession
        from node_runtime.serial_worker import CommandResult, SerialResult

        class Worker:
            ready, startup_error, session_id = True, "", "a" * 32
            def __init__(self):
                self.sent = []
                self.results = []
            def submit(self, opcode, args=(), **kwargs):
                self.sent.append((opcode, args, kwargs))
                future = Future()
                future.set_result(CommandResult(True, self.session_id, len(self.sent)))
                return future
            def drain_results(self, maximum):
                result, self.results = self.results[:maximum], self.results[maximum:]
                return result
            def latch_fault(self):
                return self.submit("STOP")

        worker, now = Worker(), [10.0]
        session = Rr2SensorDebugSession(worker, clock=lambda: now[0])
        worker.results.append(SerialResult("STATUS", 1,
            ("ULTRASONIC", "10000", "180", "1", "1", "0", "1", "0", "0", "0", "ECHO"),
            worker.session_id))
        session.tick()
        self.assertEqual(session.feedback["distance_mm"], 180)
        session.command_led("green")
        session.tick()
        self.assertIn(("LED", ("breathe", "0", "255", "0"), {"coalesce_key": "debug_led"}), worker.sent)
        now[0] += .21
        session.tick()
        self.assertEqual(worker.sent[-1][0], "STOP")
        now[0] += .21
        session.tick()
        self.assertEqual(sum(opcode == "LED" for opcode, _, _ in worker.sent), 1)
        self.assertEqual(sum(opcode == "STOP" for opcode, _, _ in worker.sent), 3)
        session.close()
        self.assertTrue(session.closed)

    def test_callbacks_and_feedback_shutdown_and_uart_conflict(self):
        script = r'''
import json
from types import SimpleNamespace
from unittest.mock import patch
from test_ros_robotics import install_fake_ros, FakeNode, FakeUart
install_fake_ros()
FakeNode.count_publishers = lambda self, topic: 0
FakeNode.resolve_service_name = lambda self, name: '/' + name
FakeNode.get_service_names_and_types = lambda self: []
from node_runtime.ros.sensor_debug_node import SensorDebugNode
now = [10.0]
uart = FakeUart()
original_init, original_timer = FakeNode.__init__, FakeNode.create_timer
def initialize_ros_clock(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._clock = SimpleNamespace(handle=object())
    self.expected_ros_clock = self._clock
def create_ros_timer(self, *args, **kwargs):
    assert self._clock is self.expected_ros_clock
    assert hasattr(self._clock, 'handle')
    return original_timer(self, *args, **kwargs)
with patch.object(FakeNode, '__init__', initialize_ros_clock), \
     patch.object(FakeNode, 'create_timer', create_ros_timer):
    node = SensorDebugNode(uart_protocol='legacy', uart_factory=lambda **kwargs: uart, exclusive_device=lambda u: None,
                           clock=lambda: now[0])
assert node._clock is node.expected_ros_clock
session = node.session.session_id
uart.pending.extend(f'DBG1_READY {session} 8 1600 106 1200 3000 601 601\r\n'.encode())
uart.pending.extend(f'DBG1_SENSOR {session} 0 10000 180 1 1 0 1 0 0 0 ECHO\r\n'.encode())
node._tick()
state = json.loads(node.publishers['debug/ultrasonic/state'].messages[-1].data)
assert state['distance_mm'] == 180 and state['detected'] and not state['held'] and state['feedback_valid']
led = json.loads(node.publishers['debug/led/state'].messages[-1].data)
assert led['feedback_valid'] and led['obstacle_override'] is False
assert set(node.subscriptions) == {'debug/led/set'}
node._on_led(SimpleNamespace(data='green'))
now[0] += .21
node._tick()
assert uart.writes[-1].endswith(b'breathe 0 255 0\r\n')
node._on_led(SimpleNamespace(data='START'))
assert not node._failed
now[0] += .61
node._tick()
assert node._failed
assert not json.loads(node.publishers['debug/ultrasonic/state'].messages[-1].data)['feedback_valid']
node.destroy_node()
assert uart.closed
FakeNode.count_publishers = lambda self, topic: 1
try:
    SensorDebugNode(uart_factory=lambda **kw: (_ for _ in ()).throw(AssertionError('opened UART')))
except RuntimeError:
    pass
else:
    raise AssertionError('mission conflict accepted')
'''
        # Raw outer string keeps source escapes; Python child needs single escape in literals.
        script = script.replace('\\\\r\\\\n', '\\r\\n')
        tests = Path(__file__).resolve().parent
        source = f'import sys; sys.path[:0] = {[str(tests), str(tests.parent / "jetson_code")]!r}\n' + script
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
