"""Mission/audio handshake behavior using isolated ROS doubles (no motors)."""
import subprocess
import sys
import types
import unittest
from pathlib import Path


def exercise():
    import asyncio
    import json
    import time
    from types import SimpleNamespace
    from test_ros_robotics import install_fake_ros, FakeUart
    install_fake_ros()
    sys.modules['std_msgs.msg'].Bool = type('Bool', (), {'__init__': lambda self, data=False: setattr(self, 'data', data)})
    from std_msgs.msg import String
    from node_runtime.ros.mission_node import MissionNode
    # This integration test exercises mission/UART/audio wiring, not vision
    # dependencies imported by the bridge's actuator helper module.
    control = types.ModuleType('node_runtime.control')
    control.force_cleaning_pairs_safe_off = lambda *_pairs: None
    control.update_cleaning_actuators = lambda *_args, **_kwargs: None
    sys.modules['node_runtime.control'] = control
    from node_runtime.ros.stm_bridge_node import StmBridgeNode
    from node_runtime.settings import DONE_TRIGGER, RESCAN_DONE_MODE

    def reply(node, request, success=True, **changes):
        data = dict(request_id=request['request_id'], cue=request['cue'], success=success, error='' if success else 'playback failed')
        data.update(changes)
        node._on_audio_result(String(data=json.dumps(data)))

    async def start_case(outcome):
        node = MissionNode()
        node._uart.responder = lambda request: SimpleNamespace(success=True, error='')
        for role, client in node._ready_clients.items():
            def readiness_reply(_request, role=role):
                assert [item.command for item in node._uart.requests] == ['PREPARE'], role
                return SimpleNamespace(success=True, message='ready')
            client.responder = readiness_reply
        task = asyncio.create_task(node._start(None, SimpleNamespace()))
        for _ in range(10):
            await asyncio.sleep(0)
        assert [item.command for item in node._uart.requests] == ['PREPARE'], 'PREPARE must precede readiness/audio and START must wait for WAV completion'
        request = json.loads(node.publishers['tts/request'].messages[-1].data)
        assert request['cue'] == 'mission_started'
        reply(node, request, request_id='late-other-request')
        reply(node, request, cue='mission_completed')
        reply(node, request, success='true')
        await asyncio.sleep(0)
        assert [item.command for item in node._uart.requests] == ['PREPARE']
        if outcome == 'ok':
            # A valid clip may last over the former 5-second readiness limit.
            node._pending_since = time.monotonic() - 6
            node._tick()
            assert [item.command for item in node._uart.requests] == ['PREPARE']
            reply(node, request)
        elif outcome == 'failure':
            reply(node, request, success=False)
        elif outcome == 'timeout':
            node._pending_since = time.monotonic() - 100
            node._tick()
            reply(node, request)
        elif outcome == 'fault':
            node._on_fault(String(data='stm_bridge: disconnected'))
            reply(node, request)
        elif outcome == 'expired':
            node._ros_seconds = request['expires_at'] + 1
            reply(node, request)
        elif outcome == 'obstacle':
            reply(node, request)
            node._on_obstacle(SimpleNamespace(data=True))
        result = await asyncio.wait_for(task, 1)
        commands = [r.command for r in node._uart.requests]
        assert result.success is (outcome == 'ok'), (outcome, result)
        expected = (
            ['PREPARE', 'START'] if outcome == 'ok' else
            ['PREPARE', 'SAFE_OFF', 'CANCEL_PREPARE'] if outcome == 'fault' else
            ['PREPARE', 'CANCEL_PREPARE']
        )
        assert commands == expected, (outcome, commands)
        reply(node, request)
        assert [r.command for r in node._uart.requests].count('START') <= 1

    for outcome in ('ok', 'failure', 'timeout', 'fault', 'expired', 'obstacle'):
        asyncio.run(start_case(outcome))

    async def readiness_rejection():
        node = MissionNode()
        node._uart.responder = lambda request: SimpleNamespace(success=True, error='')
        for role, client in node._ready_clients.items():
            def readiness_reply(_request, role=role):
                assert [item.command for item in node._uart.requests] == ['PREPARE'], role
                return SimpleNamespace(success=(role != 'vision'), message='vision not ready')
            client.responder = readiness_reply
        response = await node._start(None, SimpleNamespace())
        assert not response.success and not node._failed
        assert [item.command for item in node._uart.requests] == ['PREPARE', 'CANCEL_PREPARE']
    asyncio.run(readiness_rejection())

    async def clock_unsynchronized():
        # NTP 동기화 전에는 STM에 아무 명령도 보내지 않고 START를 거절한다.
        import node_runtime.ros.mission_node as mission_module
        original = mission_module.clock_synchronized
        mission_module.clock_synchronized = lambda: False
        try:
            node = MissionNode()
            node._uart.responder = lambda request: SimpleNamespace(success=True, error='')
            response = await node._start(None, SimpleNamespace())
            assert not response.success and 'NTP' in response.message, response.message
            assert node._uart.requests == [] and not node._failed
        finally:
            mission_module.clock_synchronized = original
    asyncio.run(clock_unsynchronized())

    async def integrated_start():
        # 실제 mission→JSON→audio adapter→worker→result→START를 연결한다.
        # 사운드카드 프로세스만 대체하여 ID 형식 등 노드 사이 계약도 검사한다.
        import tempfile
        import wave
        from test_audio_alerts import FakeProcess, until
        from node_runtime.audio_alerts import AudioAlerts
        from node_runtime.ros.audio_alert_node import AudioAlertNode
        children = []
        def spawn(args, **kwargs):
            child = FakeProcess(args, **kwargs)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory() as directory:
            for cue in ('mission_started', 'mission_completed', 'obstacle_detected'):
                with wave.open(str(Path(directory) / (cue + '.wav')), 'wb') as output:
                    output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    output.writeframes(b'\0\0' * 160)
            audio = AudioAlertNode(
                backend_factory=lambda **config: AudioAlerts(**config, popen=spawn, which=lambda _: 'aplay'),
                parameter_overrides=[SimpleNamespace(name='audio_directory', value=directory)])
            mission = MissionNode()
            mission._uart.responder = lambda request: SimpleNamespace(success=True, error='')
            mission._ready_clients['tts'].responder = lambda request: audio._ready(request, SimpleNamespace())
            task = asyncio.create_task(mission._start(None, SimpleNamespace()))
            try:
                await asyncio.sleep(0)
                request = mission.publishers['tts/request'].messages[-1]
                audio._on_request(request)
                await asyncio.to_thread(until, lambda: len(children) > 0)
                assert len(children) == 1, (len(children), audio._backend.status(), audio.log_messages)
                audio._tick()
                assert not audio.publishers['tts/result'].messages
                assert [item.command for item in mission._uart.requests] == ['PREPARE']
                children[0].returncode = 0
                def drain_result():
                    audio._tick()
                    return bool(audio.publishers['tts/result'].messages)
                await asyncio.to_thread(until, drain_result)
                result = audio.publishers['tts/result'].messages[-1]
                # Lost ACK → same request ID → cached result, not a second WAV.
                audio._on_request(request)
                audio._tick()
                mission._on_audio_result(result)
                response = await asyncio.wait_for(task, 1)
                assert response.success and len(children) == 1
                assert [r.command for r in mission._uart.requests] == ['PREPARE', 'START']
            finally:
                if not task.done():
                    mission.prepare_shutdown()
                    await task
                audio.destroy_node()
    asyncio.run(integrated_start())

    node = MissionNode()
    node.state.mark_start_sent()
    node.state.acknowledge_started()
    node.state.mode = RESCAN_DONE_MODE
    transition = node.state.prepare_transition(DONE_TRIGGER)
    node._begin_transition(transition)
    assert not node.publishers['tts/request'].messages
    node._uart.futures[-1].set_result(SimpleNamespace(success=True, error=''))
    assert not node.publishers['tts/request'].messages
    node._drain.futures[-1].set_result(SimpleNamespace(success=True, error=''))
    request = json.loads(node.publishers['tts/request'].messages[-1].data)
    assert request['cue'] == 'mission_completed'
    reply(node, request)
    before = len(node.publishers['tts/request'].messages)
    node._tick()
    assert len(node.publishers['tts/request'].messages) == before

    bridge = StmBridgeNode('fake', uart_factory=FakeUart)
    publisher = bridge.publishers['tts/obstacle_state']
    assert not publisher.messages, 'Do not claim unobserved sensor clear at boot'
    bridge._uart.pending.extend(b'OBSTACLE_HOLD\n')
    bridge._tick()
    assert publisher.messages[-1].data is True
    assert publisher.qos.durability == 'transient_local'
    bridge._uart.pending.extend(b'OBSTACLE_CLEAR\n')
    bridge._tick()
    assert publisher.messages[-1].data is False
    bridge.destroy_node()


class MissionAudioTests(unittest.TestCase):
    def test_audio_gate_and_sensor_state(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'], capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    if '--child' in sys.argv:
        exercise()
    else:
        unittest.main()
