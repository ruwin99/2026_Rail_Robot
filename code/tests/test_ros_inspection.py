"""Inspection/transport contracts with schema-restricted doubles in subprocesses.

These tests exercise callbacks and failure handling, not native ROS/DDS.
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


BOOTSTRAP = r'''
from test_ros_robotics import install_fake_ros
install_fake_ros()
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import asyncio, sys
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
import rail_robot_interfaces.msg as messages
import rail_robot_interfaces.srv as services

# __slots__ catches misspelled fields that SimpleNamespace would silently accept.
registry = {'std_msgs/Header': Header, 'sensor_msgs/Image': Image,
            'builtin_interfaces/Time': Time}
def schema_class(name, source):
    fields = [line.split()[:2] for raw in source.splitlines()
              if (line := raw.split('#')[0].strip()) and '=' not in line]
    def init(self, **values):
        for kind, field in fields:
            if kind.endswith('[]'): value = []
            elif kind == 'string': value = ''
            elif kind == 'bool': value = False
            elif kind.startswith('float'): value = 0.0
            elif kind.startswith(('int', 'uint')): value = 0
            else: value = registry[kind]()
            setattr(self, field, value)
        for field, value in values.items(): setattr(self, field, value)
    return type(name, (), {'__slots__': [field for _, field in fields], '__init__': init})
root = Path.cwd() / 'code/ros2/rail_robot_interfaces'
for path in sorted((root/'msg').glob('*.msg')):
    cls = schema_class(path.stem, path.read_text())
    registry['rail_robot_interfaces/'+path.stem] = cls
    setattr(messages, path.stem, cls)
for path in (root/'srv').glob('*.srv'):
    request, response = path.read_text().split('---')
    setattr(services, path.stem, type(path.stem, (), {
        'Request': schema_class(path.stem+'Request', request),
        'Response': schema_class(path.stem+'Response', response)}))

from node_runtime.data import CameraFrame
from node_runtime.ros.images import encode_image, decode_image, frame_to_message, message_to_frame
from node_runtime.ros.common import stamp_from_seconds, source_monotonic
from node_runtime.ros import inspection_node as module
from node_runtime.ros import upload_node

# Match the rclpy parameter query used by optional dashboard capture output.
module.Node.has_parameter = lambda self, name: name in self.parameters

def pair():
    result = services.CapturePair.Response(success=True)
    for role in ('side','top'):
        frame = CameraFrame(np.zeros((720,1280,3),dtype=np.uint8), 8, 9.9)
        message = frame_to_message(frame, role, 100.0, 10.0)
        message.stream_id = 'camera-session'
        setattr(result, role, message)
    return result

def setup_node():
    worker = Mock()
    worker.failure_message.return_value = None
    worker.wait_until_idle.return_value = True
    worker.shutdown.return_value = None
    with patch.object(module, 'CaptureAnalysisWorker', return_value=worker):
        node = module.InspectionNode(Mock(), Mock(), Mock(), Path('unused-captures'))
    node._pair_client.responder = lambda request: pair()
    return node, worker

def request(identifier='one', phase='initial', sequence=1):
    return services.CaptureInspection.Request(
        request_id=identifier, phase=phase, phase_sequence=sequence, trigger='CAMERA_CAPTURE')

def capture(node, req):
    return asyncio.run(node._capture(req, services.CaptureInspection.Response()))
'''


class RosInspectionTests(unittest.TestCase):
    def test_missing_capture_bucket_disables_only_optional_capture_output(self):
        self.contract('''
            import tempfile
            from node_runtime.live_dashboard import LiveDashboardClient
            from node_runtime import live_capture
            worker=Mock()
            worker.failure_message.return_value=None
            worker.wait_until_idle.return_value=True
            worker.shutdown.return_value=None
            client=Mock()
            client.require_capture_storage.side_effect=ValueError('storageBucket missing')
            original=module.Node.declare_parameter
            def declare(self,name,value=None,descriptor=None):
                return original(self,name,True if name=='dashboard_live' else value,descriptor)
            with tempfile.TemporaryDirectory() as root, \\
                    patch.object(module.Node,'declare_parameter',declare), \\
                    patch.object(module,'CaptureAnalysisWorker',return_value=worker), \\
                    patch.object(LiveDashboardClient,'from_environment',return_value=client), \\
                    patch.object(live_capture,'LiveCaptureOutbox') as outbox:
                workbook=Mock(path=str(Path(root)/'report.xlsx'))
                node=module.InspectionNode(Mock(),Mock(),workbook,Path(root))
                assert node._live_captures is None and node.readiness()[0]
                client.require_capture_storage.assert_called_once()
                outbox.assert_not_called()
                worker.start.assert_called_once()
                node.destroy_node()
        ''')

    def test_missing_live_config_does_not_stop_local_inspection(self):
        self.contract('''
            import tempfile
            from node_runtime.live_dashboard import LiveDashboardClient
            worker=Mock()
            worker.failure_message.return_value=None
            worker.wait_until_idle.return_value=True
            worker.shutdown.return_value=None
            original=module.Node.declare_parameter
            def declare(self,name,value=None,descriptor=None):
                return original(self,name,True if name=='dashboard_live' else value,descriptor)
            with tempfile.TemporaryDirectory() as root, \\
                    patch.object(module.Node,'declare_parameter',declare), \\
                    patch.object(module,'CaptureAnalysisWorker',return_value=worker), \\
                    patch.object(LiveDashboardClient,'from_environment',side_effect=FileNotFoundError('secret')):
                workbook=Mock(path=str(Path(root)/'report.xlsx'))
                node=module.InspectionNode(Mock(),Mock(),workbook,Path(root))
                assert node._live_captures is None
                assert node.readiness()[0]
                node.destroy_node()
        ''')

    def contract(self, source):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join((str(Path(__file__).resolve().parent),
                                                    environment.get('PYTHONPATH', '')))
        result = subprocess.run([sys.executable, '-B', '-c', BOOTSTRAP + '\n' + textwrap.dedent(source)],
                                cwd=Path(__file__).resolve().parents[2], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_image_stride_roundtrip_and_source_age(self):
        self.contract('''
            data=np.arange(18,dtype=np.uint8).reshape(2,3,3)
            message=encode_image(data)
            assert np.array_equal(decode_image(message),data)
            message.step=11
            message.data=data[0].tobytes()+b'xx'+data[1].tobytes()+b'yy'
            assert np.array_equal(decode_image(message),data)
            message.data=message.data[:-1]
            try: decode_image(message)
            except ValueError: pass
            else: raise AssertionError('truncated image accepted')
            assert abs(message_to_frame(pair().side,100.0,10.0).read_completed_at-9.9)<1e-6
            for stamp in (99.0,100.1):
                try: source_monotonic(stamp_from_seconds(stamp),100.0,10.0,0.2)
                except ValueError: pass
                else: raise AssertionError('stale/future observation accepted')
        ''')

    def test_capture_accepts_only_after_queue_and_is_idempotent(self):
        self.contract('''
            node, worker=setup_node()
            def queue(*args,**kwargs):
                assert node._counts['initial']==0 and not node._accepted
                assert args[0].shape==args[1].shape==(720,1280,3)
            with patch.object(module,'queue_dual_capture_for_analysis',side_effect=queue) as enqueue:
                assert capture(node,request()).accepted
                assert capture(node,request()).accepted
                assert enqueue.call_count==1 and len(node._pair_client.requests)==1
                assert node._counts['initial']==1
            assert not node.publishers['runtime/fault'].messages
        ''')

    def test_failed_queue_cannot_accept_or_advance(self):
        self.contract('''
            node,worker=setup_node()
            with patch.object(module,'queue_dual_capture_for_analysis',side_effect=RuntimeError('disk full')):
                reply=capture(node,request())
            assert not reply.accepted and 'disk full' in reply.error
            assert node._counts['initial']==0 and not node._accepted
            assert len(node.publishers['runtime/fault'].messages)==1
        ''')

    def test_capture_never_creates_or_destroys_timers(self):
        # 2026-10-05 실기: 요청마다 만든 시간 초과 타이머를 파괴하는 순간 다른 executor 스레드가
        # 그 타이머를 꺼내다 InvalidHandle로 spin_once가 죽어 임무 전체가 종료됐다.
        self.contract('''
            node,worker=setup_node()
            timers=list(node.timers)
            node.create_timer=Mock(side_effect=AssertionError('capture created a timer'))
            node.destroy_timer=Mock(side_effect=AssertionError('capture destroyed a timer'))
            with patch.object(module,'queue_dual_capture_for_analysis'):
                assert capture(node,request()).accepted
                assert capture(node,request('two',sequence=2)).accepted
            assert list(node.timers)==timers
            assert node._pending_pair is None and node._pending_pair_deadline is None
            node._pair_client.responder=None
            async def exercise():
                task=asyncio.create_task(node._capture(request('three',sequence=3),services.CaptureInspection.Response()))
                await asyncio.sleep(0)
                node._check_worker()
                assert not node._pair_client.futures[-1].cancelled()
                with patch.object(module.time,'monotonic',return_value=1e12):
                    node._check_worker()
                reply=await task
                assert not reply.accepted and 'timed out' in reply.error
            asyncio.run(exercise())
            assert list(node.timers)==timers and not node._busy
        ''')

    def test_slow_pair_delivery_keeps_fresh_stopped_originals(self):
        # 2026-10-05 실기: 분석 작업자가 엔진을 올리는 동안 응답 처리가 늦어져 요청 뒤 새로 읽은
        # 원본이 실시간 기준 0.2초에 걸려 stale fault가 났다. 요청 이후 원본은 받아야 한다.
        self.contract('''
            node,worker=setup_node()
            node._ros_seconds=100.0          # pair() 원본 시각은 99.9초(요청 0.1초 전)
            def slow(request):
                node._ros_seconds=101.5      # 응답 도착이 1.5초 늦음
                return pair()
            node._pair_client.responder=slow
            with patch.object(module,'queue_dual_capture_for_analysis') as enqueue:
                reply=capture(node,request())
            assert reply.accepted, reply.error
            enqueue.assert_called_once()
            # 요청 시작보다 0.2초 넘게 오래된 원본은 여전히 거절한다.
            stale,_=setup_node()
            stale._ros_seconds=100.5         # 원본 99.9초는 요청보다 0.6초 전
            with patch.object(module,'queue_dual_capture_for_analysis') as enqueue:
                reply=capture(stale,request())
            assert not reply.accepted and 'stale' in reply.error
            enqueue.assert_not_called()
        ''')

    def test_camera_timeout_cancels_without_acceptance(self):
        self.contract('''
            node,worker=setup_node()
            node._pair_client.responder=None
            async def exercise():
                task=asyncio.create_task(node._capture(request(),services.CaptureInspection.Response()))
                await asyncio.sleep(0)
                with patch.object(module.time,'monotonic',return_value=1e12):
                    node.timers[-1].callback()
                reply=await task
                assert not reply.accepted and 'timed out' in reply.error
            asyncio.run(exercise())
            assert not node._accepted and not node._busy
        ''')

    def test_finalize_requires_complete_matrix_and_publishes_once(self):
        self.contract('''
            node,worker=setup_node()
            node._counts={'initial':module.AUTOMATIC_RAIL_SECTION_TARGET,'rescan':module.AUTOMATIC_RAIL_SECTION_TARGET}
            manifest=Path('report.json')
            with patch.object(module,'finalize_dashboard_run',return_value=manifest) as finish, patch.object(module,'validate_db_test_dashboard_manifest') as validate:
                for _ in range(2):
                    reply=node._drain(services.DrainInspection.Request(finalize=True),services.DrainInspection.Response())
                    assert reply.success
                finish.assert_called_once()
                validate.assert_called_once_with(manifest)
                assert len(node.publishers['inspection/finalized'].messages)==1
            incomplete,_=setup_node()
            with patch.object(module,'finalize_dashboard_run') as finish:
                reply=incomplete._drain(services.DrainInspection.Request(finalize=True),services.DrainInspection.Response())
                assert not reply.success
                finish.assert_not_called()
        ''')

    def test_shutdown_cancels_pending_capture_without_queueing(self):
        self.contract('''
            node,worker=setup_node()
            node._pair_client.responder=None
            async def exercise():
                task=asyncio.create_task(node._capture(request(),services.CaptureInspection.Response()))
                await asyncio.sleep(0)
                node.destroy_node()
                reply=await task
                assert not reply.accepted and not node._accepted
                assert node._pair_client.futures[0].cancelled()
                node.destroy_node()
            with patch.object(module,'finalize_dashboard_run'), patch.object(module,'queue_dual_capture_for_analysis') as enqueue:
                asyncio.run(exercise())
                enqueue.assert_not_called()
            worker.shutdown.assert_called_once()
        ''')

    def test_failed_report_waits_for_worker_and_never_publishes_for_upload(self):
        self.contract('''
            node,worker=setup_node()
            node._on_fault(NS(data='camera disconnected'))
            with patch.object(module,'finalize_dashboard_run') as finish:
                worker.wait_until_idle.return_value=False
                node._check_worker()
                finish.assert_not_called()
                worker.wait_until_idle.return_value=True
                node._check_worker()
                node._check_worker()
                finish.assert_called_once()
                assert finish.call_args.kwargs['status']=='failed'
                assert not node.publishers['inspection/finalized'].messages
        ''')

    def test_fault_during_drain_cannot_finalize_complete(self):
        self.contract('''
            node,worker=setup_node()
            node._counts={'initial':module.AUTOMATIC_RAIL_SECTION_TARGET,'rescan':module.AUTOMATIC_RAIL_SECTION_TARGET}
            def drain(timeout):
                node._on_fault(NS(data='camera disconnected during drain'))
                return True
            worker.wait_until_idle.side_effect=drain
            with patch.object(module,'finalize_dashboard_run') as finish:
                reply=node._drain(services.DrainInspection.Request(finalize=True),services.DrainInspection.Response())
                assert not reply.success and 'camera disconnected' in reply.error
                finish.assert_not_called()
                assert not node.publishers['inspection/finalized'].messages
        ''')

    def test_upload_deduplicates_finalized_manifest_and_runs_outside_callback(self):
        self.contract('''
            from concurrent.futures import Future
            future=Future()
            pool=Mock()
            pool.submit.return_value=future
            uploader=Mock(return_value='uploaded')
            with patch.object(upload_node,'ThreadPoolExecutor',return_value=pool):
                node=upload_node.UploadNode(uploader)
            node._enqueue(NS(data='report.json'))
            node._enqueue(NS(data='report.json'))
            pool.submit.assert_called_once()
            uploader.assert_not_called()
            future.set_result('uploaded')
            node._poll()
            assert node.publishers['inspection/upload_status'].messages[0].data.startswith('uploaded: ')
            with patch.object(upload_node,'validate_db_test_dashboard_manifest') as validate:
                assert node._upload(Path('report.json'))=='uploaded'
                validate.assert_called_once()
            node.destroy_node()
            pool.shutdown.assert_called_once_with(wait=False,cancel_futures=True)
        ''')


if __name__ == '__main__':
    unittest.main()
