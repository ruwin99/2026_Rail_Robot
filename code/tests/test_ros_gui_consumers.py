"""GUI fallback and measurement contracts, without Firebase, ROS or hardware."""
import os
import subprocess
import sys
import textwrap
import unittest

from test_ros_vision import JETSON_CODE, ROS_DOUBLES


BOOTSTRAP = r'''
import cv2, json
from concurrent.futures import Future
from sensor_msgs.msg import CompressedImage
from node_runtime.ros import firebase_snapshot_node as fallback, gui_probe_node as probe
from node_runtime.ros.common import stamp_from_seconds
from firebase_uploader import FirebaseLivePreviewPublisher

def jpeg(stamp=100.0, value=80, shape=(720,1280,3)):
    message=CompressedImage()
    message.header=Header()
    message.header.stamp=stamp_from_seconds(stamp)
    message.header.frame_id='rail_robot_gui'
    message.format='bgr8; jpeg compressed bgr8'
    message.data=cv2.imencode('.jpg',np.full(shape,value,np.uint8))[1].tobytes()
    return message
'''


class GuiConsumerTests(unittest.TestCase):
    def run_contract(self, source):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join((str(JETSON_CODE), env.get('PYTHONPATH','')))
        result = subprocess.run([sys.executable,'-B','-c',ROS_DOUBLES+BOOTSTRAP+textwrap.dedent(source)],
                                env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_consumers_preserve_ros_clock_for_timer_creation_and_callbacks(self):
        self.run_contract('''
            original_init, original_timer = Node.__init__, Node.create_timer
            def initialize_ros_clock(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self._clock = NS(now=lambda:NS(nanoseconds=int(self.now*1e9)))
                self.expected_ros_clock = self._clock
            def create_ros_timer(self, *args, **kwargs):
                assert self._clock is self.expected_ros_clock
                assert self.get_clock().now().nanoseconds == int(self.now*1e9)
                return original_timer(self, *args, **kwargs)
            with patch.object(Node, '__init__', initialize_ros_clock), \
                 patch.object(Node, 'get_clock', lambda self:self._clock), \
                 patch.object(Node, 'create_timer', create_ros_timer):
                now = [0.0]
                observer = probe.GuiProbeNode(clock=lambda:now[0], parameters={'duration_seconds':1.0})
                uploader = Mock(poll=Mock(return_value=None),submit=Mock(return_value=True))
                snapshot = fallback.FirebaseSnapshotNode(publisher_factory=lambda:uploader,clock=lambda:now[0])
                observer._receive(jpeg())
                snapshot._receive(jpeg()); snapshot._tick()
                uploader.submit.assert_called_once()
                now[0] = 1.0
                observer._tick()
                assert observer.summary['received'] == 1
                for node in (observer,snapshot):
                    assert node._clock is node.expected_ros_clock
                    node.destroy_node()
        ''')

    def test_diagnostic_frame_trace_does_not_replace_or_enter_jpeg_consumers(self):
        self.run_contract('''
            probe_node=probe.GuiProbeNode(parameters={'duration_seconds':1.0})
            snapshot_node=fallback.FirebaseSnapshotNode(publisher_factory=lambda:None)
            probe_topics={topic for topic,_,_ in probe_node.subscriptions}
            snapshot_topics={topic for topic,_,_ in snapshot_node.subscriptions}
            assert probe_topics=={'/rail_robot/gui/image/compressed','/rail_robot/gui/status'}
            assert snapshot_topics=={'/rail_robot/gui/image/compressed'}
            assert '/rail_robot/debug/frame_trace' not in probe_topics|snapshot_topics
            image=jpeg(100.0)
            probe_node._receive(image)
            assert probe_node._received==1
            assert snapshot_node._latest is None  # unavailable uploader keeps the fallback disabled
            probe_node.destroy_node(); snapshot_node.destroy_node()
        ''')

    def test_fallback_bounds_pending_work_and_uses_latest_after_five_seconds(self):
        self.run_contract('''
            now=[10.0]
            uploader=Mock(poll=Mock(return_value=None),submit=Mock(return_value=True))
            node=fallback.FirebaseSnapshotNode(publisher_factory=lambda:uploader,clock=lambda:now[0])
            node._receive(jpeg(value=10)); node._tick()
            assert uploader.submit.call_count==1
            now[0]=16.0; node.now=106.0
            node._receive(jpeg(106.0,value=20)); node._tick()
            assert uploader.submit.call_count==1  # pending network work blocks extra decoding/submission
            node._receive(jpeg(106.1,value=30))  # future input rejected
            node._receive(jpeg(105.9,value=40))  # old input rejected
            now[0]=16.2; node.now=106.2
            node._receive(jpeg(106.2,value=50))
            uploader.poll.return_value=True
            node._tick()
            assert uploader.submit.call_count==2
            assert np.all(uploader.submit.call_args.args[0]==50)
            assert node._latest is None
            node.prepare_shutdown(); node._receive(jpeg(106.3)); node._tick()
            assert uploader.submit.call_count==2
            node.destroy_node(); uploader.close.assert_called_once()
            assert 'runtime/fault' not in node.publishers
        ''')

    def test_missing_config_errors_and_stale_images_never_become_uploads(self):
        self.run_contract('''
            node=fallback.FirebaseSnapshotNode(publisher_factory=lambda:None)
            node._receive(jpeg()); node._tick(); assert node._latest is None
            node.destroy_node()
            uploader=Mock(poll=Mock(side_effect=RuntimeError('sensitive-token')),submit=Mock(return_value=True))
            node=fallback.FirebaseSnapshotNode(publisher_factory=lambda:uploader,clock=lambda:10.0)
            node._receive(jpeg(98.0)); node._tick()
            uploader.submit.assert_not_called()
            node._receive(jpeg(100.0,shape=(240,1280,3))); node._tick()
            uploader.submit.assert_not_called()
            assert all('sensitive-token' not in m.data for m in node.publishers['gui/firebase_snapshot/status'].messages)
            assert 'runtime/fault' not in node.publishers
            node.destroy_node()
        ''')

    def test_existing_publisher_poll_reports_success_and_failure_once(self):
        self.run_contract('''
            publisher=object.__new__(FirebaseLivePreviewPublisher)
            future=Future(); publisher._pending=future
            assert publisher.poll() is None
            future.set_result(None)
            assert publisher.poll() is True and publisher.poll() is None
            future=Future(); publisher._pending=future
            future.set_exception(RuntimeError('upload failed'))
            try:publisher.poll()
            except RuntimeError:pass
            else:raise AssertionError('Failed upload hidden')
            assert publisher._pending is None
        ''')

    def test_probe_measures_fps_composition_delay_and_source_progress(self):
        self.run_contract('''
            now=[0.0]
            node=probe.GuiProbeNode(clock=lambda:now[0],parameters={'duration_seconds':2.0})
            for index in range(24):
                now[0]=index/12.0; node.now=100.0+now[0]
                node._receive(jpeg(node.now-0.02))
                source=dict(source_age_ms=120.0,stream_id='camera',sequence=index+1,fresh=True,state='raw')
                node._receive_status(NS(data=json.dumps(dict(schema_version=1,state='published',image_stamp_seconds=node.now-0.02,
                    sequence=index+1,frame_id='rail_robot_gui',sources={'side':source,'front':source}))))
            now[0]=2.0; node._tick()
            summary=node.summary
            assert summary['received']==24 and summary['invalid_messages']==0
            assert summary['first_image_shape']==[720,1280,3]
            assert summary['decoded_checks']==2
            assert abs(summary['full_window_fps']-12)<1e-6
            assert abs(summary['composition_to_receive_ms_p95']-20)<1e-5
            assert summary['source_sequence_advances']==dict(side=23,front=23)
            assert summary['missing_or_stale_source_observations']==0
            assert node.shutdown_requested()
        ''')

    def test_snapshot_and_probe_accept_vertical_and_legacy_camera_jpeg(self):
        self.run_contract('''
            for shape in ((720,640,3),(360,1280,3),(720,1280,3)):
                uploader=Mock(poll=Mock(return_value=None),submit=Mock(return_value=True))
                node=fallback.FirebaseSnapshotNode(publisher_factory=lambda:uploader,clock=lambda:10.0)
                node._receive(jpeg(100.0,shape=shape)); node._tick()
                assert uploader.submit.call_count==1
                assert uploader.submit.call_args.args[0].shape==shape
                node.destroy_node()
                node=probe.GuiProbeNode(clock=lambda:0.0,parameters={'duration_seconds':2.0})
                node._receive(jpeg(100.0,shape=shape))
                assert node._first_image_shape==list(shape)
                node.destroy_node()
        ''')

    def test_probe_rejects_delayed_and_reordered_status_and_does_not_count_backward_sources(self):
        self.run_contract('''
            now=[0.0]
            node=probe.GuiProbeNode(clock=lambda:now[0],parameters={'duration_seconds':2.0})
            def status(stamp,seq,source_seq):
                return NS(data=json.dumps(dict(schema_version=1,state='published',image_stamp_seconds=stamp,
                    sequence=seq,frame_id='rail_robot_gui',sources={role:dict(source_age_ms=20.0,stream_id='camera',sequence=source_seq) for role in ('side','front')})))
            node._receive_status(status(98.0,1,10))
            node._receive_status(status(101.0,2,10))
            assert node._status_count==0
            node._receive_status(status(99.9,3,10))
            assert node._source_advances==dict(side=0,front=0)
            node._receive_status(status(99.8,2,11))
            node.now=100.2; now[0]=0.2
            node._receive_status(status(100.1,4,9))
            assert node._source_advances==dict(side=0,front=0)
            node.now=100.3; now[0]=0.3
            node._receive_status(status(100.2,5,11))
            assert node._source_advances==dict(side=1,front=1)
            assert abs(node._sources['side']['age_at_receive_ms']-120.0)<1e-6
            now[0]=2.0; node._tick()
            assert node.summary['invalid_status_messages']==3
            assert node.summary['source_older_frame_observations']==dict(side=1,front=1)
            assert abs(node.summary['time_since_last_status_seconds']-1.7)<1e-6
        ''')

    def test_probe_zero_frames_and_invalid_timestamps_are_visible(self):
        self.run_contract('''
            now=[0.0]
            node=probe.GuiProbeNode(clock=lambda:now[0],parameters={'duration_seconds':1.0})
            node._receive(jpeg(101.0)); node._receive(jpeg(100.0,shape=(360,640,3)))
            now[0]=1.0; node._tick()
            assert node.summary['received']==0 and node.summary['invalid_messages']==2
            assert node.summary['full_window_fps']==0 and node.summary['first_image_shape'] is None
        ''')


if __name__=='__main__':
    unittest.main()
