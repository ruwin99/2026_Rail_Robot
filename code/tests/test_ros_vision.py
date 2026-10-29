"""Adapter contract tests using isolated ROS API doubles, not a DDS test."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"

ROS_DOUBLES = r'''
import sys, types
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

def module(name, **values):
    value = types.ModuleType(name)
    value.__dict__.update(values)
    sys.modules[name] = value
    return value

class Time:
    def __init__(self, sec=0, nanosec=0): self.sec, self.nanosec = sec, nanosec
class Header:
    def __init__(self): self.stamp, self.frame_id = Time(), ''
class Image:
    def __init__(self):
        self.header = Header()
        self.height = self.width = self.step = self.is_bigendian = 0
        self.encoding, self.data = '', b''
class CompressedImage:
    __slots__ = ('header', 'format', 'data')
    def __init__(self): self.header, self.format, self.data = Header(), '', b''
class String:
    def __init__(self): self.data = ''
class Float64:
    def __init__(self): self.data = 0.0
class FrameMessage:
    def __init__(self):
        self.header, self.image = Header(), Image()
        self.camera_role, self.stream_id, self.sequence = '', '', 0
        self.capture_complete_monotonic_ns = 0
class Sample:
    def __init__(self):
        self.header, self.mask = Header(), Image()
        self.camera_role = self.stream_id = self.detector = self.detector_method = self.error = ''
        self.sequence, self.rust_grade, self.crack_ratio = 0, -1, 0.0
        self.valid = self.obstacle_detected = False
        self.boxes_xyxy, self.confidences, self.class_ids = [], [], []
class Publisher:
    def __init__(self): self.messages = []
    def publish(self, message): self.messages.append(message)
class Node:
    def __init__(self, name, **kwargs):
        self.name, self.now = name, 100.0
        self.publishers, self.subscriptions, self.timers = {}, [], []
        self._publishers, self._subscriptions = [], []
        self.logger = Mock()
        self.parameters = dict(kwargs.get('parameters', {}))
    def has_parameter(self, name): return name in self.parameters
    def get_parameter(self, name): return NS(value=self.parameters[name])
    def declare_parameter(self, name, value, **kwargs):
        self.parameters.setdefault(name, value)
        return NS(value=self.parameters[name])
    def create_publisher(self, message_type, topic, qos):
        self.publishers[topic] = Publisher()
        self.publishers[topic].message_type = message_type
        self.publishers[topic].qos = qos
        self._publishers.append(self.publishers[topic])
        return self.publishers[topic]
    def create_subscription(self, message_type, topic, callback, qos, **kwargs):
        self.subscriptions.append((topic, callback, kwargs))
        self._subscriptions.append(self.subscriptions[-1])
        return self.subscriptions[-1]
    def create_service(self, service_type, topic, callback): return callback
    def create_timer(self, interval, callback, **kwargs):
        timer = Mock(callback=callback, interval=interval, kwargs=kwargs)
        self.timers.append(timer)
        return timer
    def get_clock(self): return NS(now=lambda: NS(nanoseconds=int(self.now * 1e9)))
    def get_logger(self): return self.logger
    def destroy_node(self):
        assert isinstance(self._publishers,list) and isinstance(self._subscriptions,list)
        assert len(self._subscriptions)==len(self.subscriptions)
        return True
class QoSProfile:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)

for package in ('rclpy','builtin_interfaces','sensor_msgs','std_msgs','rail_robot_interfaces'):
    module(package)
module('rclpy.node', Node=Node)
module('rclpy.callback_groups', MutuallyExclusiveCallbackGroup=type('MutuallyExclusiveCallbackGroup', (), {}))
module('rclpy.qos', QoSProfile=QoSProfile,
       ReliabilityPolicy=NS(BEST_EFFORT=1, RELIABLE=2),
       DurabilityPolicy=NS(VOLATILE=1, TRANSIENT_LOCAL=2),
       HistoryPolicy=NS(KEEP_LAST=1))
module('builtin_interfaces.msg', Time=Time)
module('sensor_msgs.msg', Image=Image, CompressedImage=CompressedImage)
module('std_msgs.msg', Header=Header, String=String, Float64=Float64)
module('rcl_interfaces')
module('rcl_interfaces.msg', ParameterDescriptor=lambda **kwargs: NS(**kwargs))
module('rail_robot_interfaces.msg', CameraFrame=FrameMessage, DetectionSample=Sample, ModelTiming=Sample,
       MissionStatus=Sample, ActuatorDecision=Sample)
module('rail_robot_interfaces.srv', CapturePair=type('CapturePair', (), {}))

from node_runtime.data import CameraFrame
from node_runtime.ros.images import frame_to_message, encode_image
from node_runtime.ros.common import stamp_seconds
from node_runtime.ros import camera_node, vision_node, display_node
from rust_detector import LetterboxTransform, result_from_class_map
from crack_detector import CrackDetectionResult
from hrsegnet_crack_detector import DETECTOR_PREFIX
from obstacle_detector import ObstacleDetectionResult, DETECTOR_PREFIX as OBSTACLE_PREFIX

def frame_message(role, sequence, source=10.0, stream='camera-session', value=0):
    frame = CameraFrame(np.full((720,1280,3), value, dtype=np.uint8), sequence, source)
    message = frame_to_message(frame, role, source + 90.0, source)
    message.stream_id = stream
    return message

def detectors():
    rust = result_from_class_map(np.zeros((240,1280),dtype=np.uint8),
        LetterboxTransform(240,1280,240,1280,0,0), 'deeplabv3plus-tensorrt/student/fake')
    crack = CrackDetectionResult(np.zeros((180,1280),dtype=np.uint8), [], 0,
        180*1280, 0.0, False, DETECTOR_PREFIX + '/fake', 0.5)
    obstacle = ObstacleDetectionResult((), OBSTACLE_PREFIX + '/fake', 0.3, False)
    events = []
    def detector(name, value):
        return Mock(detect=Mock(side_effect=lambda frame: events.append(name) or value))
    return detector('rust',rust), detector('crack',crack), detector('obstacle',obstacle), events
'''


class RosVisionAdapterTests(unittest.TestCase):
    def run_contract(self, source):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), environment.get("PYTHONPATH", "")))
        result = subprocess.run(
            [sys.executable, "-B", "-c", ROS_DOUBLES + "\n" + textwrap.dedent(source)],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_camera_source_stamp_dedup_pair_and_resource_ownership(self):
        self.run_contract('''
            side, top = Mock(), Mock()
            frame = CameraFrame(np.zeros((720,1280,3),dtype=np.uint8),1,9.9)
            side.read_latest.return_value = top.read_latest.return_value = frame
            with patch.object(camera_node, 'validate_distinct_camera_devices'), patch.object(camera_node.time, 'monotonic', return_value=10.0):
                node = camera_node.CameraNode('side','top',camera_factory=Mock(side_effect=[side,top]))
                node._publish_latest()
                node._publish_latest()
                for role in ('side','top'):
                    messages = node.publishers['camera/'+role].messages
                    assert len(messages)==1
                    assert abs(stamp_seconds(messages[0].header.stamp)-99.9)<1e-6
                    assert messages[0].stream_id == node._stream_id
                with patch.object(camera_node, 'read_stopped_capture_pair', return_value=(frame,frame)) as capture:
                    response = node._capture_pair(NS(request_id='1'),NS())
                    assert response.success and response.side.image.data == response.top.image.data
                    capture.assert_called_once_with(side,top)
                with patch.object(camera_node, 'read_stopped_capture_pair', side_effect=TimeoutError('top failed')):
                    response = node._capture_pair(NS(request_id='2'),NS())
                    assert not response.success and 'top failed' in response.error
                node.destroy_node()
                node.destroy_node()
                side.close.assert_called_once()
                top.close.assert_called_once()
            assert 'run' not in sys.modules
        ''')

    def test_vision_timing_uses_actual_detector_boundaries(self):
        self.run_contract('''
            rust, crack, obstacle, _ = detectors()
            node = vision_node.VisionNode(rust, crack, obstacle)
            node._receive('side', frame_message('side', 1, 10.0))
            ticks = iter((10_000_000_000, 10_023_500_000))
            with patch.object(vision_node.time, 'monotonic_ns', side_effect=lambda: next(ticks)):
                node._process_latest()
            timing = node.publishers['debug/vision/timing'].messages[0]
            assert timing.processing_start_monotonic_ns == 10_000_000_000
            assert timing.processing_end_monotonic_ns == 10_023_500_000
            assert timing.processing_ms == 23.5
            node.destroy_node()
        ''')

    def test_obstacle_model_boundary_excludes_roi_clipping(self):
        self.run_contract('''
            from node_runtime import vision
            rust, crack, obstacle, events = detectors()
            inference = vision.AlternatingTopRealtimeInference(obstacle, crack)
            frame = CameraFrame(np.zeros((720, 1280, 3), np.uint8), 1, 10.0)
            inference.process(frame, frame.frame, frame.frame[:180])
            events.clear()
            ticks = iter((10_000_000_000, 10_005_000_000))
            def clock():
                events.append('clock')
                return next(ticks)
            def clip(result):
                events.append('clip')
                return result
            with patch.object(vision.time, 'monotonic_ns', side_effect=clock), \
                 patch.object(vision, 'clip_realtime_obstacles_to_roi', side_effect=clip):
                outcome = inference.process(frame, frame.frame, frame.frame[:180])
            assert events == ['clock', 'obstacle', 'clock', 'clip'], events
            assert outcome.obstacle_seconds == 0.005
        ''')

    def test_multitask_results_share_one_physical_execution_id(self):
        self.run_contract('''
            rust, crack, obstacle, _ = detectors()
            rust_result = rust.detect(np.zeros((240, 1280, 3), np.uint8))
            crack_result = crack.detect(np.zeros((180, 1280, 3), np.uint8))
            rust.detect = Mock(return_value=(rust_result, crack_result))
            node = vision_node.VisionNode(rust, crack, obstacle)
            node._inference['side'].multitask_enabled = True
            for sequence in (1, 2):
                node._receive('side', frame_message('side', sequence, 9.8 + 0.1 * sequence))
                node._process_latest()
            timings = node.publishers['debug/vision/timing'].messages
            assert len(timings) == 4
            assert [item.model for item in timings] == ['rust', 'crack', 'rust', 'crack']
            assert timings[0].execution_id == timings[1].execution_id != ''
            assert timings[2].execution_id == timings[3].execution_id != timings[0].execution_id
            assert rust.detect.call_count == 2
            node.destroy_node()
        ''')

    def test_separate_detector_invocations_get_distinct_execution_ids(self):
        self.run_contract('''
            rust, crack, obstacle, _ = detectors()
            node = vision_node.VisionNode(rust, crack, obstacle)
            for sequence in (1, 2):
                node._receive('side', frame_message('side', sequence, 9.8 + 0.1 * sequence))
                node._process_latest()
            timings = node.publishers['debug/vision/timing'].messages
            assert [item.model for item in timings] == ['rust', 'crack']
            assert timings[0].execution_id and timings[1].execution_id
            assert timings[0].execution_id != timings[1].execution_id
            node.destroy_node()
        ''')

    def test_camera_partial_initialization_closes_opened_reader(self):
        self.run_contract('''
            side=Mock()
            with patch.object(camera_node,'validate_distinct_camera_devices'):
                try:
                    camera_node.CameraNode('side','top',camera_factory=Mock(side_effect=[side,RuntimeError('top failed')]))
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('camera initialization failure was swallowed')
            side.close.assert_called_once()
        ''')

    def test_vision_serializes_shared_model_and_never_restamps_cached_results(self):
        self.run_contract('''
            rust,crack,obstacle,events=detectors()
            node=vision_node.VisionNode(rust,crack,obstacle)
            with patch.object(vision_node.time,'monotonic',side_effect=lambda: node.now-90.0):
                for sequence in (1,2):
                    node.now=100.0+(sequence-1)*0.1
                    for role in ('side','top'):
                        node._receive(role,frame_message(role,sequence,node.now-90.0))
                    node._process_latest()
                    node._process_latest()
                assert events==['rust','crack','crack','obstacle'],events
                results=node.publishers['vision/results'].messages
                assert [(m.camera_role,m.detector,m.sequence) for m in results]==[
                    ('side','rust',1),('top','crack',1),('side','crack',2),('top','obstacle',2)]
                assert all(m.valid and m.stream_id=='camera-session' for m in results)
                assert [m.mask.height for m in results[:3]]==[240,180,180]
                assert abs(stamp_seconds(results[0].header.stamp)-100.0)<1e-6
                assert abs(stamp_seconds(results[2].header.stamp)-100.1)<1e-6
                assert all(sub[2]['callback_group'] is node._group for sub in node.subscriptions)
                assert node._timer.kwargs['callback_group'] is node._group
                node._receive('side',frame_message('side',2,10.1))
                node._process_latest()
                assert len(results)==4
            node.destroy_node()
            crack.close.assert_called_once()
        ''')

    def test_vision_partial_initialization_releases_every_detector_once(self):
        self.run_contract('''
            rust,crack,obstacle,events=detectors()
            with patch.object(vision_node.VisionNode,'create_subscription',side_effect=RuntimeError('ROS setup failed')):
                try:
                    vision_node.VisionNode(rust,crack,obstacle)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('ROS setup failure was swallowed')
            rust.close.assert_called_once()
            crack.close.assert_called_once()
            obstacle.close.assert_called_once()
        ''')

    def test_vision_rejects_old_input_and_recovers_after_stream_restart(self):
        self.run_contract('''
            rust,crack,obstacle,events=detectors()
            node=vision_node.VisionNode(rust,crack,obstacle)
            with patch.object(vision_node.time,'monotonic',side_effect=lambda: node.now-90.0):
                node._receive('side',frame_message('side',100,9.0))
                node._process_latest()
                assert not events and not node.publishers['runtime/fault'].messages
                node._receive('side',frame_message('side',100,10.0))
                node._process_latest()
                node.now=100.1
                node._receive('side',frame_message('side',1,10.1,stream='restarted'))
                node._process_latest()
                node._receive('side',frame_message('side',999,10.0))
                node._process_latest()
                assert events==['rust','rust'],events
                assert [m.stream_id for m in node.publishers['vision/results'].messages]==['camera-session','restarted']
        ''')

    def test_expired_inference_is_invalid_but_does_not_emit_terminal_fault(self):
        self.run_contract('''
            rust,crack,obstacle,events=detectors()
            node=vision_node.VisionNode(rust,crack,obstacle)
            detect=rust.detect.side_effect
            def slow(frame):
                result=detect(frame)
                node.now=101.0
                return result
            rust.detect.side_effect=slow
            with patch.object(vision_node.time,'monotonic',side_effect=lambda: node.now-90.0):
                node._receive('side',frame_message('side',1))
                node._process_latest()
            messages=node.publishers['vision/results'].messages
            assert len(messages)==1 and not messages[0].valid
            assert stamp_seconds(messages[0].header.stamp)==100.0
            assert not node.publishers['runtime/fault'].messages
        ''')

    def test_preview_masks_match_original_not_newest_frame_or_old_stream(self):
        self.run_contract('''
            ui=Mock()
            ui.check_status.return_value=False
            ui.submit.return_value=True
            node=display_node.DisplayNode(display_factory=lambda:ui)
            node.now=100.1
            rust,_,_,_=detectors()
            sample=Sample()
            original=frame_message('side',1,10.0,value=11)
            sample.header=original.header
            sample.stream_id=original.stream_id
            sample.camera_role,sample.sequence,sample.detector,sample.valid='side',1,'rust',True
            sample.detector_method='deeplabv3plus-tensorrt/student/fake'
            sample.mask=encode_image(np.zeros((240,1280),dtype=np.uint8),sample.header)
            with patch.object(display_node.time,'monotonic',side_effect=lambda:node.now-90.0):
                node._receive_frame('side',original)
                node._receive_frame('side',frame_message('side',2,10.1,value=22))
                node._receive_sample(sample)
                with patch.object(display_node,'annotate_realtime_control_results',side_effect=lambda frame,*_:frame) as annotate:
                    node._render_role('side',10.1)
                    assert int(annotate.call_args.args[0][0,0,0])==11
                node.now=100.2
                node._receive_frame('side',frame_message('side',1,10.2,stream='new',value=33))
                node._receive_sample(sample)
                with patch.object(display_node,'annotate_realtime_control_results') as annotate:
                    rendered=node._render_role('side',node.now-90.0)
                    assert int(rendered[0,0,0])==33
                    annotate.assert_not_called()
                node.destroy_node()
                ui.close.assert_called_once()
            assert set(node.publishers)=={'/rail_robot/gui/image/compressed','/rail_robot/gui/status'}
            assert 'runtime/fault' not in node.publishers
        ''')


if __name__ == "__main__":
    unittest.main()
