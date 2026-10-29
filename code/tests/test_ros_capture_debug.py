"""Capture debug contracts with fake ROS/models/cameras, never real devices or DDS."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

from test_ros_vision import JETSON_CODE, ROS_DOUBLES


CAPTURE_DOUBLES = r'''
import hashlib, os, tempfile, threading
from pathlib import Path
import cv2
class CaptureResult:
    __slots__=('header','annotated','capture_role','camera_role','stream_id','capture_id','error',
        'raw_path','annotated_path','primary_method','crack_method','sequence','obstacle_count',
        'primary_seconds','crack_seconds','total_seconds','rust_ratio','crack_ratio','success')
    def __init__(self):
        self.header,self.annotated=Header(),Image()
        for name in ('capture_role','camera_role','stream_id','capture_id','error','raw_path','annotated_path','primary_method','crack_method'):
            setattr(self,name,'')
        self.sequence=self.obstacle_count=0
        self.primary_seconds=self.crack_seconds=self.total_seconds=self.rust_ratio=self.crack_ratio=0.0
        self.success=False
sys.modules['rail_robot_interfaces.msg'].CaptureDebugResult=CaptureResult
sys.modules['std_msgs.msg'].Empty=type('Empty',(),{})
module('std_srvs')
module('std_srvs.srv',Trigger=type('Trigger',(),{}))
from node_runtime.ros import capture_debug_node as debug, capture_debug_models as models
from node_runtime.ros import deployment
from node_runtime.ros.images import decode_image
from obstacle_detector import ObstacleDetection
import rust_detector, hrsegnet_crack_detector, obstacle_detector

def capture_results():
    mask=np.zeros((720,1280),np.uint8)
    mask[100:120,100:120]=255
    crack=CrackDetectionResult(mask,[(100,100,20,20)],400,720*1280,400/(720*1280),True,
        hrsegnet_crack_detector.CAPTURE_DETECTOR_PREFIX+'fake',0.55)
    obstacle=ObstacleDetectionResult((ObstacleDetection((100,100,120,120),0.9,0),
        ObstacleDetection((400,400,420,420),0.8,0)),OBSTACLE_PREFIX+'fake',0.424,False)
    classes=np.zeros((720,1280),np.uint8)
    classes[100:120,100:120]=3
    classes[300:310,300:310]=2
    rust=result_from_class_map(classes,LetterboxTransform(720,1280,720,1280,0,0),
        'deeplabv3plus-tensorrt/teacher/fake')
    return rust,crack,obstacle

def finish(node):
    item=node._completed.get(timeout=5)
    node._completed.put_nowait(item)
    node._tick()
    return node.publishers['debug/'+node._role+'/result'].messages[-1]

def plan_parameters(root,roles):
    values={}
    for role in roles:
        path=Path(root)/(role+'.plan')
        data=role.encode()
        path.write_bytes(data)
        values[role+'_engine']=str(path)
        values[role+'_sha256']=hashlib.sha256(data).hexdigest()
    return values
'''


class RosCaptureDebugTests(unittest.TestCase):
    def run_contract(self, source):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), environment.get("PYTHONPATH", "")))
        result = subprocess.run(
            [sys.executable, "-B", "-c", ROS_DOUBLES + CAPTURE_DOUBLES + textwrap.dedent(source)],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_camera_opens_only_selected_role_rejects_pair_and_counts_dropped_frames(self):
        self.run_contract('''
            for selected,role in (('front','top'),('side','side')):
                camera=Mock()
                camera.read_latest.return_value=CameraFrame(np.zeros((720,1280,3),np.uint8),1,9.9)
                factory=Mock(return_value=camera)
                with patch.object(camera_node.time,'monotonic',return_value=10.0):
                    node=camera_node.CameraNode('side-device','top-device',camera_roles=selected,camera_factory=factory)
                    factory.assert_called_once_with(role+'-device')
                    assert set(node._cameras)=={role}
                    node._publish_latest()
                    assert node.readiness()[0]
                    response=node._capture_pair(NS(),NS())
                    assert not response.success and 'both' in response.error
                    assert not node.publishers['runtime/fault'].messages
                    camera.read_latest.return_value=CameraFrame(np.zeros((720,1280,3),np.uint8),4,10.0)
                    node._publish_latest()
                    assert abs(node.publishers['debug/camera/'+selected+'/fps'].messages[-1].data-30.0)<1e-6
                    camera.read_latest.side_effect=TimeoutError()
                with patch.object(camera_node.time,'monotonic',return_value=12.0):
                    node._publish_latest()
                    assert node.publishers['debug/camera/'+selected+'/fps'].messages[-1].data==0.0
                    assert not node.readiness()[0]
                node.destroy_node()
                camera.close.assert_called_once()
            factory=Mock()
            try:camera_node.CameraNode(camera_roles='invalid',camera_factory=factory)
            except ValueError:pass
            else:raise AssertionError('Invalid role opened camera')
            factory.assert_not_called()
        ''')

    def test_factories_validate_only_required_plans_and_preserve_baseline(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'0'}):
                values=plan_parameters(root,('capture_rust','capture_crack','obstacle'))
                for role in ('front','side'):
                    primary,crack=Mock(),Mock()
                    with patch.object(rust_detector,'RustDetector',return_value=primary) as rust, \
                         patch.object(obstacle_detector,'ObstacleDetector',return_value=primary) as obstacle, \
                         patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector',return_value=crack) as crack_factory:
                        node=Node('factory',parameters=values)
                        assert models.create_capture_debug_detectors(node,role)==(primary,crack)
                        crack_factory.assert_called_once_with(Path(values['capture_crack_engine']),values['capture_crack_sha256'],0.55,20,role='capture')
                        if role=='front':
                            rust.assert_not_called()
                            obstacle.assert_called_once_with(Path(values['obstacle_engine']),values['obstacle_sha256'],0.424)
                            assert 'capture_rust_profile' not in node.parameters
                        else:
                            obstacle.assert_not_called()
                            rust.assert_called_once_with(Path(values['capture_rust_engine']),'teacher',values['capture_rust_sha256'])
                            assert 'obstacle_confidence_threshold' not in node.parameters
                        assert not any(name.startswith('realtime_') for name in node.parameters)
                        assert 'inspection_report' not in models.__dict__
                bad=dict(values,capture_crack_sha256='0'*64)
                with patch.object(obstacle_detector,'ObstacleDetector') as factory:
                    try:models.create_capture_debug_detectors(Node('factory',parameters=bad),'front')
                    except ValueError:pass
                    else:raise AssertionError('Bad SHA accepted')
                    factory.assert_not_called()
                with patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'1'}),patch.object(obstacle_detector,'ObstacleDetector') as factory:
                    try:models.create_capture_debug_detectors(Node('factory',parameters=values),'front')
                    except RuntimeError:pass
                    else:raise AssertionError('TF32 conflict accepted')
                    factory.assert_not_called()
                primary=Mock()
                with patch.object(obstacle_detector,'ObstacleDetector',return_value=primary),patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector',side_effect=RuntimeError('engine failed')):
                    try:models.create_capture_debug_detectors(Node('factory',parameters=values),'front')
                    except RuntimeError:pass
                    else:raise AssertionError('Partial model failure swallowed')
                    primary.close.assert_called_once()
        ''')

    def test_foreign_factory_loads_only_the_pinned_obstacle_model(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'0'}):
                values=plan_parameters(root,('obstacle',))
                primary=Mock()
                with patch.object(obstacle_detector,'ObstacleDetector',return_value=primary) as obstacle, \
                     patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector') as crack_factory:
                    node=Node('factory',parameters=values)
                    detectors=models.create_capture_debug_detectors(node,'foreign')
                    assert detectors==(primary,None)
                    obstacle.assert_called_once_with(Path(values['obstacle_engine']),values['obstacle_sha256'],0.424)
                    crack_factory.assert_not_called()
                    assert not any(name.startswith('capture_crack_') for name in node.parameters)
        ''')

    def test_foreign_capture_uses_only_yolo_and_saves_box_annotated_top_frame(self):
        self.run_contract('''
            from obstacle_detector import ObstacleDetectionResult
            obstacle=ObstacleDetectionResult((ObstacleDetection((100,100,120,120),0.9,0),
                ObstacleDetection((400,400,420,420),0.8,0)),OBSTACLE_PREFIX+'fake',0.424,False)
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                primary=Mock(detect=Mock(return_value=obstacle))
                node=debug.CaptureDebugNode(primary,capture_role='foreign',show_gui=False,output_directory=root)
                node._receive(frame_message('top',7,9.9,value=37))
                assert node._capture(NS(),NS()).success
                result=finish(node)
                raw=cv2.imread(result.raw_path)
                annotated=cv2.imread(result.annotated_path)
                assert result.success,result.error
                assert result.capture_role=='foreign' and result.camera_role=='top'
                assert result.obstacle_count==2
                assert result.crack_method=='' and result.crack_seconds==0 and result.crack_ratio==0
                assert Path(result.raw_path).parent==Path(root)/'foreign'
                assert np.all(raw==37) and not np.array_equal(annotated,raw)
                assert not np.array_equal(annotated[100,100],raw[100,100])
                primary.detect.assert_called_once()
                assert 'debug/foreign/result' in node.publishers
                assert 'debug/front/result' not in node.publishers
                node.destroy_node()
                primary.close.assert_called_once()
        ''')

    def test_front_capture_persists_original_before_models_and_keeps_source_identity(self):
        self.run_contract('''
            rust,crack,obstacle=capture_results()
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                frames=[]
                def detect(frame):
                    files=list(Path(root).rglob('*_raw.jpg'))
                    assert len(files)==1
                    assert np.all(cv2.imread(str(files[0]))==37)
                    frames.append(frame.copy())
                    return obstacle
                primary=Mock(detect=Mock(side_effect=detect))
                crack_model=Mock(detect=Mock(return_value=crack))
                node=debug.CaptureDebugNode(primary,crack_model,capture_role='front',show_gui=False,output_directory=root)
                message=frame_message('top',5,9.9,value=37)
                node._receive(message)
                assert node._capture(NS(),NS()).success
                message.header.stamp.sec=0
                node._receive(frame_message('top',6,10.0,value=91))
                result=finish(node)
                assert result.success,result.error
                assert result.capture_role=='front' and result.camera_role=='top'
                assert result.sequence==5 and result.stream_id=='camera-session'
                assert abs(stamp_seconds(result.header.stamp)-99.9)<1e-6
                assert abs(stamp_seconds(result.annotated.header.stamp)-99.9)<1e-6
                assert result.obstacle_count==1
                assert result.crack_ratio==400/(720*1280)
                assert Path(result.raw_path).exists() and Path(result.annotated_path).exists()
                assert np.all(frames[0]==37) and frames[0].shape==(720,1280,3)
                assert np.all(crack_model.detect.call_args.args[0]==37)
                assert np.all(node._latest.frame==91)
                assert np.array_equal(node._capture_view,decode_image(result.annotated))
                assert result.total_seconds>=result.primary_seconds+result.crack_seconds>=0
                node.destroy_node()
                primary.close.assert_called_once()
                crack_model.close.assert_called_once()
                assert 'run' not in sys.modules
        ''')

    def test_side_capture_applies_crack_priority_and_does_not_write_report(self):
        self.run_contract('''
            rust,crack,obstacle=capture_results()
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                primary,crack_model=Mock(detect=Mock(return_value=rust)),Mock(detect=Mock(return_value=crack))
                node=debug.CaptureDebugNode(primary,crack_model,capture_role='side',show_gui=False,output_directory=root)
                node._receive(frame_message('top',1,9.9))
                assert not node.readiness()[0]
                node._receive(frame_message('side',2,9.9))
                node._trigger(NS())
                result=finish(node)
                assert result.success,result.error
                assert abs(result.rust_ratio-100/(720*1280))<1e-12
                assert result.primary_method.startswith('deeplabv3plus-tensorrt/teacher/')
                assert result.crack_method.startswith(hrsegnet_crack_detector.CAPTURE_DETECTOR_PREFIX)
                assert not list(Path(root).rglob('*.xlsx'))
                assert len(list(Path(root).rglob('*.jpg')))==2
                assert not node._capture(NS(),NS()).success
                assert 'already captured' in node._status
                node.destroy_node()
        ''')

    def test_gui_keys_remain_responsive_during_analysis_and_reject_busy(self):
        self.run_contract('''
            rust,crack,obstacle=capture_results()
            entered,release=threading.Event(),threading.Event()
            def detect(frame):
                entered.set()
                assert release.wait(5)
                return obstacle
            display=Mock(check_status=Mock(return_value=False),poll_key=Mock(return_value=ord('S')))
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                node=debug.CaptureDebugNode(Mock(detect=Mock(side_effect=detect)),Mock(detect=Mock(return_value=crack)),
                    capture_role='front',show_gui=True,output_directory=root,display_factory=Mock(return_value=display))
                assert not node._capture(NS(),NS()).success
                node._receive(frame_message('top',1,9.9,value=20))
                node._tick()
                assert entered.wait(3)
                response=node._capture(NS(),NS())
                assert not response.success and 'already being analyzed' in response.message
                node._receive(frame_message('top',2,10.0,value=60))
                display.poll_key.return_value=None
                node._tick()
                canvas=display.submit.call_args.args[0]
                assert canvas.shape==(440,1280,3)
                assert np.all(canvas[100:300,20:600]==60)
                assert np.all(canvas[100:300,660:1240]==20)
                display.poll_key.return_value=ord('q')
                node._tick()
                assert node.shutdown_requested()
                release.set()
                finish(node)
                node.destroy_node()
                display.close.assert_called_once()
        ''')

    def test_stale_restarted_and_duplicate_inputs_never_capture_wrong_original(self):
        self.run_contract('''
            rust,crack,obstacle=capture_results()
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                node=debug.CaptureDebugNode(Mock(detect=Mock(return_value=obstacle)),Mock(detect=Mock(return_value=crack)),
                    capture_role='front',show_gui=False,output_directory=root)
                node._receive(frame_message('top',1,9.0))
                assert not node._capture(NS(),NS()).success
                node._receive(frame_message('top',100,9.9,stream='old',value=20))
                node._receive(frame_message('top',1,10.0,stream='new',value=60))
                node._receive(frame_message('top',101,9.95,stream='old',value=90))
                assert node._latest_key[:2]==('new',1)
                assert node._capture(NS(),NS()).success
                result=finish(node)
                assert result.sequence==1 and result.stream_id=='new'
                assert np.all(cv2.imread(result.raw_path)==60)
                assert not node._capture(NS(),NS()).success
                node.prepare_shutdown()
                assert not node._capture(NS(),NS()).success
                node.destroy_node()
        ''')

    def test_model_failure_preserves_raw_and_reports_failure_without_annotated_success(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                primary,crack=Mock(detect=Mock(side_effect=RuntimeError('CUDA fake failure'))),Mock()
                node=debug.CaptureDebugNode(primary,crack,capture_role='front',show_gui=False,output_directory=root)
                node._receive(frame_message('top',1,9.9))
                assert node._capture(NS(),NS()).success
                result=finish(node)
                assert not result.success and 'CUDA fake failure' in result.error
                assert Path(result.raw_path).exists() and not result.annotated_path
                assert result.annotated.height==0
                crack.detect.assert_not_called()
                node.destroy_node()
        ''')

    def test_shutdown_does_not_close_in_use_models_or_publish_late_completion(self):
        self.run_contract('''
            rust,crack,obstacle=capture_results()
            entered,release=threading.Event(),threading.Event()
            def detect(frame):
                entered.set()
                assert release.wait(5)
                return obstacle
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                primary=Mock(detect=Mock(side_effect=detect))
                crack_model=Mock(detect=Mock(return_value=crack))
                node=debug.CaptureDebugNode(primary,crack_model,capture_role='front',show_gui=False,output_directory=root)
                node._receive(frame_message('top',1,9.9))
                node._capture(NS(),NS())
                assert entered.wait(3)
                node.prepare_shutdown()
                primary.close.assert_not_called()
                crack_model.close.assert_not_called()
                assert not node._capture(NS(),NS()).success
                release.set()
                node.destroy_node()
                node._tick()
                assert not node.publishers['debug/front/result'].messages
                assert not list(Path(root).rglob('*_annotated.jpg'))
                primary.close.assert_called_once()
                crack_model.close.assert_called_once()
                crack_model.detect.assert_not_called()
                node.destroy_node()
                primary.close.assert_called_once()
        ''')

    def test_invalid_capture_model_roles_masks_and_measurements_are_reported(self):
        self.run_contract('''
            from copy import deepcopy
            for invalid in ('crack_role','crack_shape','crack_pixels','crack_values','teacher_role','teacher_classes'):
                rust,crack,obstacle=capture_results()
                role='side' if invalid.startswith('teacher') else 'front'
                primary=rust if role=='side' else obstacle
                if invalid=='crack_role':crack.method=hrsegnet_crack_detector.DETECTOR_PREFIX+'fake'
                if invalid=='crack_shape':crack.mask=crack.mask[:128]
                if invalid=='crack_pixels':crack.crack_pixels=0
                if invalid=='crack_values':crack.mask[0,0]=1
                if invalid=='teacher_role':rust.method='deeplabv3plus-tensorrt/student/fake'
                if invalid=='teacher_classes':rust.class_map[0,0]=4
                with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                    node=debug.CaptureDebugNode(Mock(detect=Mock(return_value=primary)),Mock(detect=Mock(return_value=crack)),
                        capture_role=role,show_gui=False,output_directory=root)
                    node._receive(frame_message('side' if role=='side' else 'top',1,9.9))
                    assert node._capture(NS(),NS()).success
                    result=finish(node)
                    assert not result.success and result.error,invalid
                    assert Path(result.raw_path).exists() and not result.annotated_path,invalid
                    node.destroy_node()
        ''')

    def test_join_timeout_defers_close_until_the_model_worker_exits(self):
        self.run_contract('''
            rust,crack,obstacle=capture_results()
            entered,release=threading.Event(),threading.Event()
            def detect(frame):
                entered.set()
                assert release.wait(5)
                return obstacle
            with tempfile.TemporaryDirectory() as root,patch.object(debug.time,'monotonic',return_value=10.0):
                primary=Mock(detect=Mock(side_effect=detect))
                crack_model=Mock(detect=Mock(return_value=crack))
                node=debug.CaptureDebugNode(primary,crack_model,capture_role='front',show_gui=False,output_directory=root)
                node._receive(frame_message('top',1,9.9))
                node._capture(NS(),NS())
                assert entered.wait(3)
                with patch.object(node._worker,'join',return_value=None):
                    node.destroy_node()
                assert node._closed and node._worker.is_alive()
                primary.close.assert_not_called()
                crack_model.close.assert_not_called()
                release.set()
                node._worker.join(5)
                assert not node._worker.is_alive()
                primary.close.assert_called_once()
                crack_model.close.assert_called_once()
                assert not node.publishers['debug/front/result'].messages
        ''')
