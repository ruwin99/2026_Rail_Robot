"""Deployment factory/parameter contracts with fake models and isolated ROS APIs."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

from test_ros_vision import JETSON_CODE, ROS_DOUBLES


DEPLOYMENT_DOUBLES = r'''
import hashlib, os, tempfile
from pathlib import Path
import cv2
from node_runtime.ros import deployment
import rust_detector, hrsegnet_crack_detector, obstacle_detector

PARAMETERS={}
old_init=Node.__init__
def node_init(self,*args,**kwargs):
    old_init(self,*args,**kwargs)
    self.parameters={}
    self.overrides=dict(PARAMETERS)
    self.descriptors={}
def declare(self,name,default,descriptor=None):
    self.parameters[name]=self.overrides.get(name,default)
    self.descriptors[name]=descriptor
    return NS(value=self.parameters[name])
Node.__init__=node_init
Node.declare_parameter=declare
Node.has_parameter=lambda self,name:name in self.parameters
Node.get_parameter=lambda self,name:NS(value=self.parameters[name])
Node.create_client=lambda *args,**kwargs:Mock()
Node.create_service=lambda self,service_type,topic,callback,**kwargs:callback
Node.destroy_timer=lambda self,timer:True
module('rcl_interfaces')
module('rcl_interfaces.msg',ParameterDescriptor=lambda **kwargs:NS(**kwargs))
module('std_srvs')
module('std_srvs.srv',Trigger=type('Trigger',(),{}))
sys.modules['rclpy.callback_groups'].ReentrantCallbackGroup=type('ReentrantCallbackGroup',(),{})
sys.modules['rail_robot_interfaces.srv'].CaptureInspection=type('CaptureInspection',(),{})
sys.modules['rail_robot_interfaces.srv'].DrainInspection=type('DrainInspection',(),{})
from node_runtime.ros import inspection_node, training_capture_node

def plan_parameters(root,roles):
    values={}
    for role in roles:
        path=Path(root)/f'{role}.plan'
        data=role.encode()
        path.write_bytes(data)
        values[role+'_engine']=str(path)
        values[role+'_sha256']=hashlib.sha256(data).hexdigest()
    return values

class FakeDetector:
    def __init__(self,path,digest,method):
        assert os.environ.get('NVIDIA_TF32_OVERRIDE')=='0'
        self.engine_path,self.engine_sha256,self.method=Path(path),digest,method
        self.frames=[]
        self.closed=0
    def detect(self,frame):
        self.frames.append(frame.copy())
        return NS(status='ready',mask=np.zeros(frame.shape[:2],dtype=np.uint8))
    def close(self):self.closed+=1

def rust_factory(path,profile,digest,**kwargs):
    return FakeDetector(path,digest,'deeplabv3plus-tensorrt/'+profile+'/trt-10.3.0/cuda:0'+('/gpu-argmax' if kwargs.get('gpu_argmax') else ''))
def crack_factory(path,digest,threshold,components,*,role,gpu_preprocess=False,linear_restore=False):
    prefix=hrsegnet_crack_detector.CAPTURE_DETECTOR_PREFIX if role=='capture' else hrsegnet_crack_detector.DETECTOR_PREFIX
    result=FakeDetector(path,digest,prefix+'trt-10.3.0/cuda:0/logit-margin')
    result.threshold,result.components,result.role=threshold,components,role
    result.gpu_preprocess,result.linear_restore=gpu_preprocess,linear_restore
    return result
def obstacle_factory(path,digest,confidence):
    result=FakeDetector(path,digest,obstacle_detector.DETECTOR_PREFIX+'trt-10.3.0/cuda:0')
    result.confidence=confidence
    return result
'''


class RosDeploymentVisionTests(unittest.TestCase):
    def run_contract(self, source):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), environment.get("PYTHONPATH", "")))
        result = subprocess.run(
            [sys.executable, "-B", "-c", ROS_DOUBLES + DEPLOYMENT_DOUBLES + textwrap.dedent(source)],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tf32_and_sha_preflight_reject_before_model_construction(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root:
                PARAMETERS.update(plan_parameters(root,('realtime_rust','realtime_crack','obstacle')))
                node=Node('factory')
                with patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'1'}),patch.object(rust_detector,'RustDetector') as factory:
                    try:deployment.create_realtime_detectors(node)
                    except RuntimeError:pass
                    else:raise AssertionError('TF32 conflict accepted')
                    factory.assert_not_called()
                PARAMETERS['obstacle_sha256']='0'*64
                with patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'0'}),patch.object(rust_detector,'RustDetector') as factory:
                    try:deployment.create_realtime_detectors(Node('factory'))
                    except ValueError:pass
                    else:raise AssertionError('Bad engine digest accepted')
                    factory.assert_not_called()
        ''')

    def test_ros_obstacle_default_selects_the_fp16_production_candidate(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), environment.get("PYTHONPATH", "")))
        source = textwrap.dedent('''
            import sys, types
            sys.modules['cv2']=types.ModuleType('cv2')
            msg=types.ModuleType('rcl_interfaces.msg')
            msg.ParameterDescriptor=lambda **kwargs:types.SimpleNamespace(**kwargs)
            sys.modules['rcl_interfaces']=types.ModuleType('rcl_interfaces')
            sys.modules['rcl_interfaces.msg']=msg
            from node_runtime.ros import deployment
            class Node:
                def has_parameter(self,name): return False
                def declare_parameter(self,name,default,descriptor=None): return types.SimpleNamespace(value=default)
            values=deployment.read_parameters(Node(),('obstacle_engine','obstacle_sha256'))
            assert values['obstacle_engine']=='/home/bitdol/models/rebuilt_20260929/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-int-h736-fp16-notf32-rb20260929.plan'
            assert values['obstacle_sha256']=='280e9669624b67c9eb70c3db9661ccf6ae2a78c978603e7a39223a635cb6a7d6'
        ''')
        result = subprocess.run(
            [sys.executable, "-B", "-c", source],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ros_realtime_crack_default_selects_retrained_fp16_without_changing_capture(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), environment.get("PYTHONPATH", "")))
        source = textwrap.dedent('''
            import sys, types
            sys.modules['cv2']=types.ModuleType('cv2')
            msg=types.ModuleType('rcl_interfaces.msg')
            msg.ParameterDescriptor=lambda **kwargs:types.SimpleNamespace(**kwargs)
            sys.modules['rcl_interfaces']=types.ModuleType('rcl_interfaces')
            sys.modules['rcl_interfaces.msg']=msg
            from node_runtime.ros import deployment
            class Node:
                def has_parameter(self,name): return False
                def declare_parameter(self,name,default,descriptor=None): return types.SimpleNamespace(value=default)
            values=deployment.read_parameters(Node(),('realtime_crack_engine','realtime_crack_sha256','realtime_crack_probability_threshold','capture_crack_engine','capture_crack_sha256'))
            assert values['realtime_crack_engine']=='/home/bitdol/models/rebuilt_20260929/hrsegnet-b32-steelcrack-aug400-realtime-w1280-h128-fp16-rb20260929.plan'
            assert values['realtime_crack_sha256']=='0ae230c4c9a6cf00de0b108fc69f27e93e9306c457a2c06ffc79fb944896fcd9'
            assert values['realtime_crack_probability_threshold']==.55
            assert values['capture_crack_engine'].endswith('hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32-rb20260929.plan')
            assert values['capture_crack_sha256']=='72ad6ca53002d781ce750bb7c8857efc424ec44ddf889cf712a4351c3d3d9953'
        ''')
        result = subprocess.run(
            [sys.executable, "-B", "-c", source],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_realtime_factory_uses_role_contracts_and_cleans_partial_models(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root:
                PARAMETERS.update(plan_parameters(root,('realtime_rust','realtime_crack','obstacle')))
                node=Node('factory')
                with patch.dict(os.environ,{},clear=True),patch.object(rust_detector,'RustDetector',side_effect=rust_factory) as rust,patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector',side_effect=crack_factory),patch.object(obstacle_detector,'ObstacleDetector',side_effect=obstacle_factory):
                    models=deployment.create_realtime_detectors(node)
                    assert rust.call_args.args[1]=='student' and rust.call_args.kwargs['gpu_argmax'] is True
                    assert models[1].role=='realtime' and models[1].threshold==.55 and models[1].components==20
                    assert models[1].gpu_preprocess is True and models[1].linear_restore is True
                    assert models[2].confidence==obstacle_detector.DEFAULT_CONFIDENCE_THRESHOLD==.424
                    assert all(value.read_only for value in node.descriptors.values())
                partial=Mock()
                with patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'0'}),patch.object(rust_detector,'RustDetector',return_value=partial),patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector',side_effect=RuntimeError('load failed')):
                    try:deployment.create_realtime_detectors(Node('factory'))
                    except RuntimeError:pass
                    else:raise AssertionError('Partial model failure ignored')
                partial.close.assert_called_once()
        ''')

    def test_capture_factory_warms_models_but_keeps_expected_realtime_provenance(self):
        self.run_contract('''
            from inspection_report import InspectionWorkbook
            with tempfile.TemporaryDirectory() as root:
                PARAMETERS.update(plan_parameters(root,('capture_rust','capture_crack')))
                PARAMETERS.update(output_directory=str(Path(root)/'captures'))
                PARAMETERS['realtime_rust_engine']=str(Path(root)/'not-installed-realtime.plan')
                PARAMETERS['realtime_crack_engine']=str(Path(root)/'not-installed-crack.plan')
                with patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'0'}),patch.object(rust_detector,'RustDetector',side_effect=rust_factory),patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector',side_effect=crack_factory),patch.object(InspectionWorkbook,'append_capture') as append,patch.object(InspectionWorkbook,'append_top_crack_capture') as append_top:
                    teacher,crack,workbook,output=deployment.create_inspection_resources(Node('inspection'))
                assert teacher.frames[0].shape==crack.frames[0].shape==(720,1280,3)
                assert len(teacher.frames)==len(crack.frames)==1
                assert not np.any(teacher.frames[0])
                assert workbook.path.is_file() and output==Path(root)/'captures'
                assert workbook.capture_detector==teacher.method
                assert workbook.crack_detector==crack.method
                assert 'configured' in workbook.realtime_detector and 'configured' in workbook.realtime_crack_detector
                assert workbook.realtime_cuda_argmax_enabled
                assert workbook.realtime_model_sha256==deployment.DEFAULTS['realtime_rust_sha256']
                append.assert_not_called()
                append_top.assert_not_called()
        ''')

    def test_capture_warmup_failure_releases_models_before_report_creation(self):
        self.run_contract('''
            from inspection_report import InspectionWorkbook
            with tempfile.TemporaryDirectory() as root:
                PARAMETERS.update(plan_parameters(root,('capture_rust','capture_crack')))
                teacher=Mock(method='teacher')
                teacher.detect.side_effect=RuntimeError('CUDA warmup failed')
                crack=Mock(method='crack')
                with patch.dict(os.environ,{'NVIDIA_TF32_OVERRIDE':'0'}),patch.object(rust_detector,'RustDetector',return_value=teacher),patch.object(hrsegnet_crack_detector,'HrSegNetCrackDetector',return_value=crack),patch('inspection_report.InspectionWorkbook') as report:
                    try:deployment.create_inspection_resources(Node('inspection'))
                    except RuntimeError:pass
                    else:raise AssertionError('Warmup failure accepted')
                teacher.close.assert_called_once()
                crack.close.assert_called_once()
                report.assert_not_called()
        ''')

    def test_no_argument_constructors_resolve_resources_and_shutdown_hooks_do_not_close_early(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root:
                PARAMETERS.update(side_camera_device='side-device',top_camera_device='top-device')
                side,top=Mock(),Mock()
                frame=CameraFrame(np.zeros((720,1280,3),dtype=np.uint8),1,10.0)
                side.read_latest.return_value=top.read_latest.return_value=frame
                with patch.object(camera_node,'validate_distinct_camera_devices') as validate,patch.object(camera_node.time,'monotonic',return_value=10.0):
                    camera=camera_node.CameraNode(camera_factory=Mock(side_effect=[side,top]))
                    validate.assert_called_once_with('side-device','top-device')
                    assert camera.readiness()[0] is False
                    camera._publish_latest()
                    assert camera.readiness()[0] is True
                    camera.prepare_shutdown()
                    assert camera.readiness()[0] is False
                    side.close.assert_not_called()
                    camera.destroy_node()
                    side.close.assert_called_once()
                rust,crack,obstacle,events=detectors()
                with patch.object(vision_node,'create_realtime_detectors',return_value=(rust,crack,obstacle)) as factory:
                    vision=vision_node.VisionNode()
                    factory.assert_called_once_with(vision)
                worker=Mock()
                worker.failure_message.return_value=None
                worker.shutdown.return_value=None
                worker.wait_until_idle.return_value=True
                workbook=Mock()
                with patch.object(inspection_node,'create_inspection_resources',return_value=(rust,crack,workbook,Path(root))),patch.object(inspection_node,'create_inspection_obstacle_factory',return_value=lambda: obstacle),patch.object(inspection_node,'CaptureAnalysisWorker',return_value=worker),patch.object(inspection_node,'finalize_dashboard_run'):
                    inspection=inspection_node.InspectionNode()
                    assert inspection.readiness()[0] is True
                    pending=Mock()
                    inspection._pending_pair=pending
                    inspection.prepare_shutdown()
                    pending.cancel.assert_called_once()
                    worker.shutdown.assert_not_called()
                    inspection.destroy_node()
                    worker.shutdown.assert_called_once()
        ''')

    def test_vision_full_frame_obstacle_input_and_four_role_readiness(self):
        self.run_contract('''
            rust,crack,obstacle,events=detectors()
            node=vision_node.VisionNode(rust,crack,obstacle)
            with patch.object(vision_node.time,'monotonic',side_effect=lambda:node.now-90):
                assert node.readiness()[0] is False
                for sequence in (1,2):
                    node.now=100+(sequence-1)*.1
                    for role in ('side','top'):node._receive(role,frame_message(role,sequence,node.now-90))
                    node._process_latest()
                assert obstacle.detect.call_args.args[0].shape==(720,1280,3)
                assert node.readiness()[0] is True
                node.now=101
                assert node.readiness()[0] is False
                node.prepare_shutdown()
                crack.close.assert_not_called()
                node.destroy_node()
                crack.close.assert_called_once()
        ''')

    def test_training_service_saves_original_and_rejects_stale_or_stopping_frames(self):
        self.run_contract('''
            with tempfile.TemporaryDirectory() as root:
                PARAMETERS.update(training_camera_role='side',training_output_directory=root)
                node=training_capture_node.TrainingCaptureNode()
                with patch.object(training_capture_node.time,'monotonic',side_effect=lambda:node.now-90):
                    response=node._save(NS(),NS())
                    assert not response.success
                    message=frame_message('side',1,10.0,value=17)
                    node._receive(message)
                    response=node._save(NS(),NS())
                    assert response.success,response.message
                    path=Path(response.message)
                    assert path.parent==Path(root) and path.name.startswith('training_side_')
                    expected=cv2.imencode('.jpg',np.full((720,1280,3),17,dtype=np.uint8))[1].tobytes()
                    assert path.read_bytes()==expected
                    node.now=100.4
                    assert not node._save(NS(),NS()).success
                    assert len(list(Path(root).glob('*.jpg')))==1
                    node.prepare_shutdown()
                    assert not node.readiness()[0]
                    node.destroy_node()
                assert 'tensorrt' not in sys.modules
        ''')

    def test_shutdown_during_drain_cannot_publish_successful_finalization(self):
        self.run_contract('''
            worker=Mock()
            worker.failure_message.return_value=None
            with tempfile.TemporaryDirectory() as root,patch.object(inspection_node,'CaptureAnalysisWorker',return_value=worker),patch.object(inspection_node,'finalize_dashboard_run') as finalize:
                node=inspection_node.InspectionNode(Mock(),Mock(),Mock(),Path(root))
                node._counts={key:4 for key in node._counts}
                def drained(timeout):
                    node.prepare_shutdown()
                    return True
                worker.wait_until_idle.side_effect=drained
                response=node._drain(NS(finalize=True),NS())
                assert response.success is False and 'stopping' in response.error
                finalize.assert_not_called()
                assert not node.publishers['inspection/finalized'].messages
        ''')


if __name__ == "__main__":
    unittest.main()
