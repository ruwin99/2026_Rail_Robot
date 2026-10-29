"""GUI JPEG/status presentation contracts; ROS doubles are not native DDS tests."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

from test_ros_vision import JETSON_CODE, ROS_DOUBLES


GUI_DOUBLES = r'''
import json, threading
import cv2
from node_runtime.ros.common import qos_sensor, qos_state, stamp_from_seconds

def gui_status(node):
    return json.loads(node.publishers[node._status_topic].messages[-1].data)

def latest_trace(node):
    topic='/rail_robot/debug/frame_trace'
    assert (topic in node.publishers and node.publishers[topic].messages), (
        gui_status(node), node.logger.error.call_args_list, node.logger.warning.call_args_list)
    return node.publishers[topic].messages[-1]

def raw_sample(message,detector='rust'):
    sample=Sample()
    sample.header=message.header
    sample.camera_role=message.camera_role
    sample.sequence=message.sequence
    sample.stream_id=message.stream_id
    sample.detector=detector
    sample.valid=True
    sample.detector_method='deeplabv3plus-tensorrt/student/fake'
    mask=np.zeros((180 if detector=='crack' else 240,1280),np.uint8)
    mask[80:140,100:200]=255 if detector=='crack' else 3
    sample.mask=encode_image(mask,message.header)
    return sample

def mission_and_control(node,source=100.0):
    header=Header()
    header.stamp=stamp_from_seconds(source)
    node._receive_mission(NS(header=header,mode='REALTIME',initial_count=3,rescan_count=1,failed=False,error=''))
    node._receive_control(NS(header=header,session_id='controller-one',sequence=7,role='side',
        cleaner_duty_percent=25.0,pump_on=True,valid_until=stamp_from_seconds(source+0.4),reason='rust_detected'))
'''


class RosGuiOutputTests(unittest.TestCase):
    def run_contract(self, source):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(JETSON_CODE), environment.get("PYTHONPATH", "")))
        ros_doubles = ROS_DOUBLES.replace(
            "from node_runtime.ros import camera_node, vision_node, display_node",
            "from rail_robot_interfaces import msg as _rail_messages\n"
            "_rail_messages.FrameTrace = type('FrameTrace', (), {})\n"
            "from node_runtime.ros import camera_node, vision_node, display_node",
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", ros_doubles + GUI_DOUBLES + textwrap.dedent(source)],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_headless_full_gui_output_retains_distinct_exact_sources_and_composition_stamp(self):
        self.run_contract('''
            factory=Mock()
            with patch.object(display_node.time,'monotonic',return_value=10.2):
                node=display_node.DisplayNode(local_gui=False,display_factory=factory)
                node.now=100.2
                side=frame_message('side',1,10.0,value=30)
                front=frame_message('top',9,10.1,value=70)
                node._receive_frame('side',side)
                node._receive_frame('side',frame_message('side',2,10.2,value=110))
                node._receive_sample(raw_sample(side))
                node._receive_frame('top',front)
                mission_and_control(node)
                saved=next(iter(node._frames['side'].values())).frame.copy()
                real_encode=cv2.imencode
                def encode(*args):
                    node.now=100.5  # encoding completion must not replace composition stamp
                    return real_encode(*args)
                with patch.object(display_node.cv2,'imencode',side_effect=encode):node._render()
                factory.assert_not_called()
                image=node.publishers[node._output_topic].messages[-1]
                assert isinstance(image,CompressedImage)
                assert image.format=='bgr8; jpeg compressed bgr8'
                assert stamp_seconds(image.header.stamp)==100.2
                assert image.header.frame_id=='rail_robot_gui'
                decoded=cv2.imdecode(np.frombuffer(image.data,np.uint8),cv2.IMREAD_COLOR)
                assert decoded.shape==(360,1280,3)
                status=gui_status(node)
                assert status['schema_version']==1 and status['state']=='published'
                assert status['image_stamp_seconds']==stamp_seconds(image.header.stamp)
                assert status['sequence']==1 and status['width']==1280 and status['height']==360
                assert status['layout']=='camera_only'
                assert status['target_rate_hz']==12.0
                assert status['sources']['side']['sequence']==1
                assert status['sources']['side']['state']=='annotated'
                assert status['sources']['side']['detectors']==['rust']
                assert status['sources']['side']['source_stamp_seconds']==100.0
                assert status['sources']['front']['sequence']==9
                assert status['sources']['front']['source_stamp_seconds']==100.1
                assert status['sources']['front']['state']=='raw'
                assert status['sources']['front']['detectors']==[]
                assert abs(status['sources']['side']['source_age_ms']-200)<1e-6
                assert abs(status['sources']['front']['source_age_ms']-100)<1e-6
                assert status['mission']['mode']=='REALTIME' and status['mission']['fresh']
                assert status['control']['side']['valid'] and status['control']['side']['pump_on']
                assert status['control']['side']['reason']=='rust_detected'
                assert status['control']['front'] is None
                assert status['composition_ms']>=0 and status['encode_ms']>=0
                assert status['jpeg_bytes']==len(image.data)
                assert status['published_fps'] is None
                assert np.array_equal(next(iter(node._frames['side'].values())).frame,saved)
                assert abs(node._timer.interval-1/12)<1e-12
                for publisher in node.publishers.values():
                    assert publisher.qos is qos_sensor and publisher.qos.depth==1
                    assert publisher.qos.reliability==1
                assert 'runtime/fault' not in node.publishers
                node.destroy_node()
        ''')

    def test_disabled_frame_trace_keeps_gui_outputs_and_publishes_no_trace_topic(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False)
            assert node._profile_trace is False
            assert node._trace_output is None
            assert node._output_topic in node.publishers
            assert node._status_topic in node.publishers
            assert '/rail_robot/debug/frame_trace' not in node.publishers
            node.destroy_node()
        ''')

    def test_frame_trace_matches_composed_camera_models_and_gui_publish_boundaries(self):
        self.run_contract('''
            def timed(role, model, source, seq, stream, start, end):
                return NS(header=Header(), source_stamp=stamp_from_seconds(source+90.1),
                    camera_role=role, model=model, stream_id=stream, sequence=seq,
                    execution_id=f'{role}-{model}-{seq}', processing_start_monotonic_ns=start,
                    processing_end_monotonic_ns=end, processing_ms=(end-start)/1e6)
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            node.now=100.2
            side=frame_message('side',19,10.0,stream='s1',value=30)
            front=frame_message('top',41,10.1,stream='f1',value=70)
            side.capture_complete_monotonic_ns=1_000_000_000
            front.capture_complete_monotonic_ns=1_100_000_000
            side.header.frame_id='side-lens'
            front.header.frame_id='front-lens'
            with patch.object(display_node.time,'monotonic',return_value=10.2):
                node._receive_frame('side',side); node._receive_frame('top',front)
                for source,role,stream,seq,model in ((10.0,'side','s1',19,'rust'),
                        (10.0,'side','s1',19,'crack'),(10.1,'top','f1',41,'obstacle'),
                        (10.1,'top','f1',41,'crack')):
                    node._receive_sample(raw_sample(side if role=='side' else front,model))
                    timing=timed(role,model,source,seq,stream,1_200_000_000,1_230_000_000)
                    timing.header.stamp=stamp_from_seconds(source+90.1)
                    node._receive_timing(timing)
            ticks=iter((1_300_000_000,1_400_000_000,1_500_000_000,
                        1_600_000_000,1_700_000_000))
            with patch.object(display_node.time,'monotonic',return_value=10.2), \\
                 patch.object(display_node.time,'monotonic_ns',side_effect=lambda:next(ticks)):
                with patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1,2,3],np.uint8))):
                    node._render()
            trace=latest_trace(node)
            image=node.publishers[node._output_topic].messages[-1]
            assert trace.output_sequence==1 and trace.output_sequence==node._output_sequence
            assert (trace.front_stream_id,trace.front_frame_id,trace.front_camera_sequence,
                    trace.front_capture_complete_monotonic_ns)==('f1','front-lens',41,1_100_000_000), \\
                    ((trace.front_stream_id,trace.front_frame_id,trace.front_camera_sequence,
                      trace.front_capture_complete_monotonic_ns), gui_status(node)['sources'])
            assert (trace.side_stream_id,trace.side_frame_id,trace.side_camera_sequence,
                    trace.side_capture_complete_monotonic_ns)==('s1','side-lens',19,1_000_000_000)
            assert trace.header.frame_id==image.header.frame_id=='rail_robot_gui'
            assert stamp_seconds(trace.header.stamp)==stamp_seconds(image.header.stamp)
            assert list(trace.model_camera_roles)==['side','side','top','top']
            assert list(trace.model_ids)==['rust','crack','obstacle','crack']
            assert list(trace.model_source_sequences)==[19,19,41,41]
            assert list(trace.model_start_monotonic_ns)==[1_200_000_000]*4
            assert list(trace.model_end_monotonic_ns)==[1_230_000_000]*4
            assert list(trace.model_stale)==[False]*4
            assert (trace.compose_start_monotonic_ns,trace.compose_complete_monotonic_ns,
                    trace.jpeg_complete_monotonic_ns,trace.gui_publish_start_monotonic_ns,
                    trace.gui_publish_complete_monotonic_ns)==(1_300_000_000,1_400_000_000,
                    1_500_000_000,1_600_000_000,1_700_000_000)
            assert trace.gui_publish_complete_monotonic_ns>=trace.jpeg_complete_monotonic_ns
            assert trace.missing_camera_count==trace.stale_model_count==0
            node.destroy_node()
        ''')

    def test_trace_reports_missing_stale_and_stream_replacement_without_reusing_old_results(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            with patch.object(display_node.time,'monotonic',return_value=10.2), \\
                 patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                node._render()
            first=latest_trace(node)
            assert first.front_camera_sequence==first.side_camera_sequence==0
            assert first.missing_camera_count==2 and first.stale_model_count==4
            assert list(first.model_source_sequences)==[0,0,0,0]
            assert list(first.model_stale)==[True]*4
            old=frame_message('side',1,10.0,stream='old',value=30)
            old.capture_complete_monotonic_ns=11
            node.now=100.2
            new=frame_message('side',1,10.1,stream='new',value=90)
            new.capture_complete_monotonic_ns=22
            node.now=100.2
            with patch.object(display_node.time,'monotonic',return_value=10.2):
                node._receive_frame('side',old)
                node._receive_sample(raw_sample(old,'rust'))
                node._receive_frame('side',new)
                node._render()
            changed=latest_trace(node)
            assert changed.side_stream_id=='new' and changed.side_camera_sequence==1
            assert changed.side_capture_complete_monotonic_ns==22
            # The late old-stream rust result is not attached to the restarted source.
            assert list(changed.model_source_sequences)==[0,0,0,0]
            assert list(changed.model_stale)==[True]*4
            assert changed.missing_camera_count==1 and changed.stale_model_count==4
            node.destroy_node()
        ''')

    def test_late_model_result_is_traced_against_its_original_frame_not_latest_camera_frame(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            node.now=100.2
            older=frame_message('side',18,10.0,stream='s1',value=30)
            latest=frame_message('side',19,10.1,stream='s1',value=90)
            front=frame_message('top',41,10.1,stream='f1',value=70)
            for message,frame_id,stamp_ns in ((older,'side-lens',1_000_000_000),
                    (latest,'side-lens',1_100_000_000),(front,'front-lens',1_100_000_000)):
                message.header.frame_id=frame_id
                message.capture_complete_monotonic_ns=stamp_ns
            with patch.object(display_node.time,'monotonic',return_value=10.2):
                node._receive_frame('side',older)
                node._receive_frame('side',latest)
                node._receive_frame('top',front)
                # This result arrives after sequence 19, but is stamped for the retained sequence 18.
                node._receive_sample(raw_sample(older,'rust'))
            with patch.object(display_node.time,'monotonic',return_value=10.2), \\
                 patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                node._render()
            trace=latest_trace(node)
            status=gui_status(node)
            assert trace.side_camera_sequence==18
            assert trace.side_stream_id=='s1' and trace.side_frame_id=='side-lens'
            assert list(trace.model_source_sequences)==[18,0,0,0]
            assert list(trace.model_stream_ids)==['s1','','','']
            assert list(trace.model_stale)==[False,True,True,True]
            assert status['sources']['side']['sequence']==18
            node.destroy_node()
        ''')

    def test_expired_camera_and_pruned_model_result_are_reported_as_missing(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            node.now=100.2
            stale=frame_message('side',1,10.0,stream='s1')
            with patch.object(display_node.time,'monotonic',return_value=10.2):
                node._receive_frame('side',stale)
            node.now=101.2
            with patch.object(display_node.time,'monotonic',return_value=11.2), \\
                 patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                node._render()
            expired=latest_trace(node)
            assert expired.side_camera_sequence==0 and expired.side_stream_id==''
            assert expired.missing_camera_count==2
            # Keep an old same-stream result after its original frame falls out of the bounded cache.
            node.now=101.3
            with patch.object(display_node.time,'monotonic',return_value=11.3):
                for seq in range(2, node._frame_capacity + 4):
                    source_stamp=100.0 + seq / 100.0
                    node.now=source_stamp + 0.2
                    frame=frame_message('side',seq,source_stamp-90.0,stream='s1')
                    node._receive_frame('side',frame)
                    if seq==2:
                        node._receive_sample(raw_sample(frame,'rust'))
                node.now=101.3
            with patch.object(display_node.time,'monotonic',return_value=11.3), \\
                 patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                node._render()
            pruned=latest_trace(node)
            assert pruned.side_camera_sequence==node._frame_capacity+3
            assert list(pruned.model_source_sequences)==[0,0,0,0]
            assert list(pruned.model_stale)==[True]*4
            assert pruned.stale_model_count==4
            node.destroy_node()
        ''')

    def test_trace_counts_repeated_sources_sequence_gaps_and_busy_render_drops(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            def add(side_seq,front_seq,side_stream='s',front_stream='f',offset=0.0):
                node.now=100.3
                with patch.object(display_node.time,'monotonic',return_value=10.2), \\
                     patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                    node._receive_frame('side',frame_message('side',side_seq,10.0+offset,stream=side_stream))
                    node._receive_frame('top',frame_message('top',front_seq,10.0+offset,stream=front_stream))
                    node._render()
                return latest_trace(node)
            first=add(1,1)
            repeated=add(1,1,offset=0.01)
            assert first.duplicate_frame_count==0
            assert repeated.duplicate_frame_count==2
            gapped=add(4,3,offset=0.02)
            assert gapped.source_sequence_gap_count==3
            switched=add(1,1,side_stream='s2',front_stream='f2',offset=0.03)
            assert switched.source_sequence_gap_count==0
            node._render_guard.acquire()
            node._render()
            node._render_guard.release()
            after_drop=add(2,2,side_stream='s2',front_stream='f2',offset=0.04)
            assert after_drop.gui_render_drop_count==1
            node.destroy_node()
        ''')

    def test_trace_source_counters_compare_with_immediately_previous_published_frame(self):
        self.run_contract('''
            def trace_series(steps):
                node=display_node.DisplayNode(local_gui=False,profile_trace=True)
                output=[]
                for index,(sequence,stream) in enumerate(steps):
                    if sequence is None:
                        side={'fresh':False,'state':'waiting','stream_id':'','sequence':0,
                              'source_stamp_seconds':None,'detectors':[]}
                    else:
                        key=(stream,sequence,100.0+index)
                        node._frames['side'][key]=NS(read_completed_monotonic_ns=1000+index)
                        node._frame_ids['side'][key]=f'side-frame-{index}'
                        side={'fresh':True,'state':'raw','stream_id':stream,'sequence':sequence,
                              'source_stamp_seconds':key[2],'detectors':[]}
                    observations={'sources':{
                        'front':{'fresh':False,'state':'waiting','stream_id':'','sequence':0,
                                 'source_stamp_seconds':None,'detectors':[]},
                        'side':side},'mission':None}
                    trace,identities=node._build_frame_trace(
                        Header(),observations,index+1,10,20,30)
                    node._publish_frame_trace(trace,identities,40,50)
                    output.append(trace)
                node.destroy_node()
                return output

            repeated=trace_series([(10,'s'),(9,'s'),(9,'s')])
            assert repeated[-1].duplicate_frame_count==1
            assert repeated[1].source_sequence_gap_count==0

            returned=trace_series([(10,'s'),(9,'s'),(10,'s')])
            assert returned[-1].duplicate_frame_count==0
            assert returned[-1].source_sequence_gap_count==0

            gapped=trace_series([(10,'s'),(9,'s'),(12,'s')])
            assert gapped[1].source_sequence_gap_count==0
            assert gapped[-1].source_sequence_gap_count==2

            restarted=trace_series([(10,'old'),(1,'new'),(3,'new')])
            assert restarted[1].source_sequence_gap_count==0
            assert restarted[-1].source_sequence_gap_count==1

            placeholder=trace_series([(10,'s'),(None,'s'),(10,'s')])
            assert placeholder[-1].duplicate_frame_count==0
            assert placeholder[-1].source_sequence_gap_count==0
        ''')

    def test_trace_publisher_is_opt_in_and_default_gui_contract_is_unchanged(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False)
            assert '/rail_robot/debug/frame_trace' not in node.publishers
            node._render()
            assert len(node.publishers[node._output_topic].messages)==1
            assert '/rail_robot/debug/frame_trace' not in node.publishers
            node.destroy_node()
        ''')

    def test_trace_construction_or_publication_failure_does_not_drop_gui_jpeg(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            with patch.object(display_node.DisplayNode,'_build_frame_trace',side_effect=ValueError('trace')):
                with patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                    node._render()
            assert len(node.publishers[node._output_topic].messages)==1
            assert gui_status(node)['state']=='published'
            node.destroy_node()
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            node.publishers['/rail_robot/debug/frame_trace'].publish=Mock(side_effect=RuntimeError('trace'))
            with patch.object(display_node.cv2,'imencode',return_value=(True,np.array([1],np.uint8))):
                node._render()
            assert len(node.publishers[node._output_topic].messages)==1
            assert gui_status(node)['state']=='published'
            node.destroy_node()
        ''')

    def test_remote_jpeg_has_front_left_side_right_labels_and_local_preview_keeps_metrics(self):
        self.run_contract('''
            display=Mock(check_status=Mock(return_value=False),submit=Mock(return_value=True))
            with patch.object(display_node.time,'monotonic',return_value=10.0):
                node=display_node.DisplayNode(display_factory=lambda:display,gui_jpeg_quality=91,
                    gui_output_topic='/custom/gui/image',gui_status_topic='/custom/gui/status',gui_frame_id='gui_fixed')
                for role,value in (('side',30),('top',90)):
                    node._receive_frame(role,frame_message(role,1,10.0,value=value))
                node._receive_fps('side',NS(data=29.5))
                node._receive_fps('top',NS(data=30.0))
                real_encode=cv2.imencode
                received=[]
                def encode(extension,canvas,parameters):
                    received.append(canvas.copy())
                    assert parameters==[cv2.IMWRITE_JPEG_QUALITY,91]
                    return real_encode(extension,canvas,parameters)
                with patch.object(display_node.cv2,'imencode',side_effect=encode), \
                     patch.object(display_node.cv2,'putText',wraps=cv2.putText) as put_text:
                    node._render()
                canvas=display.submit.call_args.args[0]
                remote=received[0]
                assert remote.shape==(360,1280,3)
                # Full panes stay unchanged outside each small heading overlay.
                assert np.array_equal(canvas[112:436,640:],remote[36:,:640])
                assert np.array_equal(canvas[112:436,:640],remote[36:,640:])
                assert np.all(remote[:36,100:640]==90)
                assert np.all(remote[:36,740:]==30)
                for label,x in (('FRONT',10),('SIDE',650)):
                    assert any(call.args[1]==label and call.args[2]==(x,24)
                               for call in put_text.call_args_list)
                    assert np.max(remote[8:30,x:x+70])>200
                jpeg=node.publishers[node._output_topic].messages[-1]
                decoded=cv2.imdecode(np.frombuffer(jpeg.data,np.uint8),cv2.IMREAD_COLOR)
                assert decoded.shape==(360,1280,3)
                assert np.max(np.abs(decoded[100,100].astype(int)-90))<=2
                assert np.max(np.abs(decoded[100,740].astype(int)-30))<=2
                assert canvas.shape==(720,1280,3)
                assert np.all(canvas[76:436,:640]==30)
                assert np.all(canvas[76:436,640:]==90)
                assert canvas[-70,15].tolist()==[20,20,20]
                assert gui_status(node)['camera_fps']=={'side':29.5,'front':30.0}
                assert node.publishers['/custom/gui/image'].messages[-1].header.frame_id=='gui_fixed'
                node.destroy_node()
                display.close.assert_called_once()
        ''')

    def test_capture_modes_keep_live_camera_under_yellow_overlay_and_return_modes_hide_it(self):
        self.run_contract('''
            display=Mock(check_status=Mock(return_value=False),submit=Mock(return_value=True))
            with patch.object(display_node.time,'monotonic',return_value=10.0):
                node=display_node.DisplayNode(display_factory=lambda:display)
                for role,value in (('side',30),('top',90)):
                    node._receive_frame(role,frame_message(role,1,10.0,value=value))
                node._render()
                original=bytes(node.publishers[node._output_topic].messages[-1].data)
                header=Header()
                header.stamp=stamp_from_seconds(100.0)
                for mode,label,color,live in (
                    ('CAPTURE_SCAN','capture mode',(0,255,255),True),
                    ('RETURN','return',(0,165,255),False),
                    ('REALTIME',None,None,True),
                    ('RESCAN_RETURN','return',(0,165,255),False),
                    ('RESCAN','capture mode',(0,255,255),True),
                    ('REALTIME',None,None,True),
                ):
                    node._receive_mission(NS(header=header,mode=mode,initial_count=0,rescan_count=0,
                        failed=False,error=''))
                    with patch.object(display_node.cv2,'putText',wraps=cv2.putText) as put_text:
                        node._render()
                    image=node.publishers[node._output_topic].messages[-1]
                    decoded=cv2.imdecode(np.frombuffer(image.data,np.uint8),cv2.IMREAD_COLOR)
                    local=display.submit.call_args.args[0]
                    assert decoded.shape==(360,1280,3) and local.shape==(720,1280,3)
                    assert gui_status(node)['mission']['mode']==mode
                    if live:
                        if label is None:
                            assert bytes(image.data)==original, mode
                        else:
                            assert any(call.args[1]==label for call in put_text.call_args_list)
                            assert np.array_equal(local[82,100],color), mode
                            assert np.max(np.abs(decoded[6,100].astype(int)-np.array(color)))<=8, mode
                        assert np.all(local[180:240,40:600]==30), mode
                        assert np.all(local[180:240,680:1240]==90), mode
                        assert np.max(np.abs(decoded[100,100].astype(int)-90))<=2
                        assert np.max(np.abs(decoded[100,740].astype(int)-30))<=2
                    else:
                        assert any(call.args[1]==label for call in put_text.call_args_list)
                        assert np.array_equal(local[82,100],color), mode
                        assert np.max(np.abs(decoded[6,100].astype(int)-np.array(color)))<=8, mode
                        assert np.max(decoded[40:100,40:-40])<=3, mode
                        assert np.count_nonzero(decoded[140:220,400:880]>50)>500, mode
                        assert not any(call.args[1] in ('FRONT','SIDE') and call.args[2][1]==24
                                       for call in put_text.call_args_list), mode
                        assert np.all(local[116:176,40:-40]==0), mode
                node.destroy_node()
        ''')

    def test_stale_capture_status_keeps_capture_overlay_and_ignores_old_realtime_mode(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False)
            with patch.object(display_node.time,'monotonic',side_effect=lambda:node.now-90.0):
                header=Header()
                header.stamp=stamp_from_seconds(100.0)
                node._receive_mission(NS(header=header,mode='CAPTURE_SCAN',initial_count=0,rescan_count=0,
                    failed=False,error=''))
                node._render()
                capture=bytes(node.publishers[node._output_topic].messages[-1].data)
                node.now=102.0
                for role,value in (('side',30),('top',90)):
                    node._receive_frame(role,frame_message(role,2,12.0,value=value))
                header=Header()
                header.stamp=stamp_from_seconds(99.9)
                node._receive_mission(NS(header=header,mode='REALTIME',initial_count=0,rescan_count=0,
                    failed=False,error=''))
                node._render()
                status=gui_status(node)
                assert status['mission']['mode']=='CAPTURE_SCAN' and not status['mission']['fresh']
                assert status['sources']['side']['fresh'] and status['sources']['front']['fresh']
                image=node.publishers[node._output_topic].messages[-1]
                decoded=cv2.imdecode(np.frombuffer(image.data,np.uint8),cv2.IMREAD_COLOR)
                assert bytes(image.data)!=capture
                assert np.max(np.abs(decoded[100,100].astype(int)-90))<=2
                assert np.max(np.abs(decoded[100,740].astype(int)-30))<=2
                assert np.max(np.abs(decoded[6,100].astype(int)-np.array((0,255,255))))<=8
                node.destroy_node()
        ''')

    def test_capture_frame_trace_identifies_live_sources_in_remote_jpeg(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False,profile_trace=True)
            node.now=100.2
            with patch.object(display_node.time,'monotonic',return_value=10.2):
                side=frame_message('side',3,10.1,stream='side-stream',value=30)
                front=frame_message('top',4,10.1,stream='front-stream',value=90)
                side.header.frame_id='side-lens'
                front.header.frame_id='front-lens'
                node._receive_frame('side',side)
                node._receive_frame('top',front)
                header=Header()
                header.stamp=stamp_from_seconds(100.0)
                node._receive_mission(NS(header=header,mode='CAPTURE_SCAN',initial_count=0,
                    rescan_count=0,failed=False,error=''))
                node._render()
            trace=latest_trace(node)
            assert (trace.front_stream_id,trace.front_camera_sequence)==('front-stream',4)
            assert (trace.side_stream_id,trace.side_camera_sequence)==('side-stream',3)
            assert trace.missing_camera_count==0
            node.destroy_node()
        ''')

    def test_busy_encoder_skips_concurrent_renders_and_keeps_only_bounded_latest_inputs(self):
        self.run_contract('''
            entered,release=threading.Event(),threading.Event()
            node=display_node.DisplayNode(local_gui=False)
            with patch.object(display_node.time,'monotonic',side_effect=lambda:node.now-90.0):
                for sequence in range(1,61):
                    node.now=100+sequence/100
                    node._receive_frame('side',frame_message('side',sequence,node.now-90,value=sequence))
                assert len(node._frames['side'])==node._frame_capacity
                real_encode=cv2.imencode
                calls=[]
                def encode(*args):
                    calls.append(args[1].copy())
                    entered.set()
                    assert release.wait(5)
                    return real_encode(*args)
                with patch.object(display_node.cv2,'imencode',side_effect=encode):
                    worker=threading.Thread(target=node._render)
                    worker.start()
                    assert entered.wait(3)
                    node.now=100.7
                    node._receive_frame('side',frame_message('side',61,10.7,value=61))
                    for _ in range(100):node._render()
                    assert len(calls)==1
                    release.set()
                    worker.join(5)
                    assert not worker.is_alive()
                assert gui_status(node)['sources']['side']['sequence']==60
                node.now=100.8
                node._render()
                assert len(node.publishers[node._output_topic].messages)==2
                status=gui_status(node)
                assert status['sources']['side']['sequence']==61
                assert abs(status['published_fps']-10.0)<1e-5
                assert len(node._frames['side'])<=node._frame_capacity
                node.destroy_node()
        ''')

    def test_old_sources_remain_identified_as_stale_and_never_gain_fresh_overlays(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False)
            with patch.object(display_node.time,'monotonic',side_effect=lambda:node.now-90.0):
                message=frame_message('side',7,10.0)
                node._receive_frame('side',message)
                node._receive_sample(raw_sample(message))
                mission_and_control(node)
                node._receive_fps('side',NS(data=30.0))
                node._render()
                node.now=102.0
                node._render()
                status=gui_status(node)
                side=status['sources']['side']
                assert side['state']=='stale' and side['sequence']==7
                assert side['stream_id']=='camera-session' and side['source_stamp_seconds']==100.0
                assert side['source_age_ms']==2000.0 and side['detectors']==[]
                assert not side['fresh']
                assert status['sources']['front']['state']=='waiting'
                assert status['sources']['front']['source_stamp_seconds'] is None
                assert status['image_stamp_seconds']==102.0
                assert not status['mission']['fresh'] and status['mission']['age_ms']==2000.0
                assert not status['control']['side']['valid']
                assert status['camera_fps']['side'] is None
                decoded=cv2.imdecode(np.frombuffer(node.publishers[node._output_topic].messages[-1].data,np.uint8),cv2.IMREAD_COLOR)
                assert np.max(decoded[90:420])<=3
                node.destroy_node()
        ''')

    def test_encode_and_local_gui_failures_do_not_stop_presentation_or_fault_core(self):
        self.run_contract('''
            display=Mock(check_status=Mock(return_value=False),submit=Mock(return_value=True))
            with patch.object(display_node.time,'monotonic',return_value=10.0):
                node=display_node.DisplayNode(display_factory=lambda:display)
                with patch.object(display_node.cv2,'imencode',return_value=(False,None)):node._render()
                status=gui_status(node)
                assert status['state']=='error' and 'encode' in status['error']
                assert not node.publishers[node._output_topic].messages
                assert not node.shutdown_requested()
                node._timer.cancel.assert_not_called()
                display.submit.assert_called_once()
                display.check_status.side_effect=RuntimeError('GUI fake failure')
                node._render()
                assert gui_status(node)['state']=='published'
                assert node._display is None and not node.shutdown_requested()
                assert 'runtime/fault' not in node.publishers
                node._render()
                assert len(node.publishers[node._output_topic].messages)==2
                node.destroy_node()
                display.close.assert_called_once()
                headless=display_node.DisplayNode(display_factory=Mock(side_effect=RuntimeError('No DISPLAY')))
                headless._render()
                assert len(headless.publishers[headless._output_topic].messages)==1
                headless.destroy_node()
        ''')

    def test_output_toggle_and_local_gui_toggle_are_independent(self):
        self.run_contract('''
            display=Mock(check_status=Mock(return_value=False),submit=Mock(return_value=True))
            node=display_node.DisplayNode(gui_output=False,local_gui=True,display_factory=lambda:display)
            with patch.object(display_node.cv2,'imencode') as encode:node._render()
            encode.assert_not_called()
            display.submit.assert_called_once()
            assert not node.publishers
            display.check_status.return_value=True
            node._render()
            assert node.shutdown_requested()
            node.destroy_node()
            display.close.assert_called_once()
        ''')

    def test_shutdown_during_encoding_drops_completion_and_releases_gui_once(self):
        self.run_contract('''
            entered,release=threading.Event(),threading.Event()
            display=Mock(check_status=Mock(return_value=False),submit=Mock(return_value=True))
            node=display_node.DisplayNode(display_factory=lambda:display)
            original=cv2.imencode
            def encode(*args):
                entered.set()
                assert release.wait(5)
                return original(*args)
            with patch.object(display_node.cv2,'imencode',side_effect=encode):
                worker=threading.Thread(target=node._render)
                worker.start()
                assert entered.wait(3)
                node.prepare_shutdown()
                node.destroy_node()
                release.set()
                worker.join(5)
                assert not worker.is_alive()
            assert not node.publishers[node._output_topic].messages
            assert not node.publishers[node._status_topic].messages
            display.submit.assert_not_called()
            display.close.assert_called_once()
        ''')

    def test_invalid_configuration_fails_before_opening_gui(self):
        self.run_contract('''
            for values in ({'gui_publish_rate_hz':9.9},{'gui_publish_rate_hz':15.1},
                           {'gui_publish_rate_hz':True},{'gui_jpeg_quality':0},{'gui_jpeg_quality':80.0},
                           {'gui_frame_id':'bad frame'},{'gui_output_topic':'/same','gui_status_topic':'/same'},
                           {'local_gui':'false'},{'gui_output':'true'}):
                factory=Mock()
                try:display_node.DisplayNode(display_factory=factory,**values)
                except ValueError:pass
                else:raise AssertionError(values)
                factory.assert_not_called()
            for rate in (10.0,15.0):
                node=display_node.DisplayNode(local_gui=False,gui_publish_rate_hz=rate)
                assert node._timer.interval==1/rate
                node.destroy_node()
        ''')

    def test_bad_overlay_is_isolated_to_its_source_without_stopping_other_feed(self):
        self.run_contract('''
            node=display_node.DisplayNode(local_gui=False)
            with patch.object(display_node.time,'monotonic',return_value=10.0):
                original=frame_message('side',1,10.0,value=30)
                node._receive_frame('side',original)
                sample=raw_sample(original)
                sample.mask=encode_image(np.zeros((10,10),np.uint8),sample.header)
                node._receive_sample(sample)
                node._receive_frame('top',frame_message('top',2,10.0,value=90))
                node._render()
                status=gui_status(node)
                assert status['state']=='published'
                assert status['sources']['side']['state']=='error'
                assert not status['sources']['side']['fresh']
                assert 'mask' in status['sources']['side']['error']
                assert status['sources']['front']['state']=='raw' and status['sources']['front']['fresh']
                assert 'runtime/fault' not in node.publishers
                assert not node.shutdown_requested()
                node._timer.cancel.assert_not_called()
                node.destroy_node()
        ''')
