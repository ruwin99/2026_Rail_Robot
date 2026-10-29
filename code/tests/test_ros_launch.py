"""Launch wiring and default shell routing; no ROS or hardware processes run."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1]


class Action:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)


class LaunchNode(Action):
    pass


class Configuration:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context[self.name]


def load_launch():
    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        return result
    doubles = {
        'launch': module('launch', LaunchDescription=Action),
        'launch.actions': module('launch.actions', DeclareLaunchArgument=Action, EmitEvent=Action,
                                 OpaqueFunction=Action, RegisterEventHandler=Action),
        'launch.event_handlers': module('launch.event_handlers', OnProcessExit=Action),
        'launch.events': module('launch.events', Shutdown=Action),
        'launch.substitutions': module('launch.substitutions', LaunchConfiguration=Configuration),
        'launch_ros': module('launch_ros'),
        'launch_ros.actions': module('launch_ros.actions', Node=LaunchNode),
    }
    path = CODE/'ros2/rail_robot_nodes/launch/robot.launch.py'
    spec = importlib.util.spec_from_file_location('rail_launch_test', path)
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, doubles):
        spec.loader.exec_module(result)
    return result


class RosLaunchTests(unittest.TestCase):
    def setUp(self):
        self.launch = load_launch()
        self.context = dict(mode='mission', camera='true', display='true', upload='false',
                            namespace='', params_file='', side_camera_device='', top_camera_device='',
                            serial_port='', uart_protocol='', output_directory='', training_camera_role='',
                            training_output_directory='', capture_output_directory='',
                            gui_output='true', firebase_snapshot='false', dashboard_live='false', dashboard_control='false', gui_output_topic='',
                            gui_status_topic='', gui_frame_id='', gui_publish_rate_hz='')
        self.context.update(audio_enabled='true', audio_directory='', audio_device='',
                            profiling='false', profiling_output_directory='',
                            frame_trace_topic='/rail_robot/debug/frame_trace')

    def nodes(self, **changes):
        self.context.update(changes)
        actions = self.launch.configured_nodes(self.context)
        return [action for action in actions if isinstance(action, LaunchNode)], actions

    def test_mission_includes_all_owners_and_failure_shutdown(self):
        nodes, actions = self.nodes()
        self.assertEqual({n.executable for n in nodes}, {
            'camera_node','vision_node','control_node','stm_bridge_node',
            'inspection_node','mission_node','display_node','audio_alert_node'})
        protected = [action.args[0].target_action for action in actions if not isinstance(action, LaunchNode)]
        self.assertEqual({n.executable for n in protected}, {n.executable for n in nodes} - {'display_node', 'audio_alert_node'})
        self.assertTrue(all('autostart' not in str(n.parameters) for n in nodes))

    def test_profiling_is_opt_in_and_preserves_mission_required_roles(self):
        required = {'camera_node', 'vision_node', 'control_node', 'stm_bridge_node',
                    'inspection_node', 'mission_node'}
        default_nodes, _ = self.nodes()
        self.assertNotIn('frame_profiler_node', {node.executable for node in default_nodes})
        cmake = (CODE/'ros2/rail_robot_nodes/CMakeLists.txt').read_text(encoding='utf-8')
        self.assertIn('frame_profiler)', cmake)
        default_display = next(node for node in default_nodes if node.executable == 'display_node')
        self.assertFalse(default_display.parameters[-1]['profile_trace'])
        self.assertEqual(default_display.remappings, [])

        nodes, actions = self.nodes(
            profiling='true', profiling_output_directory='/tmp/frame-profiles',
            frame_trace_topic='/rail_a/debug/frame_trace',
        )
        by_name = {node.executable: node for node in nodes}
        self.assertTrue(required.issubset({node.executable for node in default_nodes}))
        self.assertTrue(required.issubset(set(by_name)))
        self.assertEqual(set(by_name) - {'frame_profiler_node'},
                         {node.executable for node in default_nodes})
        self.assertTrue(by_name['display_node'].parameters[-1]['profile_trace'])
        self.assertEqual(by_name['display_node'].remappings,
                         [('/rail_robot/debug/frame_trace', '/rail_a/debug/frame_trace')])
        profiler = by_name['frame_profiler_node']
        self.assertEqual(profiler.parameters[-1], {'output_directory': '/tmp/frame-profiles'})
        self.assertEqual(profiler.remappings,
                         [('/rail_robot/debug/frame_trace', '/rail_a/debug/frame_trace')])
        shutdown_targets = {action.args[0].target_action.executable for action in actions
                            if not isinstance(action, LaunchNode)}
        self.assertNotIn('frame_profiler_node', shutdown_targets)

    def test_profiling_launch_argument_defaults_off_and_invalid_without_composed_output(self):
        description = self.launch.generate_launch_description()
        declarations = description.args[0]
        profiling = next(argument for argument in declarations if argument.args == ('profiling',))
        self.assertEqual(profiling.default_value, 'false')
        for changes in (
            {'mode': 'dual_camera', 'gui_output': 'false', 'profiling': 'true'},
            {'mode': 'front_camera', 'profiling': 'true'},
            {'mode': 'monitor', 'profiling': 'true'},
        ):
            original = self.context.copy()
            with self.assertRaises(ValueError):
                self.nodes(**changes)
            self.context = original

    def test_uart_v2_reaches_only_the_mission_bridge_in_mission_mode(self):
        nodes, _ = self.nodes(uart_protocol='v2')
        bridge = next(n for n in nodes if n.executable == 'stm_bridge_node')
        self.assertIn({'uart_protocol': 'v2'}, bridge.parameters)
        self.assertNotIn('uart_protocol', str(next(n for n in nodes if n.executable == 'mission_node').parameters))

    def test_uart_protocol_defaults_to_v2_and_reaches_each_serial_owner(self):
        declarations = self.launch.generate_launch_description().args[0]
        protocol = next(argument for argument in declarations if argument.args == ('uart_protocol',))
        self.assertEqual(protocol.default_value, 'v2')
        for mode, owner in (('mission', 'stm_bridge_node'),
                            ('motor_encoder', 'actuator_debug_node'),
                            ('dc_motor', 'actuator_debug_node'),
                            ('water_pump', 'actuator_debug_node'),
                            ('actuators', 'actuator_debug_node'),
                            ('led', 'sensor_debug_node'),
                            ('ultrasonic', 'sensor_debug_node'),
                            ('sensors', 'sensor_debug_node')):
            for selected in ('v2', 'legacy'):
                nodes, _ = self.nodes(mode=mode, uart_protocol=selected)
                serial_owners = [node for node in nodes if node.executable in
                                 ('stm_bridge_node', 'actuator_debug_node', 'sensor_debug_node')]
                self.assertEqual([node.executable for node in serial_owners], [owner])
                self.assertIn({'uart_protocol': selected}, serial_owners[0].parameters)
        with self.assertRaisesRegex(ValueError, 'uart_protocol'):
            self.nodes(mode='mission', uart_protocol='unknown')

    def test_audio_paths_and_explicit_opt_out(self):
        nodes, _ = self.nodes(audio_directory='/home/robot/audio', audio_device='plughw:CARD=USB,DEV=0')
        audio = next(n for n in nodes if n.executable == 'audio_alert_node')
        self.assertEqual(audio.parameters[0]['audio_directory'], '/home/robot/audio')
        mission = next(n for n in nodes if n.executable == 'mission_node')
        self.assertTrue(mission.parameters[-1]['audio_enabled'])
        nodes, _ = self.nodes(audio_enabled='false')
        self.assertNotIn('audio_alert_node', {n.executable for n in nodes})
        self.assertFalse(next(n for n in nodes if n.executable == 'mission_node').parameters[-1]['audio_enabled'])

    def test_mission_waits_for_report_upload_only_when_upload_is_enabled(self):
        # 완료 뒤 스택을 스스로 종료할 때 업로드 노드를 끊지 않도록 임무 노드가 업로드 결과를 기다릴지 정한다.
        for upload, expected in (('false', False), ('true', True)):
            nodes, _ = self.nodes(upload=upload)
            mission = next(n for n in nodes if n.executable == 'mission_node')
            self.assertIs(mission.parameters[-1]['wait_for_upload'], expected)
            self.assertEqual(('upload_node' in {n.executable for n in nodes}), expected)

    def test_debug_modes_never_open_uart_or_start_mission(self):
        for mode, expected in (
            ('vision', {'camera_node','vision_node'}),
            ('camera', {'camera_node'}),
            ('training', {'camera_node','training_capture_node'}),
        ):
            nodes, _ = self.nodes(mode=mode, display='false', gui_output='false')
            self.assertEqual({n.executable for n in nodes}, expected)

    def test_monitor_does_not_start_hardware_with_default_launch_options(self):
        nodes, actions = self.nodes(mode='monitor')
        self.assertEqual([n.executable for n in nodes], ['monitor_node'])
        self.assertEqual(len(actions), 1)

    def test_sensor_modes_only_start_one_uart_owner(self):
        for mode in ('led', 'ultrasonic', 'sensors'):
            nodes, _ = self.nodes(mode=mode, serial_port='/dev/ttyACM9')
            self.assertEqual([n.executable for n in nodes], ['sensor_debug_node'])
            self.assertIn({'serial_port': '/dev/ttyACM9'}, nodes[0].parameters)
        with self.assertRaises(ValueError):
            self.nodes(mode='sensors', firebase_snapshot='true')

    def test_live_dashboard_is_opt_in_and_cannot_stop_control(self):
        nodes, actions = self.nodes(dashboard_live='true')
        telemetry = next(n for n in nodes if n.executable == 'dashboard_telemetry_node')
        inspection = next(n for n in nodes if n.executable == 'inspection_node')
        self.assertEqual(inspection.parameters[-1], {'dashboard_live': True})
        protected = [a.args[0].target_action for a in actions if not isinstance(a, LaunchNode)]
        self.assertNotIn(telemetry, protected)

    def test_live_dual_starts_only_video_and_optional_telemetry(self):
        nodes, actions = self.nodes(mode='dual_camera', display='false',
                                    dashboard_live='true', namespace='rail_a')
        self.assertEqual({n.executable for n in nodes}, {
            'camera_node', 'vision_node', 'display_node', 'dashboard_telemetry_node'})
        self.assertTrue(all(n.namespace == 'rail_a' for n in nodes))
        protected = [a.args[0].target_action.executable for a in actions
                     if not isinstance(a, LaunchNode)]
        self.assertEqual(set(protected), {'camera_node', 'vision_node'})

    def test_live_capture_default_is_forced_off_after_parameter_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'params.yaml'
            path.write_text('/inspection:\n  ros__parameters:\n    dashboard_live: true\n')
            nodes, _ = self.nodes(params_file=str(path))
            inspection = next(n for n in nodes if n.executable == 'inspection_node')
            self.assertEqual(inspection.parameters[-1], {'dashboard_live': False})
            self.assertNotIn('dashboard_telemetry_node', {n.executable for n in nodes})

    def test_external_camera_namespace_and_overrides(self):
        nodes, _ = self.nodes(mode='vision', camera='false', namespace='rail_a')
        self.assertEqual({n.executable for n in nodes}, {'vision_node','display_node'})
        self.assertTrue(all(n.namespace == 'rail_a' for n in nodes))
        nodes, _ = self.nodes(mode='mission', camera='true', serial_port='/dev/test',
                              side_camera_device='/dev/side', output_directory='/tmp/reports')
        by_name = {n.executable: n for n in nodes}
        self.assertEqual(by_name['stm_bridge_node'].parameters[-1], {'serial_port':'/dev/test'})
        self.assertIn({'side_camera_device':'/dev/side'}, by_name['camera_node'].parameters)
        self.assertEqual(by_name['vision_node'].parameters, [])

    def test_invalid_mode_boolean_or_upload_combination_fails_early(self):
        for changes in ({'mode':'unknown'}, {'camera':'yes'}, {'mode':'camera','upload':'true'},
                        {'mode':'camera','camera':'false','display':'false','gui_output':'false'},
                        {'gui_publish_rate_hz':'nan'}, {'gui_publish_rate_hz':'30'},
                        {'mode':'front_camera','firebase_snapshot':'true'},
                        {'gui_output':'false','firebase_snapshot':'true'}):
            original = self.context.copy()
            with self.assertRaises(ValueError):
                self.nodes(**changes)
            self.context = original

    def test_parameter_file_applies_to_every_node(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'params.yaml'
            path.write_text('/**:\n  ros__parameters: {}\n')
            nodes, _ = self.nodes(params_file=str(path))
            self.assertTrue(all(n.parameters[0] == str(path.resolve()) for n in nodes))

    def test_actuator_modes_ignore_camera_and_gui_defaults(self):
        for mode in ('motor_encoder', 'dc_motor', 'water_pump', 'actuators'):
            nodes, _ = self.nodes(mode=mode, serial_port='/dev/test')
            self.assertEqual([n.executable for n in nodes], ['actuator_debug_node'])
            self.assertEqual(nodes[0].name, 'rail_actuator_debug')
            self.assertEqual(nodes[0].parameters[-1], {'debug_capability': mode})
            self.assertIn({'serial_port': '/dev/test'}, nodes[0].parameters)

    def test_capture_mode_enforces_one_camera_and_one_capture_gui(self):
        for role in ('front', 'side'):
            nodes, _ = self.nodes(mode=role + '_camera', capture_output_directory='/tmp/captures')
            by_name = {n.executable: n for n in nodes}
            self.assertEqual(set(by_name), {'camera_node', 'capture_debug_node'})
            self.assertEqual(by_name['camera_node'].parameters[-1], {'camera_roles': role})
            self.assertEqual(by_name['capture_debug_node'].parameters[-1], {'capture_role': role, 'show_gui': True})
            self.assertIn({'capture_output_directory': '/tmp/captures'}, by_name['capture_debug_node'].parameters)
        nodes, _ = self.nodes(mode='front_camera', display='false', camera='false')
        self.assertEqual([n.executable for n in nodes], ['capture_debug_node'])
        self.assertFalse(nodes[0].parameters[-1]['show_gui'])

    def test_foreign_capture_uses_top_camera_and_foreign_capture_role(self):
        nodes, _ = self.nodes(mode='foreign_camera', capture_output_directory='/tmp/captures')
        by_name = {n.executable: n for n in nodes}
        self.assertEqual(set(by_name), {'camera_node', 'capture_debug_node'})
        self.assertEqual(by_name['camera_node'].parameters[-1], {'camera_roles': 'front'})
        self.assertEqual(by_name['capture_debug_node'].parameters[-1],
                         {'capture_role': 'foreign', 'show_gui': True})
        self.assertIn({'capture_output_directory': '/tmp/captures'},
                      by_name['capture_debug_node'].parameters)

    def test_dual_mode_matches_default_video_test_without_uart(self):
        nodes, _ = self.nodes(mode='dual_camera')
        self.assertEqual({n.executable for n in nodes}, {'camera_node', 'vision_node', 'display_node'})
        camera = next(n for n in nodes if n.executable == 'camera_node')
        self.assertEqual(camera.parameters[-1], {'camera_roles': 'both'})

    def test_headless_gui_output_and_optional_firebase_are_independent_of_control(self):
        nodes, actions = self.nodes(mode='dual_camera', display='false', firebase_snapshot='true',
                                    gui_output_topic='/rail_a/gui/image/compressed', gui_publish_rate_hz='12.0')
        by_name = {node.executable: node for node in nodes}
        self.assertEqual(set(by_name), {'camera_node','vision_node','display_node','firebase_snapshot_node'})
        self.assertEqual(by_name['display_node'].parameters[-1],
                         {'local_gui': False, 'gui_output': True, 'profile_trace': False, 'gui_publish_rate_hz': 12.0})
        for role in ('display_node', 'firebase_snapshot_node'):
            self.assertIn({'gui_output_topic': '/rail_a/gui/image/compressed'}, by_name[role].parameters)
        protected = [action.args[0].target_action.executable for action in actions if not isinstance(action, LaunchNode)]
        self.assertEqual(set(protected), {'camera_node','vision_node'})
        nodes, _ = self.nodes(mode='dual_camera', firebase_snapshot='false', gui_output='false')
        self.assertEqual({node.executable for node in nodes}, {'camera_node','vision_node'})

    def test_local_gui_can_run_without_compressed_output(self):
        nodes, _ = self.nodes(mode='camera', gui_output='false')
        display = next(n for n in nodes if n.executable == 'display_node')
        self.assertEqual(display.parameters[-1], {'local_gui': True, 'gui_output': False, 'profile_trace': False})

    def test_mode_constraints_are_last_and_target_named_nodes_after_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'conflicting.yaml'
            path.write_text('/rail_a/rail_camera:\n  ros__parameters:\n    camera_roles: side\n')
            nodes, _ = self.nodes(mode='dual_camera', namespace='rail_a', params_file=str(path))
            camera = next(n for n in nodes if n.executable == 'camera_node')
            self.assertEqual(camera.name, 'rail_camera')
            self.assertEqual(camera.namespace, 'rail_a')
            self.assertEqual(camera.parameters, [str(path.resolve()), {'camera_roles': 'both'}])

    def test_python_frontend_routes_to_ros_and_rejects_old_flags(self):
        path = CODE/'jetson_code/node_runtime/ros/launcher.py'
        spec = importlib.util.spec_from_file_location('rail_frontend_test',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module.subprocess,'call',return_value=0) as run:
            self.assertEqual(module.main(['mode:=vision','display:=false']),0)
            run.assert_called_once_with(['ros2','launch','rail_robot_nodes','robot.launch.py',
                                         'mode:=vision','display:=false'])
            run.reset_mock()
            self.assertEqual(module.main(['--no-uart']),2)
            run.assert_not_called()


class RosShellRoutingTests(unittest.TestCase):
    def run_wrapper(self, arguments):
        bash = Path(r'C:\Program Files\Git\bin\bash.exe')
        bash = str(bash) if bash.is_file() else shutil.which('bash')
        if not bash:
            self.skipTest('bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, binary = root/'scripts', root/'bin'
            scripts.mkdir()
            binary.mkdir()
            wrapper = scripts/'run.sh'
            wrapper.write_bytes((CODE/'scripts/run.sh').read_bytes())
            ros2 = binary/'ros2'
            ros2.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "${ROUTE_LOG}"\n',encoding='utf-8')
            ros2.chmod(0o755)
            log = root/'route.txt'
            env = os.environ.copy()
            env.update(PATH=str(binary)+os.pathsep+env['PATH'], ROUTE_LOG=log.as_posix(),
                       RAIL_ROBOT_ROS_SETUP=str(root/'not-installed.bash'))
            result = subprocess.run([bash,str(wrapper),*arguments], env=env, capture_output=True,text=True,timeout=15)
            return result, log.read_text().splitlines() if log.exists() else []

    def test_default_uses_ros_without_legacy_python_environment(self):
        result, args = self.run_wrapper([])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(args,['launch','rail_robot_nodes','robot.launch.py'])

    def test_mode_and_display_are_forwarded(self):
        result, args = self.run_wrapper(['mode:=vision','display:=false'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(args[-2:],['mode:=vision','display:=false'])

    def test_live_dashboard_option_is_forwarded(self):
        result, args = self.run_wrapper(['mode:=dual_camera', 'dashboard_live:=true'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[-2:], ['mode:=dual_camera', 'dashboard_live:=true'])

    def test_legacy_test_flags_do_not_accidentally_launch_mission(self):
        result, args = self.run_wrapper(['--realtime-test'])
        self.assertEqual(result.returncode,2)
        self.assertEqual(args,[])


if __name__ == '__main__':
    unittest.main()
