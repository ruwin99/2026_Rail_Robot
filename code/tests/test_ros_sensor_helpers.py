"""Short commands route to the real ROS CLI without contacting a device."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1]


class HelpersTests(unittest.TestCase):
    def run_helper(self, command, setup='true\n'):
        bash = Path('C:/Program Files/Git/bin/bash.exe')
        executable = str(bash) if bash.is_file() else shutil.which('bash')
        if not executable:
            self.skipTest('bash is unavailable')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'ros2/install').mkdir(parents=True)
            (root/'ros2/install/setup.bash').write_text(setup)
            script = root/'helpers.sh'
            script.write_bytes((CODE/'scripts/ros_sensor_helpers.sh').read_bytes())
            env = os.environ.copy()
            env.update(RAIL_ROBOT_HOME=root.as_posix(), RAIL_ROBOT_NAMESPACE='bench',
                       RAIL_ROBOT_ROS_SETUP=(root/'ros2/install/setup.bash').as_posix(),
                       RAIL_ROBOT_SERIAL_PORT='/dev/ttyACM8')
            source = ('ros2() { printf "%s\\n" "$@"; }; python3() { printf "%s\\n" "$@"; }; '
                      'source "$1"; ' + command)
            result = subprocess.run([executable, '-c', source, 'test', str(script)],
                                    capture_output=True, text=True, env=env, timeout=15)
            return result, result.stdout.splitlines()

    def test_launch_aliases_use_same_sensor_only_owner(self):
        for command in ('rail_sensors', 'rail_led', 'rail_ultrasonic'):
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args, ['launch', 'rail_robot_nodes', 'robot.launch.py',
                                    'mode:=sensors', 'serial_port:=/dev/ttyACM8', 'namespace:=bench'])

    def test_color_and_rgb_are_one_shot_namespaced_requests(self):
        for command, value in [('rail_led red', 'red'), ('rail_led rgb 12 034 255', '12,034,255')]:
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(args), 4, args)
            self.assertTrue(args[0].endswith('/rail_ros_once.py'), args[0])
            self.assertEqual(args[1:], ['string', '/bench/debug/led/set', value])

    def test_bad_rgb_does_not_publish(self):
        for command in ('rail_led rgb 256 0 0', 'rail_led rgb -1 0 0', 'rail_led rgb 1 2',
                        'rail_led red extra', 'rail_ultrasonic typo'):
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(args, [])

    def test_ultrasonic_status_reads_only(self):
        result, args = self.run_helper('rail_ultrasonic status')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args, ['topic', 'echo', '/bench/debug/ultrasonic/state',
                                'std_msgs/msg/String', '--field', 'data'])

    def test_tts_without_arguments_runs_only_namespaced_audio_node(self):
        result, args = self.run_helper('rail_tts')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args, ['run', 'rail_robot_nodes', 'audio_alert_node',
                                '--ros-args', '-r', '__ns:=/bench'])
        result, args = self.run_helper('RAIL_ROBOT_NAMESPACE=/; rail_tts')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[-1], '__ns:=/')

    def test_tts_start_done_publish_only_selected_debug_cue(self):
        for command, cue in [('start', 'mission_started'), ('done', 'mission_completed')]:
            result, args = self.run_helper('rail_tts ' + command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(args[0].endswith('/rail_ros_once.py'), args)
            self.assertEqual(args[1:], ['string', '/bench/tts/debug', cue])

    def test_estop_calls_the_namespaced_stm_service_once(self):
        result, args = self.run_helper('rail_estop')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(args[0].endswith('/rail_ros_once.py'), args)
        self.assertEqual(args[1:], ['estop', '/bench/stm/command'])
        result, args = self.run_helper('rail_estop now')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(args, [])

    def test_no_helper_uses_a_short_timeout_around_estop(self):
        source = (CODE/'scripts/ros_sensor_helpers.sh').read_text(encoding='utf-8')
        code = [line for line in source.splitlines() if not line.lstrip().startswith('#')]
        self.assertFalse([line for line in code if 'timeout' in line and 'ros2' in line])

    def test_alert_calls_one_python_process_with_namespaced_topic_and_mode(self):
        for command, mode in [('rail_alert on', 'on'), ('rail_alert off', 'off'),
                              ('rail_alert', '3'), ('rail_alert 0.1', '0.1'), ('rail_alert 60', '60')]:
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 0, command + result.stderr)
            self.assertEqual(len(args), 3, command)
            self.assertTrue(args[0].endswith('/rail_alert.py'), args[0])
            self.assertEqual(args[1:], ['/bench/tts/obstacle_state', mode], command)

    def test_alert_script_sits_next_to_the_helper(self):
        self.assertTrue((CODE/'scripts/rail_alert.py').is_file())
        self.assertTrue((CODE/'scripts/rail_ros_once.py').is_file())
        self.assertTrue((CODE/'scripts/ros_sensor_helpers.sh').is_file())

    def test_alert_status_echoes_audio_status_only(self):
        result, args = self.run_helper('rail_alert status')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args, ['topic', 'echo', '/bench/tts/status',
                                'std_msgs/msg/String', '--field', 'data'])

    def test_bad_alert_arguments_do_not_start_anything(self):
        for command in ('rail_alert typo', 'rail_alert 0', 'rail_alert -1', 'rail_alert 61',
                        'rail_alert 1e3', 'rail_alert .5', 'rail_alert on off', 'rail_alert ""'):
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 2, command + result.stderr)
            self.assertEqual(args, [], command)

    def test_ros_environment_is_not_sourced_again_in_a_loaded_shell(self):
        marker = 'echo sourced >> "$RAIL_ROBOT_HOME/marker"\n'
        count = '; wc -l < "$RAIL_ROBOT_HOME/marker" 2>/dev/null || echo 0'
        fresh, out = self.run_helper('rail_tts start' + count, setup=marker)
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual(out[-1].strip(), '1', out)
        loaded = ('export ROS_DISTRO=humble AMENT_PREFIX_PATH="$RAIL_ROBOT_HOME/ros2/install/pkg"; '
                  'rail_tts start; rail_tts start' + count)
        again, out = self.run_helper(loaded, setup=marker)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(out[-1].strip(), '0', out)
        self.assertEqual(out.count('/bench/tts/debug'), 2)
        other = 'export ROS_DISTRO=humble AMENT_PREFIX_PATH=/opt/ros/humble; rail_tts start' + count
        different, out = self.run_helper(other, setup=marker)
        self.assertEqual(different.returncode, 0, different.stderr)
        self.assertEqual(out[-1].strip(), '1', out)

    def test_tts_result_and_status_echo_only(self):
        for command, topic in [('result', '/bench/tts/result'),
                               ('status', '/bench/tts/status')]:
            result, args = self.run_helper('rail_tts ' + command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args, ['topic', 'echo', topic, 'std_msgs/msg/String',
                                    '--field', 'data'])

    def test_tts_start_and_result_work_with_nounset(self):
        for command in ('start', 'result'):
            result, args = self.run_helper('set -u; rail_tts ' + command)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(args)

    def test_tts_rejects_extra_or_unknown_arguments_without_ros(self):
        for command in ('rail_tts start extra', 'rail_tts done extra',
                        'rail_tts result extra', 'rail_tts typo'):
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(args, [])

    def test_record_runs_only_recorder_with_default_directory(self):
        result, args = self.run_helper("RAIL_ROBOT_HOME='/tmp/rail project'; unset RAIL_RECORD_DIR; rail_record")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args, ['run', 'rail_robot_nodes', 'record_node', '--ros-args',
                                '-r', '__ns:=/bench', '-p',
                                'output_directory:=/tmp/rail project/records'])

    def test_record_directory_argument_overrides_environment_and_normalizes_namespace(self):
        result, args = self.run_helper("RAIL_RECORD_DIR=/tmp/default; RAIL_ROBOT_NAMESPACE=/bench/; rail_record '/tmp/run records'")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[-4:], ['-r', '__ns:=/bench', '-p', 'output_directory:=/tmp/run records'])

    def test_record_uses_directory_environment_and_root_namespace(self):
        result, args = self.run_helper("RAIL_RECORD_DIR=/tmp/records; RAIL_ROBOT_NAMESPACE=/; rail_record")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[-4:], ['-r', '__ns:=/', '-p', 'output_directory:=/tmp/records'])

    def test_record_invalid_arguments_do_not_start_ros(self):
        for command in ("rail_record a b", "rail_record ''"):
            result, args = self.run_helper(command)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(args, [])


if __name__ == '__main__':
    unittest.main()
