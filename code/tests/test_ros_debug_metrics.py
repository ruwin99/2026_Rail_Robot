"""Acquisition FPS and per-model preview metrics, using source clocks."""
import os
import subprocess
import sys
import textwrap
import unittest

from test_ros_vision import JETSON_CODE, ROS_DOUBLES


class DebugMetricsTests(unittest.TestCase):
    def run_contract(self, source, bootstrap=ROS_DOUBLES):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = os.pathsep.join((str(JETSON_CODE), environment.get('PYTHONPATH', '')))
        result = subprocess.run([sys.executable, '-B', '-c', bootstrap + '\n' + textwrap.dedent(source)],
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_fps_counts_skipped_frames_and_expires(self):
        self.run_contract('''
            from node_runtime.metrics import FrameRateMeter
            meter = FrameRateMeter()
            meter.record('one', 1, 100.0, 10.0)
            assert meter.fps(10.0) is None
            meter.record('one', 31, 101.0, 10.1)
            assert abs(meter.fps(10.2) - 30.0) < 1e-6
            meter.record('one', 31, 101.0, 11.2)  # duplicate does not refresh
            assert meter.fps(11.2) is None
            meter.record('two', 1, 102.0, 11.3)
            assert meter.fps(11.3) is None
            meter.record('two', 16, 102.5, 11.4)
            assert abs(meter.fps(11.4) - 30.0) < 1e-6
            meter.record('two', 2, 102.1, 11.5)  # source clock/sequence reset
            assert meter.fps(11.5) is None
        ''', bootstrap='')

    def test_only_executed_models_publish_timing_with_original_identity(self):
        self.run_contract('''
            rust, crack, obstacle, events = detectors()
            with patch.object(vision_node.time, 'monotonic', return_value=10.0):
                node = vision_node.VisionNode(rust, crack, obstacle)
                for role in ('side', 'top'):
                    node._receive(role, frame_message(role, 1))
                node._process_latest()
                timings = node.publishers['debug/vision/timing'].messages
                assert len(timings) == 2
                assert {(t.camera_role, t.model) for t in timings} == {('side','rust'),('top','crack')}
                assert all(t.sequence == 1 and t.stream_id == 'camera-session' for t in timings)
                assert all(t.processing_ms >= 0 and stamp_seconds(t.source_stamp) == 100.0 for t in timings)
                node._process_latest()
                assert len(timings) == 2
                for role in ('side', 'top'):
                    node._receive(role, frame_message(role, 2))
                node._process_latest()
                assert {(t.camera_role,t.model) for t in timings[2:]} == {('side','crack'),('top','obstacle')}
                node.destroy_node()
        ''')

    def test_overlay_uses_each_model_and_camera_fps_and_hides_stale_values(self):
        self.run_contract('''
            with patch.object(display_node.time, 'monotonic', return_value=10.0):
                display = Mock(check_status=Mock(return_value=False), submit=Mock(return_value=True))
                node = display_node.DisplayNode(display_factory=lambda: display)
                node._receive_fps('side', NS(data=29.7))
                node._receive_fps('top', NS(data=30.1))
                for role, model, ms in [('side','rust',12.3),('side','crack',23.4),('top','obstacle',34.5),('top','crack',45.6)]:
                    timing = NS(header=Header(), camera_role=role, model=model, processing_ms=ms, stream_id='camera-session')
                    timing.header.stamp = Time(sec=100)
                    node._receive_timing(timing)
                lines = node._metrics_lines(10.0)
                assert '29.7' in lines[0] and '30.1' in lines[0]
                assert '12.3 ms' in lines[1] and '23.4 ms' in lines[1]
                assert '34.5 ms' in lines[2] and '45.6 ms' in lines[2]
                assert all('N/A' in line for line in node._metrics_lines(12.1))
                original = frame_message('side', 1, value=100)
                node._receive_frame('side', original)
                saved = next(iter(node._frames['side'].values())).frame.copy()
                node._render()
                image = display.submit.call_args.args[0]
                assert np.array_equal(next(iter(node._frames['side'].values())).frame, saved)
                assert image[-70, 15].tolist() == [20,20,20]  # bottom-left panel
                node.destroy_node()
        ''')


if __name__ == '__main__':
    unittest.main()
