"""Host execution of the same integer C controller included by STM firmware."""
import importlib.util
import ctypes
import os
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/simulate_drive_sync_pid.py'
spec = importlib.util.spec_from_file_location('pid_simulation', SCRIPT)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def raw2(forward_counts):
    """Measured board polarity: physical forward decreases TIM4, modulo 601."""
    return (-forward_counts) % 601


class TrimHeaderTests(unittest.TestCase):
    def test_simulator_reads_single_firmware_setting_and_rounds_hundredths(self):
        from unittest.mock import patch
        from decimal import Decimal, ROUND_HALF_UP
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / 'robot_config.h'
            for bp, comment in ((9920, ''), (10000, ' // neutral'),
                                (9950, ' /* trim */'), (9921, '')):
                header.write_text(f'#define RIGHT_MOTOR_FORWARD_TRIM_BP {bp}U{comment}\n#define RIGHT_MOTOR_REVERSE_TRIM_BP 10000U\n', encoding='utf-8')
                value = sim.read_right_motor_trim_bp(header)
                self.assertEqual(value, bp)
                self.assertEqual(sim.read_right_motor_trim_bp(header, direction=-1), 10000)
                with patch.object(sim, 'RIGHT_MOTOR_FORWARD_TRIM_BP', value):
                    for base in (1600, 800, 63):
                        expected=int((Decimal(base)*Decimal(bp)/Decimal(10000)).quantize(Decimal(1),rounding=ROUND_HALF_UP))
                        self.assertEqual(sim.encoder_1_base_pwm(base), expected)
            for text in ('#define RIGHT_MOTOR_FORWARD_TRIM_BP 0U', '#define RIGHT_MOTOR_FORWARD_TRIM_BP 20001U', ''):
                header.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    sim.read_right_motor_trim_bp(header)


class DriveSyncPidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='rail-pid-test-')
        # Test-only access to the real C configuration; no duplicate PID math.
        original_wrapper = sim.WRAPPER
        # All pre-existing regression cases exercise the neutral, bit-exact path.
        sim.WRAPPER = sim.WRAPPER.replace('config = DriveSyncPid_DefaultConfig();',
            'config = DriveSyncPid_DefaultConfig(); config.forward_right_target_bp = 10000; config.reverse_left_target_bp = 10000;')
        sim.WRAPPER += '''
EXPORT void test_gains(int p, int i, int d) {
  config.kp_q8 = p; config.ki_q8 = i; config.kd_q8 = d;
}
EXPORT void test_target(unsigned bp) { config.forward_right_target_bp = bp; }
EXPORT void test_reverse_target(unsigned bp) { config.reverse_left_target_bp = bp; }
EXPORT int test_integral(void) { return state.integral_q8; }
EXPORT int test_remainder(void) { return state.integral_remainder; }
EXPORT void test_default_config(void) { config = DriveSyncPid_DefaultConfig(); }
EXPORT void test_large_encoders(void) {
  config.encoder1_modulus = config.encoder2_modulus = 65536;
  config.encoder1_scale = config.encoder2_scale = 16;
}
'''
        try:
            cls.dll = sim.build_controller(Path(cls.temp.name))
        finally:
            sim.WRAPPER = original_wrapper
        cls.dll.test_gains.argtypes = [ctypes.c_int] * 3
        cls.dll.test_gains.restype = None
        cls.dll.test_target.argtypes = [ctypes.c_uint32]
        cls.dll.test_target.restype = None
        cls.dll.test_reverse_target.argtypes = [ctypes.c_uint32]
        cls.dll.test_reverse_target.restype = None
        cls.dll.test_integral.restype = cls.dll.test_remainder.restype = ctypes.c_int
        cls.dll.test_large_encoders.restype = None
        cls.dll.test_default_config.restype = None

    @classmethod
    def tearDownClass(cls):
        if os.name == 'nt':
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            ctypes.windll.kernel32.FreeLibrary(cls.dll._handle)
        del cls.dll
        cls.temp.cleanup()

    def setUp(self):
        self.dll.sim_reset(0, 0, 0)

    def test_forward_target_bias_survives_subcount_error(self):
        self.dll.test_target(9980)
        for n in range(1, 101):
            result = sim.step(self.dll, (10*n)%601, raw2(20*n), 10*n)
        self.assertEqual(result[2], 0)  # Legacy integer error cannot show -0.04 count.
        self.assertLess(result[3], 0)
        self.assertGreater(result[0], sim.encoder_1_base_pwm(800))
        self.assertLess(result[1], 800)
        self.assertEqual(self.dll.test_integral(), -512)
        self.assertEqual(self.dll.test_remainder(), 0)

    def test_explicit_reverse_left_target_survives_subcount_error(self):
        self.dll.test_default_config()
        self.dll.test_reverse_target(9900)
        for n in range(1,101):
            result=sim.step(self.dll,(-10*n)%601,(20*n)%601,10*n,-1)
        self.assertEqual(result[2],0)
        self.assertEqual(self.dll.test_integral(),2560)
        self.assertEqual(result[3],11)
        self.assertEqual(result[:2],[sim.encoder_1_base_pwm(800,-1)-11,811])

    def test_reverse_neutral_golden_and_forward_target_independence(self):
        vectors=[(591,10,10,-10,-20),(586,25,25,3,0),(566,55,45,-5,-20),
                 (564,63,55,4,0),(544,83,75,-10,-20),(539,99,85,6,0)]
        for c1,c2,tick,error,correction in vectors:
            self.assertEqual(sim.step(self.dll,c1,c2,tick,-1),
                [sim.encoder_1_base_pwm(800,-1)-correction,800+correction,error,correction,1,0])
        traces=[]
        for reverse_target in (10000,9900):
            self.dll.sim_reset(0,0,0);self.dll.test_target(9800);self.dll.test_reverse_target(reverse_target)
            traces.append([sim.step(self.dll,(10*n)%601,raw2(20*n),10*n) for n in range(1,101)])
        self.assertEqual(*traces)

    def test_reverse_target_bounds_fractional_elapsed_and_limits(self):
        for target in (9700,9900,9990,10000,10100):
            self.dll.sim_reset(0,0,0);self.dll.test_reverse_target(target)
            self.assertEqual(sim.step(self.dll,591,20,10,-1)[4:],[1,0])
        for target in (9699,10101):
            self.dll.sim_reset(0,0,0);self.dll.test_reverse_target(target)
            self.assertEqual(sim.step(self.dll,591,20,10,-1)[5],1)
        for target,expected_i in ((9900,2560),(9990,256)):
            for interval in (10,20):
                self.dll.sim_reset(0,0,0);self.dll.test_reverse_target(target)
                for tick in range(interval,1001,interval):
                    sim.step(self.dll,(-tick)%601,(2*tick)%601,tick,-1)
                self.assertEqual(self.dll.test_integral(),expected_i)
        self.dll.sim_reset(0,0,0);self.dll.test_reverse_target(9900)
        cap=min(400,sim.encoder_1_base_pwm(800,-1)//4,200,3599-sim.encoder_1_base_pwm(800,-1))
        previous=0
        for n in range(1,4001):
            result=sim.step(self.dll,(-10*n)%601,(20*n)%601,10*n,-1)
            self.assertEqual(result[5],0)
            self.assertLessEqual(abs(result[3]-previous),20)
            self.assertLessEqual(abs(result[3]),cap)
            previous=result[3]
        self.assertEqual(previous,cap)
        self.assertLessEqual(self.dll.test_integral(),cap*256)
        self.dll.sim_reset(0,0,0)
        self.assertEqual(self.dll.test_remainder(),0)
        self.assertEqual(self.dll.test_integral(),0)

    def test_neutral_target_matches_prechange_mixed_elapsed_vectors(self):
        # Captured from production C before target-bias implementation.
        vectors = [(10,591,10,-10,-20), (15,576,25,3,0), (35,546,45,-5,-20),
                   (37,538,55,4,0), (57,518,75,-10,-20), (62,502,85,6,0)]
        for c1,c2,tick,error,correction in vectors:
            self.assertEqual(sim.step(self.dll,c1,c2,tick),
                [sim.encoder_1_base_pwm(800)-correction,800+correction,error,correction,1,0])
            self.assertEqual(self.dll.test_remainder(),0)

    def test_fractional_bias_reset_and_reverse_neutrality(self):
        self.dll.test_target(9980)
        sim.step(self.dll,10,581,10)
        self.assertNotEqual(self.dll.test_remainder(),0)
        self.dll.sim_reset(0,0,0)
        self.assertEqual(self.dll.test_integral(),0)
        self.assertEqual(self.dll.test_remainder(),0)
        before=[sim.step(self.dll,(601-5*n)%601,(16*n)%601,n*10,-1) for n in range(1,21)]
        self.dll.sim_reset(0,0,0); self.dll.test_target(9980)
        after=[sim.step(self.dll,(601-5*n)%601,(16*n)%601,n*10,-1) for n in range(1,21)]
        self.assertEqual(after,before)
        self.assertEqual(self.dll.test_remainder(),0)

    def test_fractional_target_range_is_checked(self):
        for target in (9700,9800,10100):
            self.dll.sim_reset(0,0,0); self.dll.test_target(target)
            self.assertEqual(sim.step(self.dll,10,581,10)[4:],[1,0])
        for target in (0,9299,10501):
            self.dll.sim_reset(0,0,0); self.dll.test_target(target)
            self.assertEqual(sim.step(self.dll,10,581,10)[5],1)

    def test_fractional_math_does_not_overflow_at_supported_encoder_limits(self):
        self.dll.test_target(9980); self.dll.test_large_encoders()
        result=sim.step(self.dll,32000,65535,10)
        self.assertEqual(result[2:],[-511984,-20,1,0])

    def test_fractional_bias_preserves_elapsed_time_slew_cap_and_antiwindup(self):
        for interval in (10,20):
            self.dll.sim_reset(0,0,0); self.dll.test_target(9980)
            for tick in range(interval,1001,interval):
                sim.step(self.dll,tick%601,raw2(2*tick),tick)
            self.assertEqual(self.dll.test_integral(),-512)
        self.dll.sim_reset(0,0,0); self.dll.test_target(9980)
        cap=min(400,sim.encoder_1_base_pwm(800)//4,200,3599-sim.encoder_1_base_pwm(800))
        previous=0
        for n in range(1,15001):
            result=sim.step(self.dll,(10*n)%601,raw2(20*n),10*n)
            self.assertEqual(result[5],0)
            self.assertLessEqual(abs(result[3]-previous),20)
            self.assertLessEqual(abs(result[3]),cap)
            previous=result[3]
        self.assertEqual(previous,-cap)
        self.assertGreaterEqual(self.dll.test_integral(),-cap*256)
        self.assertLess(abs(self.dll.test_remainder()),100000)

    def test_direction_specific_feedback_signs_preserve_forward_and_invert_reverse(self):
        base = sim.encoder_1_base_pwm(1600)
        self.assertEqual(sim.step(self.dll, 10, 591, 10, base=1600),
                         [base+20, 1580, -10, -20, 1, 0])
        self.dll.sim_reset(0, 0, 0)
        self.assertEqual(sim.step(self.dll, 5, 581, 10, base=1600),
                         [base-20, 1620, 10, 20, 1, 0])
        self.dll.sim_reset(0, 0, 0)
        reverse_base = sim.encoder_1_base_pwm(1600, -1)
        self.assertEqual(sim.step(self.dll, 591, 10, 10, -1, 1600),
                         [reverse_base+20, 1580, -10, -20, 1, 0])
        self.dll.sim_reset(0, 0, 0)
        self.assertEqual(sim.step(self.dll, 596, 20, 10, -1, 1600),
                         [reverse_base-20, 1620, 10, 20, 1, 0])

    def test_measured_forward_sample_normalizes_decreasing_second_encoder(self):
        # Actual board: both wheels move physically forward, TIM3 +4/TIM4 -7.
        result = sim.step(self.dll, 4, 594, 10)
        self.assertEqual(result[2], -1)  # physical R-L=1; forward feedback coupling reverses it
        self.assertEqual(result[4:], [1, 0])

    def test_second_encoder_true_wrong_direction_still_faults(self):
        # With forward requested, TIM4 +2 is now genuine reverse feedback.
        result = sim.step(self.dll, 4, 2, 10)
        self.assertEqual(result[:2], [0, 0])
        self.assertEqual(result[5], 5)

    def test_encoder_scaling_does_not_create_false_error(self):
        self.assertEqual(sim.step(self.dll, 5, raw2(10), 10), [sim.encoder_1_base_pwm(800), 800, 0, 0, 1, 0])

    def test_reverse_uses_independent_trim_for_zero_error_sample(self):
        from unittest.mock import patch
        with patch.object(sim, 'RIGHT_MOTOR_FORWARD_TRIM_BP', 9920), patch.object(sim, 'RIGHT_MOTOR_REVERSE_TRIM_BP', 10000):
            result = sim.step(self.dll, 596, 10, 10, direction=-1)
            self.assertEqual(result, [800, 800, 0, 0, 1, 0])
            self.assertEqual(sim.encoder_1_base_pwm(1600, direction=1), 1587)
            self.assertEqual(sim.encoder_1_base_pwm(1600, direction=-1), 1600)

    def test_no_early_update(self):
        self.assertEqual(sim.step(self.dll, 4, raw2(8), 9)[4:], [0, 0])
        self.assertEqual(sim.step(self.dll, 5, raw2(10), 10)[4:], [1, 0])

    def test_forward_and_reverse_counter_wrap(self):
        # Forward +5/-10; reverse -5/+10, including wraps of both counters.
        self.dll.sim_reset(598, 4, 0)
        self.assertEqual(sim.step(self.dll, 2, 595, 10)[2:], [0, 0, 1, 0])
        self.dll.sim_reset(2, 598, 0)
        self.assertEqual(sim.step(self.dll, 598, 7, 10, -1)[2:], [0, 0, 1, 0])

    def test_uint32_clock_wrap(self):
        self.dll.sim_reset(0, 0, 0xFFFFFFFA)
        self.assertEqual(sim.step(self.dll, 5, raw2(10), 4)[4:], [1, 0])

    def test_delay_trips_and_latches(self):
        self.assertEqual(sim.step(self.dll, 10, raw2(20), 21)[5], 2)
        result = sim.step(self.dll, 15, raw2(30), 31)
        self.assertEqual(result[:2], [0, 0])
        self.assertEqual(result[5], 2)

    def test_exact_gap_limit_is_allowed(self):
        self.assertEqual(sim.step(self.dll, 10, raw2(20), 20)[4:], [1, 0])

    def test_integral_accumulates_same_amount_over_equal_elapsed_time(self):
        # Physical R-L=2 becomes control error -2; 20ms integrates to -4 PWM,
        # whether sampled once at 20ms or twice at 10ms (Ki=1 per 10ms).
        self.dll.test_gains(0, 256, 0)
        self.assertEqual(sim.step(self.dll, 3, raw2(4), 10)[3], -2)
        self.assertEqual(sim.step(self.dll, 6, raw2(8), 20)[3], -4)
        self.dll.sim_reset(0, 0, 0)
        self.dll.test_gains(0, 256, 0)
        self.assertEqual(sim.step(self.dll, 6, raw2(8), 20)[3], -4)

    def test_derivative_uses_elapsed_time_between_speed_errors(self):
        # The same error change spread over twice the time gives half D.
        self.dll.test_gains(0, 0, 256)
        self.assertEqual(sim.step(self.dll, 5, raw2(10), 10)[3], 0)
        self.assertEqual(sim.step(self.dll, 12, raw2(20), 20)[3], -4)
        self.dll.sim_reset(0, 0, 0)
        self.dll.test_gains(0, 0, 256)
        self.assertEqual(sim.step(self.dll, 5, raw2(10), 10)[3], 0)
        self.assertEqual(sim.step(self.dll, 19, raw2(30), 30)[3], -2)

    def test_wrong_direction_latches_but_one_count_noise_is_allowed(self):
        self.assertEqual(sim.step(self.dll, 600, 0, 10)[5], 0)
        self.dll.sim_reset(0, 0, 0)
        # TIM3 -2 remains wrong even though TIM4 has the expected -10 direction.
        result = sim.step(self.dll, 599, raw2(10), 10)
        self.assertEqual(result[:2], [0, 0])
        self.assertEqual(result[5], 5)
        self.assertEqual(sim.step(self.dll, 5, raw2(20), 20)[5], 5)

    def test_slew_and_correction_bounds(self):
        previous = 0
        cap = min(400, sim.encoder_1_base_pwm(800)//4, 800//4, 3599-sim.encoder_1_base_pwm(800))
        for n in range(1, 101):
            result = sim.step(self.dll, (15*n) % 601, raw2(2*n), 10*n)
            self.assertEqual(result[5], 0)
            self.assertLessEqual(abs(result[3] - previous), 20)
            self.assertLessEqual(abs(result[3]), cap)
            self.assertTrue(0 <= result[0] <= 3599 and 0 <= result[1] <= 3599)
            previous = result[3]
        self.assertEqual(result[3], -cap)
        for n in range(1, 201):
            result = sim.step(self.dll, (1500 + 2*n) % 601, raw2(200 + 30*n), 1000 + 10*n)
        self.assertGreater(result[3], 0)

    def test_stalled_encoder_stops_both_outputs(self):
        for n in range(1, 51):
            result = sim.step(self.dll, (5*n) % 601, 0, 10*n)
        self.assertEqual(result[:2], [0, 0])
        self.assertEqual(result[5], 4)

    def test_full_pwm_correction_cap_tracks_right_feedforward(self):
        cap = min(400, sim.encoder_1_base_pwm(1600)//4, 1600//4, 3599-sim.encoder_1_base_pwm(1600))
        for n in range(1, 101):
            result = sim.step(self.dll, (15*n) % 601, raw2(2*n), 10*n, base=1600)
            self.assertEqual(result[5], 0)
            self.assertLessEqual(abs(result[3]), cap)
        self.assertEqual(result[3], -cap)

    def test_bad_sample_rejected(self):
        self.assertEqual(sim.step(self.dll, 601, raw2(10), 10)[5], 3)

    def test_section_reset_rebases_sampling(self):
        sim.step(self.dll, 5, raw2(10), 10)
        self.dll.sim_reset(0, raw2(10), 1000)
        self.assertEqual(sim.step(self.dll, 5, raw2(20), 1010)[2:], [0, 0, 1, 0])

    def test_inverted_feedback_does_not_stabilize_the_original_positive_response_plant(self):
        for base in (800, 1600):
            before, _ = sim.run_case(self.dll, 'open', controlled=False, base=base)
            after, _ = sim.run_case(self.dll, 'closed', base=base)
            self.assertEqual(after['fault_code'], 0)
            # This old plant assumes PWM1 drives ENC1 and PWM2 drives ENC2.
            # Do not silently swap its wiring to claim the new sign improves hardware.
            self.assertGreater(after['mean_speed_mismatch_pct'], before['mean_speed_mismatch_pct'])
        for gap in (20, 60, 100):
            result, _ = sim.run_case(self.dll, 'late', gap_ms=gap)
            self.assertEqual(result['fault_code'], 2)
            self.assertEqual(result['fault_at_ms'], 3000+gap)
            self.assertEqual(result['final_pwm'], [0, 0])

    def test_reverse_inverted_sign_does_not_stabilize_positive_response_plant(self):
        # This math harness explicitly retains forward default gains in both directions.
        # Actual direction-specific scheduling is covered by test_stm_directional_gains.
        # The assumed positive-response plant is not proof of physical wheel mapping.
        for gain in (0.86, 1.10):
            with self.subTest(physical_right_gain=gain):
                plant = dict(direction=-1, base=1600, left_gain=gain, right_loaded_gain=1.0)
                before, _ = sim.run_case(self.dll, 'reverse_open', controlled=False, **plant)
                after, _ = sim.run_case(self.dll, 'reverse_pi', **plant)
                self.assertEqual(after['fault_code'], 0)
                self.assertGreater(after['mean_speed_mismatch_pct'], before['mean_speed_mismatch_pct'])


if __name__ == '__main__':
    unittest.main()
