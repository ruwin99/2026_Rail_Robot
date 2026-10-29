"""Real firmware feedforward and READY unit, with no board access."""
from decimal import Decimal, ROUND_HALF_UP
import re
import unittest
from pathlib import Path
from stm32_host import run_scenario

# READY의 DC 필드는 SIDE 저속 PWM 설정값을 그대로 보고한다. 튜닝값을 여기 다시 적지 않는다.
_ROBOT_CONFIG = (Path(__file__).resolve().parents[2] / 'stm_code/code/encoder/Core/Inc/robot_config.h').read_text(encoding='utf-8')
SIDE_DC_PWM = int(re.search(r'#define SIDE_DC_MOTOR_PWM_DUTY_12_5 (\d+)', _ROBOT_CONFIG).group(1))
PUMP_PWM = int(re.search(r'#define WATER_PUMP_PWM_DUTY (\d+)', _ROBOT_CONFIG).group(1))

class StmTrimConfigTests(unittest.TestCase):
    def test_basis_points_match_rounding_in_both_directions_and_pid_modes(self):
        for bp, reverse_bp in ((9920, 10000), (10000, 9950), (9950, 9921), (9921, 9920)):
            for pid in (True, False):
                with self.subTest(bp=bp,pid=pid):
                    output=run_scenario('trim_probe',pid_enabled=pid,right_trim_bp=bp,right_reverse_trim_bp=reverse_bp)
                    rows=re.findall(r'^(TRIM|REVERSE) (\d+) (\d+) (\d+)$',output,re.M)
                    self.assertEqual(len(rows),6)
                    for direction,base,right,left in rows:
                        applied_bp = bp if direction == "TRIM" else reverse_bp
                        expected=int((Decimal(base)*Decimal(applied_bp)/Decimal(10000)).quantize(Decimal(1),rounding=ROUND_HALF_UP))
                        self.assertEqual(int(right),expected)
                        self.assertEqual(int(left),int(base))
                    self.assertIn(f' 8 1600 {bp} {SIDE_DC_PWM} {PUMP_PWM} 601 601 BP',output)

    def test_invalid_trim_cannot_compile(self):
        for field in ('right_trim_bp', 'right_reverse_trim_bp'):
            for bp in (0,-1,20001,4294967295):
                with self.subTest(field=field,bp=bp),self.assertRaisesRegex(AssertionError,'RIGHT_MOTOR_'):
                    run_scenario('trim_probe',**{field:bp})

    def test_direction_change_discards_forward_pi_integral(self):
        for pid in (True,False):
            self.assertIn('PASS trim_reverse_reset',run_scenario('trim_reverse_reset',pid_enabled=pid))
