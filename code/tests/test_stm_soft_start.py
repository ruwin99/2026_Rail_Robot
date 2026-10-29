"""주행 출발 가속(soft start): 실제 펌웨어 C 코드를 운영 설정(DRIVE_SOFT_START_MS)으로 빌드해 가짜 HAL에서 실행한다.
보드나 모터에는 접근하지 않는다."""
from pathlib import Path
import re
import unittest

from stm32_host import run_scenario

CORE = Path(__file__).resolve().parents[2] / "stm_code/code/encoder/Core"
SCENARIOS = (
    "ramp_start_curve",
    "ramp_stall_deferred_then_detected",
    "ramp_stop_is_immediate",
    "ramp_section_transition_keeps_speed",
    "ramp_section_transition_mid_ramp_continues",
    "ramp_restart_after_real_stop",
    "ramp_direction_change_starts_from_floor",
    "ramp_carry_helper_table",
    "ramp_carry_same_direction_section",
    "ramp_carry_direction_change_and_mission_reset",
)


class SoftStartTests(unittest.TestCase):
    def test_production_soft_start_scenarios(self):
        for name in SCENARIOS:
            with self.subTest(name=name):
                # soft_start_ms=None: 운영 헤더 값 그대로 빌드한다.
                self.assertIn("PASS " + name, run_scenario(name, soft_start_ms=None))

    def test_production_default_and_bounds(self):
        config = (CORE / "Inc/drive_sync_config.h").read_text(encoding="utf-8")
        value = int(re.search(r"#define DRIVE_SOFT_START_MS (\d+)U", config).group(1))
        stall = int(re.search(r"#define DRIVE_SYNC_PID_STALL_TIMEOUT_MS (\d+)U", config).group(1))
        self.assertEqual(value, 500)
        # 바퀴가 막혔을 때 폴트까지 최대 가속 시간 + 정지 판정 시간이다. 1초를 넘기지 않는다.
        self.assertLessEqual(value + stall, 1000)

    def test_stopping_paths_never_ramp(self):
        # 정지는 항상 즉시 0이어야 한다. Motor_Stop에는 가속 코드가 들어가지 않는다.
        source = (CORE / "Src/drive_control.c").read_text(encoding="utf-8")
        body = re.search(r"\nvoid Motor_Stop\(void\)\s*\{(.*?)\n\}", source, re.S).group(1)
        self.assertNotIn("Ramp", body)
        self.assertIn("TIM_CHANNEL_1, 0", body)

    def test_isr_marks_stop_sections_and_obstacle_release_carries_coast(self):
        mission = (CORE / "Src/mission.c").read_text(encoding="utf-8")
        isr = re.search(r"void Mission_OnEncoderUpdate\(TIM_HandleTypeDef \*htim\)\s*\{(.*?)\n\}",
                        mission, re.S).group(1)
        # 구간마다 멈추는 분기(INITIAL_SCAN/RETURNING/RESCAN)부터 마지막 else 앞까지.
        stop_branch = isr[isr.index("(mission.state == INITIAL_SCAN)"):]
        stop_branch = stop_branch[:stop_branch.rindex("else")]
        self.assertIn("Drive_MarkSectionBoundary(Drive_GetDirection());", stop_branch)
        # 연속 주행(실시간 청소, 재검사 복귀)은 마지막 경계에서만 표시한다(다음 반대 방향 단계의 보정용).
        continuous = isr[:isr.index("(mission.state == INITIAL_SCAN)")]
        last = continuous[continuous.index("if (encoder.limit >= mission.section_limit)"):]
        self.assertEqual(continuous.count("Drive_MarkSectionBoundary"), 1)
        self.assertIn("Drive_MarkSectionBoundary(Drive_GetDirection());", last)
        # 방향 전환 첫 보정 이벤트는 어떤 구간 판정보다 먼저 소비한다.
        self.assertLess(isr.index("Drive_ConsumeSectionPreroll()"), isr.index("encoder.limit++"))
        ultrasonic = (CORE / "Src/ultrasonic.c").read_text(encoding="utf-8")
        self.assertIn("Drive_CarrySectionCount(saved_cnt3", ultrasonic)
        self.assertIn("Drive_GetObstacleResumeDirection()", ultrasonic)

    def test_floor_is_bounded(self):
        config = (CORE / "Inc/drive_sync_config.h").read_text(encoding="utf-8")
        floor = int(re.search(r"#define DRIVE_SOFT_START_FLOOR_PERCENT (\d+)U", config).group(1))
        self.assertTrue(0 < floor <= 50, floor)


if __name__ == "__main__":
    unittest.main()
