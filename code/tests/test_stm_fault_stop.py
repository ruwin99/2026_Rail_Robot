"""폴트 시 출력 정지와 독립 워치독: 실제 펌웨어 C 코드를 가짜 HAL로 실행하고, CubeMX 재생성에
핸들러 연결이 지워지지 않았는지 소스로 확인한다. 보드에는 접근하지 않는다."""
from pathlib import Path
import re
import unittest

from stm32_host import run_scenario

CORE = Path(__file__).resolve().parents[2] / "stm_code/code/encoder/Core"
FAULT_HANDLERS = ("NMI", "HardFault", "MemManage", "BusFault", "UsageFault")


def function_body(source, name):
    match = re.search(r"\n(?:void|static void)\s+" + re.escape(name) + r"\s*\(void\)\s*\{(.*?)\n\}", source, re.S)
    if match is None:
        raise AssertionError(name + " not found")
    return match.group(1)


class FaultStopTests(unittest.TestCase):
    def test_fault_outputs_off_zeroes_every_pwm_and_direction_pin(self):
        self.assertIn("PASS fault_outputs_off", run_scenario("fault_outputs_off"))

    def test_fault_outputs_off_after_real_running_state(self):
        self.assertIn("PASS fault_outputs_off_while_running", run_scenario("fault_outputs_off_while_running"))

    def test_watchdog_configuration_and_main_loop_feed(self):
        self.assertIn("PASS fault_watchdog", run_scenario("fault_watchdog"))

    def test_every_fault_vector_stops_outputs_instead_of_spinning(self):
        # 빈 while(1)은 PWM이 마지막 값으로 계속 출력된다. CubeMX가 다시 생성해도 USER CODE 구역이
        # 유지되지만, 핸들러가 바뀌면 이 시험이 먼저 실패한다.
        source = (CORE / "Src/stm32f1xx_it.c").read_text(encoding="utf-8")
        for name in FAULT_HANDLERS:
            body = function_body(source, name + "_Handler")
            self.assertIn("Fault_StopAndHalt();", body, name)
            # 정지 호출이 NMI의 HAL 호출보다 먼저 와야 한다.
            self.assertLess(body.index("Fault_StopAndHalt();"),
                            body.index("while") if "while" in body else len(body), name)

    def test_main_starts_watchdog_after_init_and_error_handler_keeps_feeding(self):
        main = (CORE / "Src/main.c").read_text(encoding="utf-8")
        self.assertLess(main.index("Robot_Init();"), main.index("Watchdog_Start();"))
        self.assertLess(main.index("Watchdog_Start();"), main.index("USER CODE END 2"))
        handler = function_body(main, "Error_Handler")
        self.assertGreaterEqual(handler.count("Watchdog_Feed();"), 2)
        self.assertIn("Robot_FatalStop();", handler)

    def test_robot_update_feeds_watchdog_first(self):
        app = (CORE / "Src/robot_app.c").read_text(encoding="utf-8")
        body = function_body(app, "Robot_Update")
        self.assertLess(body.index("Watchdog_Feed();"), body.index("Drive_MarkMainAlive();"))

    def test_watchdog_uses_registers_not_hal_module(self):
        # CubeMX가 hal_conf.h를 다시 만들면 HAL_IWDG_MODULE_ENABLED가 꺼진다. 레지스터 직접 사용이어야 한다.
        source = (CORE / "Src/fault_stop.c").read_text(encoding="utf-8")
        self.assertNotIn("HAL_IWDG", source)
        self.assertNotIn("hiwdg", source)
        self.assertIn("IWDG->KR", source)


if __name__ == "__main__":
    unittest.main()
