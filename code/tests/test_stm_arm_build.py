"""Compile and link the current STM32 sources in a temporary directory; never flash."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2] / "stm_code/code/encoder"
# Cube가 선택한 HAL 드라이버만 빌드한다. Debug/objects.list는 새 .c 파일이
# 추가된 뒤 재생성되기 전까지 오래된 목록일 수 있으므로 사용하지 않는다.
HAL_UNITS = (
    "hal", "hal_cortex", "hal_dma", "hal_exti", "hal_flash", "hal_flash_ex",
    "hal_gpio", "hal_gpio_ex", "hal_pwr", "hal_rcc", "hal_rcc_ex",
    "hal_tim", "hal_tim_ex", "hal_uart",
)


class ArmFirmwareBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("arm-none-eabi-gcc")
        if compiler is None and os.name == "nt":
            found = sorted(Path("C:/ST").glob(
                "*/STM32CubeIDE/plugins/*/tools/bin/arm-none-eabi-gcc.exe"))
            compiler = str(found[-1]) if found else None
        if compiler is None:
            raise unittest.SkipTest("ARM cross compiler unavailable; target build not performed")
        cls.compiler = compiler

    def run_checked(self, command):
        result = subprocess.run(command, cwd=PROJECT, capture_output=True,
                                text=True, errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def build_firmware(self, pid_enabled=None):
        include_dirs = (
            "Core/Inc", "Drivers/STM32F1xx_HAL_Driver/Inc",
            "Drivers/STM32F1xx_HAL_Driver/Inc/Legacy",
            "Drivers/CMSIS/Device/ST/STM32F1xx/Include", "Drivers/CMSIS/Include",
        )
        flags = ["-mcpu=cortex-m3", "-mfloat-abi=soft", "-mthumb", "-Og", "-g3",
                 "-ffunction-sections", "-fdata-sections", "-DDEBUG",
                 "-DUSE_HAL_DRIVER", "-DSTM32F103xB", "--specs=nano.specs"]
        flags += ["-I" + str(PROJECT / include) for include in include_dirs]
        if pid_enabled is not None:
            flags += ["-DDRIVE_SYNC_PID_ENABLED=" + str(int(pid_enabled))]
        sources = sorted((PROJECT / "Core/Src").glob("*.c"))
        sources += [PROJECT / "Core/Startup/startup_stm32f103rbtx.s"]
        sources += [PROJECT / "Drivers/STM32F1xx_HAL_Driver/Src" /
                    ("stm32f1xx_" + name + ".c") for name in HAL_UNITS]
        with tempfile.TemporaryDirectory(prefix="rail-modular-arm-") as directory:
            path, objects = Path(directory), []
            for source in sources:
                target = path / ("_".join(source.relative_to(PROJECT).parts) + ".o")
                command = [self.compiler, str(source), "-c", "-o", str(target)] + flags
                if source.suffix == ".c":
                    command += ["-std=gnu11", "-Wall", "-Wextra",
                                "-Werror=implicit-function-declaration"]
                self.run_checked(command)
                objects.append(str(target))
            elf = path / "encoder-check.elf"
            self.run_checked([self.compiler, "-o", str(elf)] + objects + [
                "-mcpu=cortex-m3", "-mfloat-abi=soft", "-mthumb",
                "-T" + str(PROJECT / "STM32F103RBTX_FLASH.ld"),
                "--specs=nosys.specs", "--specs=nano.specs", "-Wl,--gc-sections",
                "-static", "-Wl,--start-group", "-lc", "-lm", "-Wl,--end-group",
            ])
            suffix = ".exe" if os.name == "nt" else ""
            bindir = Path(self.compiler).parent
            symbols = self.run_checked([str(bindir / ("arm-none-eabi-nm" + suffix)), str(elf)])
            # 실제 main/IRQ에서 참조하여 링크했는지 확인한다. 사용하지 않는
            # 소스만 컴파일했다면 --gc-sections에 의해 해당 심벌이 제거된다.
            for name in ("Robot_Init", "Robot_Update", "Robot_OnUartRx",
                         "Robot_OnUartError", "Robot_OnEncoderUpdate", "Robot_FatalStop",
                         "RGB_Init", "RGB_Set", "RGB_Tick50us", "TIM2_IRQHandler",
                         "Ultrasonic_Init", "Ultrasonic_Tick50us", "Ultrasonic_OnEcho",
                         "Ultrasonic_Update", "EXTI9_5_IRQHandler",
                         # 폴트 핸들러가 부르는 출력 정지와 독립 워치독이 실제로 링크됐는지 확인한다.
                         "Fault_StopAndHalt", "Fault_OutputsOff",
                         "Watchdog_Start", "Watchdog_Feed", "HardFault_Handler"):
                self.assertRegex(symbols, r"(?m)\b[Tt] " + name + r"$")
            if pid_enabled is not False:
                self.assertRegex(symbols, r"(?m)\b[Tt] DriveSyncPid_Step$")
            else:
                self.assertNotRegex(symbols, r"(?m)\b[Tt] DriveSyncPid_Step$")
            size = self.run_checked([str(bindir / ("arm-none-eabi-size" + suffix)), str(elf)])
            label = "default" if pid_enabled is None else str(int(pid_enabled))
            print("Temporary ARM image size (no flash; PID=" + label + "):\n" + size.strip())

    def test_default_pid_firmware_builds_and_links_all_modules(self):
        self.build_firmware()

    def test_explicit_pid_on_firmware_builds_and_links_all_modules(self):
        self.build_firmware(True)

    def test_pid_off_firmware_builds_without_controller_runtime(self):
        self.build_firmware(False)


if __name__ == "__main__":
    unittest.main()
