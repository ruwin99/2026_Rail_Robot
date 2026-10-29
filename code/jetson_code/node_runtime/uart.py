"""STM protocol I/O and the approved cleaner/pump command set.

The caller owns opening and closing the serial transport. Importing this module
never opens UART or changes actuator state.
"""
from __future__ import annotations

import time

import serial

from .settings import *

def read_uart_messages(
    uart: serial.Serial, pending: bytearray
) -> list[str]:
    waiting = min(uart.in_waiting, 1024)
    if waiting > 0:
        pending.extend(uart.read(waiting))

    messages: list[str] = []
    while b"\n" in pending:
        raw_line, _, remainder = pending.partition(b"\n")
        pending[:] = remainder
        message = raw_line.decode("ascii", errors="replace").strip()
        if message:
            messages.append(message)
    return messages


def send_uart_test_command(uart, command: bytes) -> None:
    """Write one complete actuator-test protocol line or fail closed."""

    command_name = command.decode("ascii").strip()
    try:
        written = uart.write(command)
    except (serial.SerialException, OSError) as exc:
        raise RuntimeError(
            f"Could not send {command_name} to the STM: {exc}"
        ) from exc
    if written != len(command):
        raise RuntimeError(
            f"Could not send the complete {command_name} command "
            f"({written}/{len(command)} bytes)."
        )
    print(f"UART TX: {command_name}")


def enter_uart_actuator_test(
    uart,
    timeout_seconds: float = ACTUATOR_TEST_READY_TIMEOUT_SECONDS,
) -> None:
    """Enter the STM test state before any actuator ON/PWM command is allowed."""

    if timeout_seconds <= 0.0:
        raise ValueError("Actuator-test READY timeout must be positive.")
    try:
        uart.reset_input_buffer()
    except (AttributeError, serial.SerialException, OSError) as exc:
        raise RuntimeError(f"Could not clear stale STM UART input: {exc}") from exc

    send_uart_test_command(uart, ACTUATOR_TEST_START_COMMAND)
    pending = bytearray()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            messages = read_uart_messages(uart, pending)
        except (serial.SerialException, OSError) as exc:
            raise RuntimeError(
                f"Could not read {ACTUATOR_TEST_READY_TRIGGER} from the STM: {exc}"
            ) from exc
        for message in messages:
            print(f"UART RX: {message}")
            if message == ACTUATOR_TEST_READY_TRIGGER:
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"STM did not send {ACTUATOR_TEST_READY_TRIGGER} within "
                f"{timeout_seconds:g} seconds; actuator commands were not enabled."
            )
        time.sleep(ACTUATOR_TEST_READY_POLL_SECONDS)


class CleanerController:
    """Command one of the two approved cleaner PWM setpoints or safe OFF."""

    def __init__(
        self,
        uart,
        commands: ActuatorCommandSet = LEGACY_ACTUATOR_COMMANDS,
    ) -> None:
        self.uart = uart
        self.commands = commands
        self.is_on = False
        self.duty_percent = 0.0
        self._last_on_sent: float | None = None

    def _send(self, command: bytes) -> None:
        if self.uart is None:
            return
        try:
            written = self.uart.write(command)
        except (serial.SerialException, OSError) as exc:
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send {command_name} to the STM: {exc}"
            ) from exc
        if written != len(command):
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send the complete {command_name} command "
                f"({written}/{len(command)} bytes)."
            )

    def force_off(self) -> None:
        """Reset debounce state and immediately request a safe cleaner stop."""

        self.is_on = False
        self.duty_percent = 0.0
        self._last_on_sent = None
        self._send(self.commands.cleaner_off)

    def force_on(self) -> None:
        """Compatibility helper: start at the approved 12.5% baseline."""

        self.set_duty_percent(CLEANER_BASE_DUTY_PERCENT)

    def set_duty_percent(self, duty_percent: float) -> None:
        """Apply OFF, 12.5%, or 25%; active commands also act as heartbeats."""

        duty_percent = float(duty_percent)
        if duty_percent == 0.0:
            if self.is_on:
                self.force_off()
            return
        if duty_percent == CLEANER_BASE_DUTY_PERCENT:
            command = self.commands.cleaner_pwm_12_5
        elif duty_percent == CLEANER_STAGE_ONE_DUTY_PERCENT:
            command = self.commands.cleaner_pwm_25
        else:
            raise ValueError(
                "Cleaner duty must be exactly 0, 12.5, or 25 percent."
            )

        now = time.monotonic()
        if not self.is_on or duty_percent != self.duty_percent:
            self._send(command)
            self.is_on = True
            self.duty_percent = duty_percent
            self._last_on_sent = now
        elif (
            self._last_on_sent is not None
            and now - self._last_on_sent >= CLEANER_HEARTBEAT_SECONDS
        ):
            self._send(command)
            self._last_on_sent = now


class PumpController:
    """Control water immediately; temporal stability is handled upstream."""

    def __init__(
        self,
        uart,
        commands: ActuatorCommandSet = LEGACY_ACTUATOR_COMMANDS,
    ) -> None:
        self.uart = uart
        self.commands = commands
        self.is_on = False
        self._last_on_sent: float | None = None

    def _send(self, command: bytes) -> None:
        if self.uart is None:
            return
        try:
            written = self.uart.write(command)
        except (serial.SerialException, OSError) as exc:
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send {command_name} to the STM: {exc}"
            ) from exc
        if written != len(command):
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send the complete {command_name} command "
                f"({written}/{len(command)} bytes)."
            )

    def force_off(self) -> None:
        """Stop water immediately and arm an immediate first clean observation."""

        self.is_on = False
        self._last_on_sent = None
        self._send(self.commands.pump_off)

    def update(self, water_allowed: bool) -> None:
        now = time.monotonic()
        requested_state = bool(water_allowed)

        if requested_state != self.is_on:
            command = (
                self.commands.pump_on
                if requested_state
                else self.commands.pump_off
            )
            self._send(command)
            self.is_on = requested_state
            self._last_on_sent = now if requested_state else None

        if (
            self.is_on
            and self._last_on_sent is not None
            and now - self._last_on_sent >= PUMP_HEARTBEAT_SECONDS
        ):
            self._send(self.commands.pump_on)
            self._last_on_sent = now


def send_capture_ok(uart) -> None:
    """Acknowledge only a raw capture already accepted by the worker."""

    try:
        written = uart.write(CAPTURE_OK_COMMAND)
    except (serial.SerialException, OSError) as exc:
        raise RuntimeError(f"Could not send CAPTURE_OK to the STM: {exc}") from exc
    if written != len(CAPTURE_OK_COMMAND):
        raise RuntimeError(
            "Could not send the complete CAPTURE_OK command "
            f"({written}/{len(CAPTURE_OK_COMMAND)} bytes)."
        )
    print("UART TX: CAPTURE_OK")
