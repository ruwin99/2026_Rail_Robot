from __future__ import annotations

import argparse
import sys
import threading
import time


DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_BAUD_RATE = 115200
HEARTBEAT_SECONDS = 0.4
READY_TIMEOUT_SECONDS = 2.0

ACTUATOR_TEST_START = b"ACTUATOR_TEST_START\r\n"
ACTUATOR_TEST_STOP = b"ACTUATOR_TEST_STOP\r\n"
FRONT_CLEANER_PWM_12_5 = b"FRONT_CLEANER_PWM_12_5\r\n"
FRONT_CLEANER_OFF = b"FRONT_CLEANER_OFF\r\n"
FRONT_PUMP_ON = b"FRONT_PUMP_ON\r\n"
FRONT_PUMP_OFF = b"FRONT_PUMP_OFF\r\n"
SIDE_CLEANER_PWM_12_5 = b"SIDE_CLEANER_PWM_12_5\r\n"
SIDE_CLEANER_OFF = b"SIDE_CLEANER_OFF\r\n"
SIDE_PUMP_ON = b"SIDE_PUMP_ON\r\n"
SIDE_PUMP_OFF = b"SIDE_PUMP_OFF\r\n"
ACTUATOR_TEST_READY = "ACTUATOR_TEST_READY"

ALL_OUTPUTS_OFF = (
    FRONT_CLEANER_OFF,
    FRONT_PUMP_OFF,
    SIDE_CLEANER_OFF,
    SIDE_PUMP_OFF,
)


def send_command(uart, command: bytes) -> None:
    written = uart.write(command)
    if written != len(command):
        name = command.decode("ascii").strip()
        raise RuntimeError(
            f"UART command was only partially sent: {name} "
            f"({written}/{len(command)} bytes)"
        )


def wait_until_ready(uart, timeout_seconds: float = READY_TIMEOUT_SECONDS) -> None:
    uart.reset_input_buffer()
    send_command(uart, ACTUATOR_TEST_START)
    pending = bytearray()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        waiting = min(uart.in_waiting, 1024)
        if waiting:
            pending.extend(uart.read(waiting))
        while b"\n" in pending:
            raw_line, _, remainder = pending.partition(b"\n")
            pending[:] = remainder
            message = raw_line.decode("ascii", errors="replace").strip()
            if message:
                print(f"STM: {message}")
            if message == ACTUATOR_TEST_READY:
                return
        time.sleep(0.01)

    raise TimeoutError(
        f"STM did not send {ACTUATOR_TEST_READY} within {timeout_seconds:g} seconds"
    )


class ActuatorTestSession:
    def __init__(self, uart, heartbeat_seconds: float = HEARTBEAT_SECONDS) -> None:
        self.uart = uart
        self.heartbeat_seconds = heartbeat_seconds
        self._mode = "stop"
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def start(self) -> None:
        wait_until_ready(self.uart)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="actuator-test-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def select(self, mode: str) -> None:
        with self._lock:
            if mode == "front_dc":
                self._mode = "stop"
                self._send_all_off()
                send_command(self.uart, FRONT_CLEANER_PWM_12_5)
                self._mode = mode
                print("FRONT DC motor: ON (PWM 12.5%); all other outputs: OFF")
            elif mode == "front_pump":
                self._mode = "stop"
                self._send_all_off()
                send_command(self.uart, FRONT_PUMP_ON)
                self._mode = mode
                print("FRONT pump: ON; all other outputs: OFF")
            elif mode == "side_dc":
                self._mode = "stop"
                self._send_all_off()
                send_command(self.uart, SIDE_CLEANER_PWM_12_5)
                self._mode = mode
                print("SIDE DC motor: ON (PWM 12.5%); all other outputs: OFF")
            elif mode == "side_pump":
                self._mode = "stop"
                self._send_all_off()
                send_command(self.uart, SIDE_PUMP_ON)
                self._mode = mode
                print("SIDE pump: ON; all other outputs: OFF")
            elif mode == "stop":
                self._mode = "stop"
                self._send_all_off()
                print("FRONT/SIDE DC motors and pumps: OFF")
            else:
                raise ValueError(f"Unknown dc_test command: {mode}")

    def _send_all_off(self) -> None:
        for command in ALL_OUTPUTS_OFF:
            send_command(self.uart, command)

    def _heartbeat_loop(self) -> None:
        while not self._closed.wait(self.heartbeat_seconds):
            try:
                with self._lock:
                    if self._mode == "front_dc":
                        send_command(self.uart, FRONT_CLEANER_PWM_12_5)
                    elif self._mode == "front_pump":
                        send_command(self.uart, FRONT_PUMP_ON)
                    elif self._mode == "side_dc":
                        send_command(self.uart, SIDE_CLEANER_PWM_12_5)
                    elif self._mode == "side_pump":
                        send_command(self.uart, SIDE_PUMP_ON)
            except Exception as exc:
                print(f"UART heartbeat failed: {exc}", file=sys.stderr)
                self._closed.set()
                return

    def close(self) -> None:
        self._closed.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        with self._lock:
            for command in (*ALL_OUTPUTS_OFF, ACTUATOR_TEST_STOP):
                try:
                    send_command(self.uart, command)
                except Exception as exc:
                    name = command.decode("ascii").strip()
                    print(f"Could not send {name} during shutdown: {exc}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively test one STM cleaner or water pump at a time."
    )
    parser.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help=f"STM UART device (default: {DEFAULT_SERIAL_PORT})",
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help=f"UART baud rate (default: {DEFAULT_BAUD_RATE})",
    )
    return parser.parse_args(argv)


def run_interactive(session: ActuatorTestSession) -> None:
    valid_commands = {"front_dc", "front_pump", "side_dc", "side_pump", "stop"}
    print(
        "Commands: front_dc | front_pump | side_dc | side_pump | stop  "
        "(Ctrl+C to exit)"
    )
    while True:
        try:
            command = input("dc_test> ").strip().lower()
        except EOFError:
            return
        if command in valid_commands:
            session.select(command)
        elif command:
            print(
                "Unknown command. Use: front_dc | front_pump | "
                "side_dc | side_pump | stop"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import serial
    except ModuleNotFoundError:
        print("pyserial is unavailable in the selected virtual environment.", file=sys.stderr)
        return 1

    uart = None
    session = None
    try:
        uart = serial.Serial(
            port=args.serial_port,
            baudrate=args.baud_rate,
            timeout=0,
            write_timeout=0.2,
        )
        session = ActuatorTestSession(uart)
        session.start()
        print(f"Connected to STM on {args.serial_port} at {args.baud_rate} baud.")
        run_interactive(session)
    except KeyboardInterrupt:
        print("\nStopping FRONT/SIDE actuators.")
    except Exception as exc:
        print(f"dc_test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if session is not None:
            session.close()
        if uart is not None:
            uart.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
