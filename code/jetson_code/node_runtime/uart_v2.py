"""Bounded, opt-in RR2 mission command/heartbeat transport.

RR2 is deliberately separate from the legacy actuator/debug strings. A valid
serial write is not an execution acknowledgement: only an MCU ACK after the
FSM accepts the command makes a mission command successful. This does not
prove physical motion or output completion.
"""
from __future__ import annotations

import time
import uuid


from .rr2_protocol import (
    MAX_LINE_BYTES, _fields, _from_fields, crc16_ccitt, decode_rr2, encode_rr2,
)


def encode_frame(*fields: str) -> bytes:
    """Compatibility call shape; omitted handshake capability means MISSION.

    Wire frames always use the capability-bearing contract. Capless wire frames
    are not accepted by the shared decoder.
    """
    if (len(fields) == 2 and fields[0] == "HELLO") or (
        len(fields) == 3 and fields[0] == "READY"
    ):
        fields += ("MISSION",)
    return encode_rr2(_from_fields(fields))


def decode_frame(line: bytes) -> tuple[str, ...]:
    """Return the existing tuple view after shared strict frame validation."""
    return _fields(decode_rr2(line))


class UartV2:
    """One critical command in flight; same sequence is retried, never replayed anew."""

    def __init__(self, uart, *, session: str | None = None, timeout: float = 0.14, attempts: int = 3):
        self.uart = uart
        self.session = session or uuid.uuid4().hex
        if len(self.session) != 32 or any(char not in "0123456789abcdef" for char in self.session):
            raise ValueError("RR2 session must be 32 lowercase hex characters")
        if timeout <= 0 or attempts < 1:
            raise ValueError("RR2 timeout and attempts must be positive")
        self.timeout = timeout
        self.attempts = attempts
        self._pending = bytearray()
        self._discarding_line = False
        self._seq = 0
        self._last_ack = time.monotonic()
        self._last_ping = 0.0
        self._pending_ping = None

    def _next_seq(self) -> int:
        if self._seq >= 0xFFFFFFFF:
            raise RuntimeError("RR2 sequence exhausted; restart the session")
        self._seq += 1
        return self._seq

    def _write(self, frame: bytes) -> None:
        if self.uart.write(frame) != len(frame):
            raise RuntimeError("Incomplete RR2 UART write")

    def _read(self) -> list[tuple[str, ...] | str]:
        waiting = min(self.uart.in_waiting, 1024)
        if waiting:
            chunk = self.uart.read(waiting)
            if self._discarding_line:
                # Clearing an oversized buffer is not a frame boundary. Ignore
                # its suffix across reads until LF, including valid-looking ACKs.
                _, terminator, chunk = chunk.partition(b"\n")
                if not terminator:
                    return []
                self._discarding_line = False
            self._pending.extend(chunk)
        if len(self._pending) > MAX_LINE_BYTES and b"\n" not in self._pending:
            self._pending.clear()
            self._discarding_line = True
            raise ValueError("RR2 UART line exceeds 128 bytes")
        results = []
        while b"\n" in self._pending:
            raw, _, remainder = self._pending.partition(b"\n")
            self._pending[:] = remainder
            if len(raw) + 1 > MAX_LINE_BYTES:
                raise ValueError("RR2 UART line exceeds 128 bytes")
            line = bytes(raw) + b"\n"
            if line.startswith(b"RR2 "):
                try:
                    results.append(decode_frame(line))
                except ValueError:
                    # A damaged response is never an ACK; a bounded retry follows.
                    continue
            else:
                token = line.decode("ascii", errors="replace").strip()
                if token:
                    results.append(token)
        return results

    def _exchange(self, frame: bytes, expected: tuple[str, ...]) -> list[str]:
        events: list[str] = []
        for _ in range(self.attempts):
            self._write(frame)
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                matched = False
                for item in self._read():
                    if isinstance(item, str):
                        events.append(item)
                    elif item[0] == "ACK" and len(item) == 6 and item[1] == self.session and item[3] == "PING":
                        if item[2] == self._pending_ping and item[4] == "ACCEPTED":
                            self._last_ack = time.monotonic()
                            self._pending_ping = None
                    elif item[:len(expected)] == expected:
                        if item[0] == "ACK" and (len(item) != 6 or item[4] != "ACCEPTED"):
                            raise RuntimeError("Malformed RR2 execution ACK")
                        self._last_ack = time.monotonic()
                        # A successful command also proves liveness; an older
                        # lost PING reply must not strand the heartbeat slot.
                        if item[0] == "ACK":
                            self._pending_ping = None
                        matched = True
                    elif item[0] == "NACK" and item[1:len(expected)] == expected[1:]:
                        raise RuntimeError("STM rejected RR2 command: " + " ".join(item[4:]))
                if matched:
                    return events
                time.sleep(0.001)
        raise TimeoutError("STM RR2 acknowledgement timed out")

    def hello(self) -> list[str]:
        return self._exchange(encode_frame("HELLO", self.session), ("READY", self.session, "2", "MISSION"))

    def command(self, opcode: str) -> list[str]:
        if opcode not in ("START", "CAPTURE_OK"):
            raise ValueError("RR2 mission command is not allowed")
        # Default response windows total 420 ms, excluding writes/scheduling.
        # Require a recent MCU ACK before beginning a bounded command exchange.
        if time.monotonic() - self._last_ack >= 0.14:
            raise TimeoutError("RR2 link is stale before mission command")
        sequence = str(self._next_seq())
        return self._exchange(
            encode_frame("CMD", self.session, sequence, opcode),
            ("ACK", self.session, sequence, opcode),
        )

    def poll(self) -> list[str]:
        events = []
        for item in self._read():
            if isinstance(item, str):
                events.append(item)
            elif item[0] == "ACK" and len(item) == 6 and item[1] == self.session and item[3] == "PING":
                if item[2] == self._pending_ping and item[4] == "ACCEPTED":
                    self._last_ack = time.monotonic()
                    self._pending_ping = None
            elif item[0] == "NACK" and len(item) >= 5 and item[1] == self.session:
                raise RuntimeError("STM rejected RR2 frame: " + " ".join(item[4:]))
        return events

    def heartbeat(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if now - self._last_ack >= 0.6:
            raise TimeoutError("STM RR2 heartbeat acknowledgement timed out")
        if now - self._last_ping >= 0.1:
            # A lost ACK is retried with the same sequence; the MCU caches
            # its answer and never treats it as a second command.
            sequence = self._pending_ping or str(self._next_seq())
            self._write(encode_frame("PING", self.session, sequence))
            self._pending_ping = sequence
            self._last_ping = now
