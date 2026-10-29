"""Validated RR2 wire frames; this module never opens a serial port.

The codec validates framing and field grammar, not session ownership, command
permissions, retry state, or actuator-specific argument ranges. Those belong to
the session/command dispatcher. ACK means accepted state, not physical completion.
"""
from __future__ import annotations

from dataclasses import dataclass


MAX_LINE_BYTES = 128
MAX_SEQUENCE = 0xFFFFFFFF
CAPABILITIES = frozenset({
    "MISSION", "ACTUATOR_DEBUG_ENCODER", "ACTUATOR_DEBUG_DC",
    "ACTUATOR_DEBUG_PUMP", "ACTUATOR_DEBUG_ALL", "SENSOR_DEBUG",
})
_KINDS = frozenset({"HELLO", "READY", "CMD", "ACK", "NACK", "EVENT",
                    "EVENT_ACK", "STATUS", "PING"})
_HANDSHAKES = frozenset({"HELLO", "READY"})
_PAYLOAD_LENGTHS = {"HELLO": 1, "READY": 2, "ACK": 3, "NACK": 3,
                    "EVENT_ACK": 1, "PING": 0}


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly=0x1021, init=0xffff, xorout=0."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _decimal(text: str, *, minimum: int = 0) -> int:
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("RR2 numeric fields must be unsigned ASCII decimal tokens")
    value = int(text)
    if not minimum <= value <= MAX_SEQUENCE:
        raise ValueError("RR2 numeric field is outside uint32 range")
    return value


def _fields(frame: Rr2Frame) -> tuple[str, ...]:
    prefix = (frame.kind, frame.session)
    if frame.sequence is not None:
        prefix += (str(frame.sequence),)
    return prefix + frame.payload


@dataclass(frozen=True)
class Rr2Frame:
    """An immutable frame with validated shape, tokens, metadata and wire length."""

    kind: str
    session: str
    sequence: int | None
    payload: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _KINDS:
            raise ValueError("Unknown RR2 frame kind")
        if (type(self.session) is not str or len(self.session) != 32 or
                any(char not in "0123456789abcdef" for char in self.session)):
            raise ValueError("RR2 session must be 32 lowercase hex characters")
        if self.kind in _HANDSHAKES:
            if self.sequence is not None:
                raise ValueError("RR2 handshake frames have no sequence")
        elif type(self.sequence) is not int or not 1 <= self.sequence <= MAX_SEQUENCE:
            raise ValueError("RR2 sequence must be a positive uint32 integer")
        if type(self.payload) is not tuple or any(
            type(token) is not str or not token or
            any(not 33 <= ord(char) <= 126 for char in token)
            for token in self.payload
        ):
            raise ValueError("RR2 payload must be a tuple of nonempty printable ASCII tokens")
        required = _PAYLOAD_LENGTHS.get(self.kind)
        if (required is not None and len(self.payload) != required) or (
            required is None and not self.payload
        ):
            raise ValueError("Wrong payload field count for RR2 frame kind")
        if self.kind in _HANDSHAKES and self.payload[-1] not in CAPABILITIES:
            raise ValueError("Unknown RR2 capability")
        if self.kind == "READY":
            _decimal(self.payload[0], minimum=1)
        if self.kind in ("ACK", "NACK"):
            _decimal(self.payload[2])
        # Prefix + separating space/checksum + CRLF; all fields are ASCII.
        if len("RR2 " + " ".join(_fields(self))) + 7 > MAX_LINE_BYTES:
            raise ValueError("RR2 frame exceeds 128 bytes")


def _from_fields(fields: tuple[str, ...]) -> Rr2Frame:
    if len(fields) < 2 or any(type(field) is not str for field in fields):
        raise ValueError("Missing or invalid RR2 frame fields")
    kind, session, *tail = fields
    sequence = None
    if kind not in _HANDSHAKES:
        if not tail:
            raise ValueError("Missing RR2 sequence")
        sequence = _decimal(tail.pop(0), minimum=1)
    return Rr2Frame(kind, session, sequence, tuple(tail))


def encode_rr2(frame: Rr2Frame) -> bytes:
    """Encode a validated frame with a CRC over the body, excluding CRLF."""
    if not isinstance(frame, Rr2Frame):
        raise ValueError("encode_rr2 requires an Rr2Frame")
    body = ("RR2 " + " ".join(_fields(frame))).encode("ascii")
    return body + f" {crc16_ccitt(body):04X}\r\n".encode("ascii")


def decode_rr2(line: bytes) -> Rr2Frame:
    """Reject malformed lines, including CRC-valid frames with invalid fields."""
    if not isinstance(line, bytes) or len(line) > MAX_LINE_BYTES or not line.endswith(b"\r\n"):
        raise ValueError("RR2 line must be at most 128 bytes and end with CRLF")
    try:
        body, supplied = line[:-2].rsplit(b" ", 1)
        if len(supplied) != 4 or any(byte not in b"0123456789abcdefABCDEF" for byte in supplied):
            raise ValueError("Invalid RR2 CRC token")
        if int(supplied, 16) != crc16_ccitt(body):
            raise ValueError("RR2 CRC mismatch")
        tokens = body.decode("ascii").split(" ")
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid RR2 frame") from exc
    if tokens[0] != "RR2" or any(not token for token in tokens):
        raise ValueError("Invalid RR2 prefix or field separator")
    return _from_fields(tuple(tokens[1:]))


class Rr2StreamParser:
    """Bounded incremental parser; a bad input chunk yields no usable frames.

    Invalid complete lines raise ValueError. Oversized lines are discarded through
    their next LF, even across feed calls, so a suffix cannot become a fresh frame.
    The caller must apply its fault policy on ValueError; parsing can subsequently
    resume at the next line boundary. An incomplete line alone is not a command.
    """

    def __init__(self) -> None:
        self._pending = bytearray()
        self._discarding = False

    def feed(self, chunk: bytes) -> list[Rr2Frame]:
        if not isinstance(chunk, bytes):
            raise ValueError("RR2 stream chunks must be bytes")
        frames: list[Rr2Frame] = []
        error = None
        for byte in chunk:
            if self._discarding:
                if byte == 10:
                    self._discarding = False
                continue
            if len(self._pending) == MAX_LINE_BYTES:
                self._pending.clear()
                self._discarding = byte != 10
                error = ValueError("RR2 frame exceeds 128 bytes")
                continue
            self._pending.append(byte)
            if byte == 10:
                line = bytes(self._pending)
                self._pending.clear()
                try:
                    frame = decode_rr2(line)
                except ValueError as exc:
                    error = exc
                else:
                    if error is None:
                        frames.append(frame)
        if error is not None:
            raise error
        return frames
