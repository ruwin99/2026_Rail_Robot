"""Non-driving RR2 UART probe for a directly connected STM32.

The optional ESTOP step latches the MCU fault and needs a board reset before
another mission. This script never sends a command that can start the drive.
"""

from __future__ import annotations

import argparse
import binascii
import os
import secrets
import time

import serial


def frame(body: str, *, bad_crc: bool = False) -> bytes:
    crc = binascii.crc_hqx(body.encode("ascii"), 0xFFFF)
    if bad_crc:
        crc ^= 0xFFFF
    return f"{body} {crc:04X}\r\n".encode("ascii")


def receive(port: serial.Serial, session: str, kind: str, sequence: int | None,
            opcode: str | None = None, timeout: float = 0.30) -> list[str] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        if os.environ.get("RAIL_PROBE_VERBOSE"):
            print(f"RX {raw!r}", flush=True)
        try:
            line = raw.decode("ascii").strip()
            body, checksum = line.rsplit(" ", 1)
            if binascii.crc_hqx(body.encode("ascii"), 0xFFFF) != int(checksum, 16):
                continue
            tokens = body.split(" ")
        except (UnicodeDecodeError, ValueError):
            continue
        if len(tokens) < 4 or tokens[:3] != ["RR2", kind, session]:
            continue
        if sequence is not None and tokens[3] != str(sequence):
            continue
        if opcode is not None and (len(tokens) < 5 or tokens[4] != opcode):
            continue
        return tokens
    return None


def expect(port: serial.Serial, session: str, body: str, kind: str,
           sequence: int | None, opcode: str | None = None,
           result: str | None = None) -> list[str]:
    port.write(frame(body))
    reply = receive(port, session, kind, sequence, opcode)
    if reply is None or (result is not None and (len(reply) < 6 or reply[5] != result)):
        raise RuntimeError(f"{body}: expected {kind}/{result}, received {reply}")
    print(f"PASS {body.split(' ')[1:]} -> {' '.join(reply[1:])}", flush=True)
    return reply


def run(port_name: str, estop: bool) -> None:
    session = secrets.token_hex(16)
    with serial.Serial(port_name, 115200, timeout=0.02, write_timeout=0.2,
                       exclusive=True) as port:
        port.reset_input_buffer()
        expect(port, session, f"RR2 HELLO {session} MISSION", "READY", None)
        ping = 0

        def heartbeat() -> None:
            nonlocal ping
            ping += 1
            expect(port, session, f"RR2 PING {session} {ping}", "ACK", ping,
                   "PING", "ACCEPTED")

        heartbeat()
        expect(port, session, f"RR2 CMD {session} 1 BOGUS", "NACK", 1,
               "BOGUS", "COMMAND")
        heartbeat()
        expect(port, session, f"RR2 CMD {session} 1 BOGUS", "NACK", 1,
               "BOGUS", "COMMAND")
        expect(port, session, f"RR2 CMD {session} 1 CAPTURE_OK", "NACK", 1,
               "CAPTURE_OK", "CONFLICT")
        heartbeat()
        expect(port, session, f"RR2 CMD {session} 2 CAPTURE_OK", "NACK", 2,
               "CAPTURE_OK", "STATE")
        heartbeat()
        port.write(frame(f"RR2 CMD {session} 3 BOGUS", bad_crc=True))
        rejected = receive(port, session, "ACK", 3, "BOGUS", timeout=0.12)
        if rejected is not None:
            raise RuntimeError("Invalid CRC command was unexpectedly accepted")
        print("PASS invalid CRC command -> no ACK", flush=True)
        heartbeat()
        expect(port, session, f"RR2 CMD {session} 3 BOGUS", "NACK", 3,
               "BOGUS", "COMMAND")

        if estop:
            heartbeat()
            expect(port, session, f"RR2 CMD {session} 4 ESTOP", "ACK", 4,
                   "ESTOP", "ACCEPTED")
            expect(port, session, f"RR2 CMD {session} 4 ESTOP", "ACK", 4,
                   "ESTOP", "ACCEPTED")
            expect(port, session, f"RR2 CMD {session} 5 CAPTURE_OK", "NACK", 5,
                   "CAPTURE_OK", "FAULT")
            print("PASS ESTOP latch: new mission requires STM reset", flush=True)
        else:
            print("CRC/ACK/NACK probe complete; ESTOP was not sent", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--estop", action="store_true")
    args = parser.parse_args()
    run(args.port, args.estop)
