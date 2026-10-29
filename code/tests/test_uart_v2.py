"""Wire-level checks for RR2 mission and device-debug sessions."""

import ast
import binascii
import importlib
import importlib.util
import re
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))

from node_runtime.uart_v2 import UartV2, crc16_ccitt, decode_frame, encode_frame


SESSION = "0123456789abcdef0123456789abcdef"
VECTOR_HEADER = Path(__file__).parent / "fixtures" / "rr2_vectors.h"


def parse_rr2_vector_header():
    """Read the C header instead of maintaining a separate Python fixture copy."""
    definitions = dict(re.findall(
        r"^#define (RR2_VECTOR_\w+) (.+)$", VECTOR_HEADER.read_text(encoding="ascii"), re.M
    ))
    return {name.removeprefix("RR2_VECTOR_"): (
        ast.literal_eval(value) if value.startswith('"') else int(value.rstrip("U"), 16)
    ) for name, value in definitions.items()}


def fixture_wire(body):
    """An independent CRC oracle exercises field validation after a valid CRC."""
    data = body.encode("ascii")
    return data + f" {binascii.crc_hqx(data, 0xffff):04X}\r\n".encode("ascii")


class Rr2CodecTests(unittest.TestCase):
    def codec(self):
        self.assertIsNotNone(importlib.util.find_spec("node_runtime.rr2_protocol"),
                             "The shared RR2 codec has not been implemented")
        return importlib.import_module("node_runtime.rr2_protocol")

    def test_shared_c_header_vectors_match_independent_crc_and_round_trip(self):
        codec = self.codec()
        vectors = parse_rr2_vector_header()
        self.assertEqual(vectors["CRC_VALUE"], 0x29B1)
        self.assertEqual(codec.crc16_ccitt(vectors["CRC_INPUT"].encode()), 0x29B1)
        kinds = {key[:-5] for key in vectors if key.endswith("_BODY")}
        self.assertEqual(kinds, {"HELLO", "READY", "CMD", "ACK", "NACK", "EVENT",
                                 "EVENT_ACK", "STATUS", "PING"})
        for kind in sorted(kinds):
            with self.subTest(kind=kind):
                wire = vectors[kind + "_WIRE"].encode("ascii")
                self.assertEqual(fixture_wire(vectors[kind + "_BODY"]), wire)
                frame = codec.decode_rr2(wire)
                self.assertEqual(frame.kind, kind)
                self.assertEqual(frame.session, SESSION)
                self.assertEqual(frame.sequence, None if kind in ("HELLO", "READY") else
                                 (2 if kind in ("NACK", "PING") else 1))
                self.assertEqual(codec.encode_rr2(frame), wire)

    def test_negotiated_capabilities_round_trip(self):
        codec = self.codec()
        for capability in ("MISSION", "ACTUATOR_DEBUG_ENCODER", "ACTUATOR_DEBUG_DC",
                           "ACTUATOR_DEBUG_PUMP", "ACTUATOR_DEBUG_ALL", "SENSOR_DEBUG"):
            for kind, payload in (("HELLO", (capability,)), ("READY", ("2", capability))):
                with self.subTest(kind=kind, capability=capability):
                    frame = codec.Rr2Frame(kind, SESSION, None, payload)
                    self.assertEqual(codec.decode_rr2(codec.encode_rr2(frame)), frame)

    def test_sequence_boundary_and_payload_args_survive_round_trip(self):
        codec = self.codec()
        frame = codec.Rr2Frame("CMD", SESSION, 0xffffffff, ("SET", "-1", "1250", "0"))
        self.assertEqual(codec.decode_rr2(codec.encode_rr2(frame)), frame)

    def test_invalid_frame_fields_are_rejected_at_construction(self):
        codec = self.codec()
        invalid = [
            ("UNKNOWN", SESSION, 1, ()), ("PING", "a" * 31, 1, ()),
            ("PING", "A" * 32, 1, ()), ("PING", "g" * 32, 1, ()),
            ("PING", SESSION, 0, ()), ("PING", SESSION, -1, ()),
            ("PING", SESSION, 0x100000000, ()), ("PING", SESSION, True, ()),
            ("PING", SESSION, 1.0, ()), ("PING", SESSION, "1", ()),
            ("PING", SESSION, None, ()), ("HELLO", SESSION, 1, ("MISSION",)),
            ("HELLO", SESSION, None, ()), ("HELLO", SESSION, None, ("UNSUPPORTED",)),
            ("READY", SESSION, None, ("2",)), ("READY", SESSION, None, ("two", "MISSION")),
            ("READY", SESSION, None, ("0", "MISSION")),
            ("READY", SESSION, None, ("2", "UNSUPPORTED")),
            ("CMD", SESSION, 1, ()), ("EVENT", SESSION, 1, ()),
            ("STATUS", SESSION, 1, ()), ("PING", SESSION, 1, ("EXTRA",)),
            ("ACK", SESSION, 1, ("START", "ACCEPTED")),
            ("ACK", SESSION, 1, ("START", "ACCEPTED", "NaN")),
            ("NACK", SESSION, 1, ("START", "STATE", "-1")),
            ("EVENT_ACK", SESSION, 1, ()),
            ("EVENT_ACK", SESSION, 1, ("ACCEPTED", "EXTRA")),
            ("CMD", SESSION, 1, ["START"]),
        ]
        invalid += [("CMD", SESSION, 1, (token,))
                    for token in ("", "TWO WORDS", "A\tB", "A\rB", "A\nB", "\x00", "\x7f", "é", 1)]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                codec.Rr2Frame(*fields)

    def test_valid_crc_does_not_bypass_field_validation(self):
        codec = self.codec()
        bodies = [
            "RR2 HELLO " + SESSION,
            f"RR2 CMD {SESSION} 0 START", f"RR2 CMD {SESSION} -1 START",
            f"RR2 CMD {SESSION} +1 START", f"RR2 CMD {SESSION} 1.0 START",
            f"RR2 CMD {SESSION} 4294967296 START", f"RR2 CMD {SESSION} 1",
            f"RR2 CMD {'A' * 32} 1 START", f"RR2 PING {SESSION} 1 EXTRA",
            f"RR2 ACK {SESSION} 1 START ACCEPTED 1 EXTRA",
            f"RR2 ACK {SESSION} 1 START ACCEPTED NaN",
            f"RR2 CMD {SESSION} 1 A\x00B", f"RR2 CMD {SESSION} 1 A\tB",
            f"RR2  CMD {SESSION} 1 START", f"RR2 UNKNOWN {SESSION} 1",
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                codec.decode_rr2(fixture_wire(body))

    def test_decoder_requires_exact_crlf_and_four_hex_crc_digits(self):
        codec = self.codec()
        wire = parse_rr2_vector_header()["ACK_WIRE"].encode("ascii")
        invalid = [wire[:-1], wire[:-2], wire.replace(b"\r\n", b"\n"), wire + b"\r\n",
                   wire.replace(b"4853", b"+853"), wire.replace(b"4853", b"485"),
                   wire.replace(b"START", b"STARP"), b"STARTED\r\n"]
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(ValueError):
                codec.decode_rr2(line)

    def test_wire_limit_includes_crc_and_crlf(self):
        codec = self.codec()
        # 53 fixed bytes for this STATUS frame leave 75 bytes of payload.
        frame = codec.Rr2Frame("STATUS", SESSION, 1, ("X" * 75,))
        wire = codec.encode_rr2(frame)
        self.assertEqual(len(wire), 128)
        self.assertEqual(codec.decode_rr2(wire), frame)
        with self.assertRaises(ValueError):
            codec.Rr2Frame("STATUS", SESSION, 1, ("X" * 76,))
        with self.assertRaises(ValueError):
            codec.decode_rr2(fixture_wire(f"RR2 STATUS {SESSION} 1 " + "X" * 76))

    def test_stream_accepts_every_chunk_boundary_for_shared_vectors(self):
        codec = self.codec()
        vectors = parse_rr2_vector_header()
        for key, value in vectors.items():
            if not key.endswith("_WIRE"):
                continue
            wire = value.encode("ascii")
            expected = codec.decode_rr2(wire)
            for split in range(len(wire) + 1):
                with self.subTest(vector=key, split=split):
                    parser = codec.Rr2StreamParser()
                    self.assertEqual(parser.feed(wire[:split]) + parser.feed(wire[split:]), [expected])

    def test_stream_preserves_multiple_frames_and_byte_at_a_time_input(self):
        codec = self.codec()
        vectors = parse_rr2_vector_header()
        wires = [value.encode("ascii") for key, value in vectors.items() if key.endswith("_WIRE")]
        expected = [codec.decode_rr2(wire) for wire in wires]
        stream = b"".join(wires)
        self.assertEqual(codec.Rr2StreamParser().feed(stream), expected)
        parser, actual = codec.Rr2StreamParser(), []
        for byte in stream:
            actual.extend(parser.feed(bytes([byte])))
        self.assertEqual(actual, expected)

    def test_stream_waits_for_lf_after_cr(self):
        codec = self.codec()
        wire = parse_rr2_vector_header()["ACK_WIRE"].encode("ascii")
        parser = codec.Rr2StreamParser()
        self.assertEqual(parser.feed(wire[:-1]), [])
        self.assertEqual(parser.feed(b""), [])
        self.assertEqual(parser.feed(b"\n"), [codec.decode_rr2(wire)])

    def test_stream_rejects_bad_crc_and_resynchronizes_at_line_boundary(self):
        codec = self.codec()
        wire = parse_rr2_vector_header()["ACK_WIRE"].encode("ascii")
        parser = codec.Rr2StreamParser()
        with self.assertRaises(ValueError):
            parser.feed(wire.replace(b"START", b"STARP"))
        self.assertEqual(parser.feed(wire), [codec.decode_rr2(wire)])

    def test_stream_overflow_discards_remainder_of_damaged_line(self):
        codec = self.codec()
        wire = parse_rr2_vector_header()["ACK_WIRE"].encode("ascii")
        parser = codec.Rr2StreamParser()
        with self.assertRaises(ValueError):
            parser.feed(b"X" * 100000)
        self.assertEqual(parser.feed(wire), [])
        self.assertEqual(parser.feed(wire), [codec.decode_rr2(wire)])

    def test_compatibility_helpers_use_same_validation(self):
        for fields in (("CMD", SESSION, "0", "START"), ("PING", SESSION, "1", "EXTRA"),
                       ("ACK", SESSION, "1", "START", "ACCEPTED", "NaN")):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                encode_frame(*fields)
        with self.assertRaises(ValueError):
            decode_frame(fixture_wire(f"RR2 PING {SESSION} 0"))

    def test_compatibility_handshake_defaults_to_mission_capability(self):
        vectors = parse_rr2_vector_header()
        self.assertEqual(encode_frame("HELLO", SESSION), vectors["HELLO_WIRE"].encode())
        self.assertEqual(encode_frame("READY", SESSION, "2"), vectors["READY_WIRE"].encode())


class FakeSerial:
    def __init__(self, on_write=None):
        self.on_write = on_write
        self.pending = bytearray()
        self.writes = []

    @property
    def in_waiting(self):
        return len(self.pending)

    def read(self, size):
        result = bytes(self.pending[:size])
        del self.pending[:size]
        return result

    def write(self, data):
        self.writes.append(data)
        if self.on_write:
            self.on_write(self, data)
        return len(data)

    def close(self):
        pass


class UartV2Tests(unittest.TestCase):
    def test_crc_known_vector_and_corruption(self):
        self.assertEqual(crc16_ccitt(b"123456789"), 0x29B1)
        line = encode_frame("HELLO", "a" * 32)
        self.assertEqual(decode_frame(line), ("HELLO", "a" * 32, "MISSION"))
        with self.assertRaises(ValueError):
            decode_frame(line.replace(b"HELLO", b"HELLU"))

    def test_hello_and_command_wait_for_mcu_ack_and_preserve_events(self):
        session = "a" * 32

        def reply(serial, data):
            frame = decode_frame(data)
            if frame[0] == "HELLO":
                serial.pending.extend(encode_frame("READY", session, "2"))
            elif frame[0] == "CMD":
                serial.pending.extend(b"STARTED\r\n")
                serial.pending.extend(
                    encode_frame("ACK", session, frame[2], frame[3], "ACCEPTED", "1")
                )

        serial = FakeSerial(reply)
        link = UartV2(serial, session=session, timeout=0.01, attempts=2)
        self.assertEqual(link.hello(), [])
        self.assertEqual(link.command("START"), ["STARTED"])
        self.assertEqual(decode_frame(serial.writes[-1])[:2], ("CMD", session))

    def test_lost_ack_retries_same_sequence(self):
        session = "b" * 32
        seen = []

        def reply(serial, data):
            frame = decode_frame(data)
            if frame[0] == "CMD":
                seen.append(frame[2])
                if len(seen) == 2:
                    serial.pending.extend(encode_frame("ACK", session, frame[2], "CAPTURE_OK", "ACCEPTED", "2"))

        serial = FakeSerial(reply)
        link = UartV2(serial, session=session, timeout=0.002, attempts=2)
        self.assertEqual(link.command("CAPTURE_OK"), [])
        self.assertEqual(seen, [seen[0], seen[0]])

    def test_event_after_ack_in_same_read_is_not_lost(self):
        session = "f" * 32

        def reply(serial, data):
            frame = decode_frame(data)
            serial.pending.extend(encode_frame("ACK", session, frame[2], "START", "ACCEPTED", "1"))
            serial.pending.extend(b"STARTED\r\n")

        link = UartV2(FakeSerial(reply), session=session, timeout=0.01, attempts=1)
        self.assertEqual(link.command("START"), ["STARTED"])

    def test_wrong_session_and_crc_do_not_count_as_ack(self):
        session = "c" * 32

        def reply(serial, data):
            frame = decode_frame(data)
            serial.pending.extend(encode_frame("ACK", "d" * 32, frame[2], "START", "ACCEPTED", "1"))

        link = UartV2(FakeSerial(reply), session=session, timeout=0.002, attempts=1)
        with self.assertRaises(TimeoutError):
            link.command("START")

    def test_oversize_line_is_rejected(self):
        serial = FakeSerial()
        serial.pending.extend(b"X" * 130 + b"\n")
        link = UartV2(serial, session="e" * 32)
        with self.assertRaises(ValueError):
            link.poll()

    def test_split_oversize_line_discards_rr2_suffix_through_first_lf(self):
        serial = FakeSerial()
        link = UartV2(serial, session=SESSION)
        wire = parse_rr2_vector_header()["ACK_WIRE"].encode("ascii")
        serial.pending.extend(b"X" * 129)
        with self.assertRaises(ValueError):
            link._read()
        # This valid-looking ACK is still part of the damaged, unterminated line.
        serial.pending.extend(wire[:-1])
        self.assertEqual(link._read(), [])
        self.assertEqual(link._read(), [])
        # Only the complete frame following the first LF is eligible for parsing.
        serial.pending.extend(b"\n" + wire)
        self.assertEqual(link._read(), [decode_frame(wire)])

    def test_missing_ping_ack_retries_same_sequence(self):
        serial = FakeSerial()
        link = UartV2(serial, session="a" * 32)
        now = link._last_ack + 0.11
        link.heartbeat(now)
        link.heartbeat(now + 0.11)
        self.assertEqual(len(serial.writes), 2)
        self.assertEqual(serial.writes[0], serial.writes[1])

    def test_stale_link_rejects_motion_before_writing_command(self):
        serial = FakeSerial()
        link = UartV2(serial, session="a" * 32)
        link._last_ack -= 0.2
        with self.assertRaises(TimeoutError):
            link.command("START")
        self.assertEqual(serial.writes, [])


    def test_corrupted_ack_is_retried_and_never_accepted(self):
        def reply(serial, data):
            frame = decode_frame(data)
            body = f"RR2 ACK {SESSION} {frame[2]} START ACCEPTED 1"
            serial.pending.extend(fixture_wire(body).replace(b"ACCEPTED", b"ACCEPTEF"))

        serial = FakeSerial(reply)
        link = UartV2(serial, session=SESSION, timeout=0.002, attempts=3)
        with self.assertRaises(TimeoutError):
            link.command("START")
        self.assertEqual(len(serial.writes), 3)
        self.assertEqual(serial.writes, [serial.writes[0]] * 3)

    def test_matching_ack_with_invalid_state_is_not_accepted(self):
        def reply(serial, data):
            frame = decode_frame(data)
            serial.pending.extend(fixture_wire(f"RR2 ACK {SESSION} {frame[2]} START ACCEPTED NaN"))

        link = UartV2(FakeSerial(reply), session=SESSION, timeout=0.002, attempts=1)
        with self.assertRaises(TimeoutError):
            link.command("START")

    def test_ready_for_other_capability_is_not_mission_handshake(self):
        def reply(serial, _data):
            serial.pending.extend(fixture_wire(f"RR2 READY {SESSION} 2 SENSOR_DEBUG"))

        link = UartV2(FakeSerial(reply), session=SESSION, timeout=0.002, attempts=1)
        with self.assertRaises(TimeoutError):
            link.hello()


class SerialWorkerTests(unittest.TestCase):
    def worker_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("node_runtime.serial_worker"),
            "The serial-owner worker has not been implemented",
        )
        return importlib.import_module("node_runtime.serial_worker")

    def test_ready_and_first_debug_status_in_same_read_are_both_retained(self):
        module = self.worker_module()

        def reply(serial, data):
            frame = decode_frame(data)
            if frame[0] == "HELLO":
                serial.pending.extend(fixture_wire(f"RR2 READY {SESSION} 2 SENSOR_DEBUG"))
                serial.pending.extend(fixture_wire(
                    f"RR2 STATUS {SESSION} 1 ULTRASONIC 100 180 1 1 0 1 0 0 0 ECHO"))
            elif frame[0] == "PING":
                serial.pending.extend(fixture_wire(
                    f"RR2 ACK {SESSION} {frame[2]} PING ACCEPTED 0"))

        worker = module.SerialWorker(lambda **kw: FakeSerial(reply), serial_kwargs={},
                                     session=SESSION, capability="SENSOR_DEBUG",
                                     poll_interval=.001)
        try:
            self.assertTrue(worker.wait_ready(.5), worker.startup_error)
            self.assertEqual([item.payload[0] for item in worker.drain_results(10)], ["ULTRASONIC"])
        finally:
            worker.stop(.5)

    def test_event_queue_overflow_never_acknowledges_an_undelivered_event(self):
        module = self.worker_module()
        instances = []

        def reply(serial, data):
            frame = decode_frame(data)
            if frame[0] == "HELLO":
                serial.pending.extend(encode_frame("READY", SESSION, "2"))
            elif frame[0] == "PING":
                serial.pending.extend(encode_frame(
                    "ACK", SESSION, frame[2], "PING", "ACCEPTED", "0"))

        def create_serial(**_kwargs):
            serial = FakeSerial(reply)
            instances.append(serial)
            return serial

        worker = module.SerialWorker(create_serial, serial_kwargs={}, session=SESSION,
                                     result_capacity=1, poll_interval=.001,
                                     ack_timeout=.05, retry_limit=2)
        try:
            self.assertTrue(worker.wait_ready(.5), worker.startup_error)
            serial = instances[0]
            serial.pending.extend(encode_frame("EVENT", SESSION, "1", "STARTED"))
            deadline = time.monotonic() + .5
            while time.monotonic() < deadline and not any(
                    decode_frame(wire)[:3] == ("EVENT_ACK", SESSION, "1")
                    for wire in serial.writes):
                time.sleep(.002)
            self.assertTrue(any(decode_frame(wire)[:3] == ("EVENT_ACK", SESSION, "1")
                                for wire in serial.writes))
            active_start = worker.submit("START")
            deadline = time.monotonic() + .5
            while time.monotonic() < deadline and not any(
                    decode_frame(wire)[0:4] == ("CMD", SESSION, "1", "START")
                    for wire in serial.writes):
                time.sleep(.002)
            self.assertTrue(any(decode_frame(wire)[0:4] == ("CMD", SESSION, "1", "START")
                                for wire in serial.writes))
            serial.pending.extend(encode_frame("EVENT", SESSION, "2", "CAMERA_CAPTURE"))
            deadline = time.monotonic() + .5
            while time.monotonic() < deadline and worker.ready:
                time.sleep(.002)
            self.assertFalse(worker.ready)
            self.assertEqual([item.payload for item in worker.drain_results(10)],
                             [("STARTED",)])
            self.assertFalse(any(decode_frame(wire)[:3] == ("EVENT_ACK", SESSION, "2")
                                 for wire in serial.writes))
            self.assertFalse(active_start.result(timeout=.2).success)
            time.sleep(.12)
            self.assertEqual(sum(decode_frame(wire)[0:4] == ("CMD", SESSION, "1", "START")
                                 for wire in serial.writes), 1)
        finally:
            worker.stop(.5)

    def test_debug_capability_handshake_restricts_commands_and_owns_io(self):
        module = self.worker_module()
        owner_threads, frames = set(), []

        class DebugSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                owner_threads.add(threading.get_ident())

            @property
            def in_waiting(self):
                owner_threads.add(threading.get_ident())
                return super().in_waiting

            def read(self, size):
                owner_threads.add(threading.get_ident())
                return super().read(size)

            def write(self, data):
                owner_threads.add(threading.get_ident())
                frame = decode_frame(data)
                frames.append(frame)
                if frame[0] == "HELLO":
                    self.pending.extend(fixture_wire(f"RR2 READY {SESSION} 2 ACTUATOR_DEBUG_ENCODER"))
                elif frame[0] == "CMD":
                    self.pending.extend(fixture_wire(
                        f"RR2 ACK {SESSION} {frame[2]} {frame[3]} ACCEPTED 0"))
                elif frame[0] == "PING":
                    self.pending.extend(fixture_wire(
                        f"RR2 ACK {SESSION} {frame[2]} PING ACCEPTED 0"))
                return len(data)

            def close(self):
                owner_threads.add(threading.get_ident())

        worker = module.SerialWorker(DebugSerial, serial_kwargs={"port": "fake"},
                                     session=SESSION, capability="ACTUATOR_DEBUG_ENCODER",
                                     poll_interval=0.001)
        try:
            self.assertTrue(worker.wait_ready(0.5), worker.startup_error)
            self.assertFalse(worker.submit("START").result().success)
            self.assertFalse(worker.submit("LED", ("auto", "0", "0", "0")).result().success)
            self.assertFalse(worker.submit("SET", ("0", "0", "1")).result().success)
            self.assertTrue(worker.submit("SET", ("1", "1250", "0")).result(timeout=0.5).success)
            self.assertEqual(frames[0], ("HELLO", SESSION, "ACTUATOR_DEBUG_ENCODER"))
            self.assertIn(("CMD", SESSION, "1", "SET", "1", "1250", "0"), frames)
        finally:
            self.assertTrue(worker.stop(0.5))
        self.assertIn("STOP", [frame[3] for frame in frames if frame[0] == "CMD"])
        self.assertEqual(len(owner_threads), 1)

    def test_sensor_debug_led_validation_and_stop_priority(self):
        module = self.worker_module()
        frames = []

        def reply(serial, data):
            frame = decode_frame(data)
            frames.append(frame)
            if frame[0] == "HELLO":
                serial.pending.extend(fixture_wire(f"RR2 READY {SESSION} 2 SENSOR_DEBUG"))
            elif frame[0] == "CMD":
                serial.pending.extend(fixture_wire(
                    f"RR2 ACK {SESSION} {frame[2]} {frame[3]} ACCEPTED 0"))
            elif frame[0] == "PING":
                serial.pending.extend(fixture_wire(
                    f"RR2 ACK {SESSION} {frame[2]} PING ACCEPTED 0"))

        worker = module.SerialWorker(lambda **kw: FakeSerial(reply), serial_kwargs={},
                                     session=SESSION, capability="SENSOR_DEBUG",
                                     poll_interval=0.001)
        try:
            self.assertTrue(worker.wait_ready(0.5), worker.startup_error)
            self.assertFalse(worker.submit("SET", ("0", "0", "0")).result().success)
            self.assertFalse(worker.submit("LED", ("bogus", "0", "0", "0")).result().success)
            self.assertFalse(worker.submit("LED", ("manual", "256", "0", "0")).result().success)
            self.assertTrue(worker.submit("LED", ("manual", "255", "0", "4")).result(timeout=0.5).success)
            self.assertTrue(worker.latch_fault().result(timeout=0.5).success)
            self.assertFalse(worker.submit("LED", ("auto", "0", "0", "0")).result().success)
        finally:
            self.assertTrue(worker.stop(0.5))
        self.assertIn(("CMD", SESSION, "1", "LED", "manual", "255", "0", "4"), frames)
        self.assertIn("STOP", [frame[3] for frame in frames if frame[0] == "CMD"])

    def test_owner_thread_handshake_ack_io_and_close_once(self):
        module = self.worker_module()
        io_threads, instances = set(), []

        class OwnedSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.close_count = 0
                io_threads.add(threading.get_ident())
                instances.append(self)

            @property
            def in_waiting(self):
                io_threads.add(threading.get_ident())
                return super().in_waiting

            def read(self, size):
                io_threads.add(threading.get_ident())
                return super().read(size)

            def write(self, data):
                io_threads.add(threading.get_ident())
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                    ))
                return super().write(data)

            def close(self):
                io_threads.add(threading.get_ident())
                self.close_count += 1

        worker = module.SerialWorker(
            OwnedSerial, serial_kwargs={"port": "fake"}, session=SESSION,
            ack_timeout=0.02, retry_limit=2, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            result = worker.submit("START").result(timeout=0.5)
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.session_id, SESSION)
        finally:
            worker.stop(0.5)
            worker.stop(0.5)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].close_count, 1)
        self.assertEqual(len(io_threads), 1)

    def test_retry_reuses_same_wire_frame_and_has_a_hard_cap(self):
        module = self.worker_module()
        instances = []

        class RetrySerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.command_writes = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    self.command_writes.append(data)
                    if len(self.command_writes) == 2:
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                        ))
                return super().write(data)

        worker = module.SerialWorker(
            RetrySerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.015, retry_limit=2, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            result = worker.submit("CAPTURE_OK").result(timeout=0.5)
            self.assertTrue(result.success, result.error)
            self.assertEqual(len(instances[0].command_writes), 2)
            self.assertEqual(instances[0].command_writes[0], instances[0].command_writes[1])
        finally:
            worker.stop(0.5)

    def test_event_ack_and_heartbeat_continue_during_command_ack_wait(self):
        module = self.worker_module()
        instances, event_ack_seen, ack_sent = [], threading.Event(), threading.Event()

        class SlowAckSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.pings = 0
                self.ping_sequences = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    self.pending.extend(encode_frame("EVENT", frame[1], "1", "STARTED"))
                    def delayed_ack():
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                        ))
                        ack_sent.set()
                    threading.Timer(0.08, delayed_ack).start()
                elif frame[0] == "PING":
                    self.pings += 1
                    self.ping_sequences.append(frame[2])
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                    ))
                elif frame[0] == "EVENT_ACK":
                    event_ack_seen.set()
                return super().write(data)

        worker = module.SerialWorker(
            SlowAckSerial, serial_kwargs={}, session=SESSION, ack_timeout=0.12,
            retry_limit=1, heartbeat_interval=0.015, heartbeat_timeout=0.3,
            poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            result = worker.submit("START").result(timeout=0.5)
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.sequence, 1)
            self.assertTrue(ack_sent.is_set())
            self.assertTrue(event_ack_seen.wait(0.2))
            self.assertGreaterEqual(instances[0].pings, 1)
            self.assertEqual(instances[0].ping_sequences[0], "1")
            self.assertTrue(any(item.kind == "EVENT" and item.payload == ("STARTED",)
                                for item in worker.drain_results(8)))
        finally:
            worker.stop(0.5)

    def test_opposite_lane_nack_does_not_fail_active_command_with_same_sequence(self):
        module = self.worker_module()

        class OppositeNackSerial(FakeSerial):
            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                    ))
                elif frame[0] == "CMD" and frame[3] == "START":
                    self.pending.extend(encode_frame(
                        "NACK", frame[1], frame[2], "PING", "INVALID_STATE", "1"
                    ))
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "START", "ACCEPTED", "1"
                    ))
                return super().write(data)

        worker = module.SerialWorker(
            OppositeNackSerial, serial_kwargs={}, session=SESSION,
            poll_interval=0.001, heartbeat_interval=0.02,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            result = worker.submit("START").result(timeout=0.3)
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.sequence, 1)
            self.assertTrue(worker.ready)
        finally:
            worker.stop(0.5)

    def test_opposite_opcode_nack_does_not_fail_ping_with_same_sequence(self):
        module = self.worker_module()
        ping_sent = threading.Event()

        class PingNackSerial(FakeSerial):
            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    ping_sent.set()
                    self.pending.extend(encode_frame(
                        "NACK", frame[1], frame[2], "START", "INVALID_STATE", "1"
                    ))
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                    ))
                elif frame[0] == "CMD":
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                    ))
                return super().write(data)

        worker = module.SerialWorker(
            PingNackSerial, serial_kwargs={}, session=SESSION,
            poll_interval=0.001, heartbeat_interval=0.02,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            self.assertTrue(ping_sent.wait(0.3))
            time.sleep(0.02)
            self.assertTrue(worker.ready)
            result = worker.submit("START").result(timeout=0.3)
            self.assertTrue(result.success, result.error)
            self.assertTrue(worker.ready)
        finally:
            worker.stop(0.5)

    def test_write_failure_resolves_dequeued_command_future(self):
        module = self.worker_module()

        for failure_mode in ("raise", "partial"):
            with self.subTest(failure_mode=failure_mode):
                class FailingWriteSerial(FakeSerial):
                    def write(self, data):
                        frame = decode_frame(data)
                        if frame[0] == "HELLO":
                            self.pending.extend(encode_frame("READY", frame[1], "2"))
                        elif frame[0] == "CMD":
                            if failure_mode == "raise":
                                raise OSError("injected serial write failure")
                            super().write(data)
                            return len(data) - 1
                        return super().write(data)

                worker = module.SerialWorker(
                    FailingWriteSerial, serial_kwargs={}, session=SESSION,
                    ack_timeout=0.03, retry_limit=1, poll_interval=0.001,
                )
                try:
                    self.assertTrue(worker.wait_ready(0.5))
                    future = worker.submit("START")
                    result = future.result(timeout=0.2)
                    self.assertFalse(result.success)
                    self.assertIn("write", result.error.lower())
                finally:
                    worker.stop(0.5, safe_off=False)

    def test_safe_off_waits_behind_active_on_and_precedes_later_on(self):
        module = self.worker_module()
        instances, active_on_sent, release_ack = [], threading.Event(), threading.Event()

        class ActiveOnSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD" and frame[3] == "FRONT_PUMP_ON":
                    result = super().write(data)
                    active_on_sent.set()
                    if release_ack.wait(0.5):
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                        ))
                    return result
                elif frame[0] == "CMD":
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                    ))
                return super().write(data)

        worker = module.SerialWorker(
            ActiveOnSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.2, retry_limit=1, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            active = worker.submit("FRONT_PUMP_ON", coalesce_key="front_pump")
            self.assertTrue(active_on_sent.wait(0.3))
            safe_off = worker.submit("SAFE_OFF")
            later_on = worker.submit("SIDE_PUMP_ON", coalesce_key="side_pump")
            self.assertFalse(later_on.result(timeout=0.1).success)
            time.sleep(0.02)
            sent_before_active_finishes = [
                decode_frame(frame)[3] for frame in instances[0].writes
                if decode_frame(frame)[0] == "CMD"
            ]
            self.assertEqual(sent_before_active_finishes, ["FRONT_PUMP_ON"])
            release_ack.set()
            self.assertTrue(active.result(timeout=0.3).success)
            self.assertTrue(safe_off.result(timeout=0.5).success)
            opcodes = [
                decode_frame(frame)[3] for frame in instances[0].writes
                if decode_frame(frame)[0] == "CMD"
            ]
            off_index = next(i for i, opcode in enumerate(opcodes)
                             if opcode == "FRONT_CLEANER_OFF")
            self.assertEqual(opcodes[0], "FRONT_PUMP_ON")
            self.assertTrue(all(opcode.endswith("_OFF") for opcode in opcodes[off_index:]))
        finally:
            release_ack.set()
            worker.stop(0.5)

    def test_safety_command_uses_reserved_priority_and_blocks_later_on(self):
        module = self.worker_module()
        instances, active_sent, release_ack = [], threading.Event(), threading.Event()

        class PrioritySerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD" and frame[3] == "CAPTURE_OK":
                    active_sent.set()
                    if release_ack.wait(0.3):
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                        ))
                elif frame[0] == "CMD" and frame[3].endswith("_OFF"):
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                    ))
                return super().write(data)

        worker = module.SerialWorker(
            PrioritySerial, serial_kwargs={}, session=SESSION, queue_capacity=1,
            ack_timeout=0.03, retry_limit=1, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            active = worker.submit("CAPTURE_OK")
            self.assertTrue(active_sent.wait(0.3))
            queued_on = worker.submit("FRONT_CLEANER_PWM_12_5", coalesce_key="front_cleaner")
            overflow = worker.submit("FRONT_PUMP_ON", coalesce_key="front_pump")
            self.assertFalse(overflow.result(timeout=0.1).success)
            self.assertIn("queue is full", overflow.result().error)
            safe_off = worker.submit("SAFE_OFF")
            for _ in range(100):
                self.assertIs(worker.submit("SAFE_OFF"), safe_off)
            self.assertEqual(len(worker._safety.futures), 1)
            release_ack.set()
            self.assertTrue(active.result(timeout=0.5).success)
            self.assertTrue(safe_off.result(timeout=1.0).success)
            self.assertFalse(queued_on.result(timeout=0.5).success)
            opcodes = [decode_frame(frame)[3] for frame in instances[0].writes
                       if decode_frame(frame)[0] == "CMD"]
            self.assertNotIn("FRONT_CLEANER_PWM_12_5", opcodes)
            self.assertIn("FRONT_CLEANER_OFF", opcodes)
        finally:
            release_ack.set()
            worker.stop(0.5)

    def test_estop_preempts_pending_command_and_latches_worker(self):
        module = self.worker_module()
        instances, active_sent = [], threading.Event()

        def reply(serial, data):
            frame = decode_frame(data)
            if frame[0] == "HELLO":
                serial.pending.extend(encode_frame("READY", SESSION, "2"))
            elif frame[0] == "PING":
                serial.pending.extend(encode_frame(
                    "ACK", SESSION, frame[2], "PING", "ACCEPTED", "0"))
            elif frame[0] == "CMD" and frame[3] == "CAPTURE_OK":
                active_sent.set()
            elif frame[0] == "CMD" and frame[3] == "ESTOP":
                serial.pending.extend(encode_frame(
                    "ACK", SESSION, frame[2], "ESTOP", "ACCEPTED", "10"))

        def create_serial(**_kwargs):
            serial = FakeSerial(reply)
            instances.append(serial)
            return serial

        worker = module.SerialWorker(
            create_serial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.1, retry_limit=2, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5), worker.startup_error)
            active = worker.submit("CAPTURE_OK")
            self.assertTrue(active_sent.wait(0.2))
            queued = worker.submit("FRONT_PUMP_ON")
            estop = worker.emergency_stop()
            self.assertTrue(estop.result(timeout=0.5).success)
            self.assertFalse(active.result(timeout=0.1).success)
            self.assertFalse(queued.result(timeout=0.1).success)
            self.assertIs(worker.emergency_stop(), estop)
            self.assertIs(worker.submit("SAFE_OFF"), estop)
            self.assertFalse(worker.submit("START").result(timeout=0.1).success)
            opcodes = [decode_frame(wire)[3] for wire in instances[0].writes
                       if decode_frame(wire)[0] == "CMD"]
            self.assertEqual(opcodes, ["CAPTURE_OK", "ESTOP"])
            self.assertTrue(worker._heartbeats_suspended)
        finally:
            worker.stop(0.5)

    def test_acknowledged_safe_off_is_idempotent_until_on_or_forced_retry(self):
        module = self.worker_module()
        instances = []

        class OffSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                    ))
                return super().write(data)

        worker = module.SerialWorker(
            OffSerial, serial_kwargs={}, session=SESSION, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            first = worker.submit("SAFE_OFF")
            self.assertTrue(first.result(timeout=0.5).success)
            def off_writes():
                count = 0
                for wire in instances[0].writes:
                    fields = decode_frame(wire)
                    if fields[0] == "CMD" and fields[3].endswith("_OFF"):
                        count += 1
                return count
            count = off_writes()
            self.assertEqual(count, 4)

            repeated = worker.submit("SAFE_OFF")
            self.assertTrue(repeated.result(timeout=0.1).success)
            self.assertEqual(off_writes(), count)

            forced = worker.submit("SAFE_OFF", force=True)
            self.assertTrue(forced.result(timeout=0.5).success)
            self.assertEqual(off_writes(), count + 4)

            self.assertTrue(worker.submit("FRONT_PUMP_ON").result(timeout=0.5).success)
            self.assertTrue(worker.submit("SAFE_OFF").result(timeout=0.5).success)
            self.assertEqual(off_writes(), count + 8)
        finally:
            worker.stop(0.5)

    def test_startup_failure_and_bounded_stop_resolve_futures_and_close_once(self):
        module = self.worker_module()
        instances = []

        class NoReadySerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.close_count = 0
                self.io_threads = set()
                instances.append(self)

            @property
            def in_waiting(self):
                self.io_threads.add(threading.get_ident())
                return super().in_waiting

            def read(self, size):
                self.io_threads.add(threading.get_ident())
                return super().read(size)

            def write(self, data):
                self.io_threads.add(threading.get_ident())
                return super().write(data)

            def close(self):
                self.io_threads.add(threading.get_ident())
                self.close_count += 1

        startup = module.SerialWorker(
            NoReadySerial, serial_kwargs={}, session=SESSION,
            hello_timeout=0.005, hello_attempts=2, poll_interval=0.001,
        )
        self.assertFalse(startup.wait_ready(0.1))
        self.assertIn("READY timed out", startup.startup_error)
        startup.stop(0.5, safe_off=False)
        self.assertEqual(instances[0].close_count, 1)
        self.assertEqual(len(instances[0].io_threads), 1)

        entered = threading.Event()

        class PendingSerial(NoReadySerial):
            def write(self, data):
                self.io_threads.add(threading.get_ident())
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    entered.set()
                return super().write(data)

        pending = module.SerialWorker(
            PendingSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.1, retry_limit=2, poll_interval=0.001,
        )
        self.assertTrue(pending.wait_ready(0.5))
        future = pending.submit("START")
        self.assertTrue(entered.wait(0.3))
        start = time.monotonic()
        pending.stop(0.25, safe_off=False)
        self.assertLess(time.monotonic() - start, 0.3)
        self.assertFalse(future.result(timeout=0.1).success)
        self.assertEqual(instances[-1].close_count, 1)
        self.assertEqual(len(instances[-1].io_threads), 1)

    def test_stop_suspends_ping_while_waiting_for_safe_off_ack_then_closes(self):
        module = self.worker_module()
        instances = []
        safe_off_started = threading.Event()
        delayed_off_ack_sent = threading.Event()
        active_on_written = threading.Event()

        class DelayedSafeOffSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.ping_count = 0
                self.commands = []
                self.close_count = 0
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    self.ping_count += 1
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                    ))
                elif frame[0] == "CMD":
                    self.commands.append(frame[3])
                    if frame[3] == "FRONT_PUMP_ON":
                        active_on_written.set()
                    elif frame[3] == "FRONT_CLEANER_OFF" and not safe_off_started.is_set():
                        safe_off_started.set()

                        def delayed_ack():
                            self.pending.extend(encode_frame(
                                "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                            ))
                            delayed_off_ack_sent.set()

                        threading.Timer(0.15, delayed_ack).start()
                    else:
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                        ))
                return super().write(data)

            def close(self):
                self.close_count += 1

        worker = module.SerialWorker(
            DelayedSafeOffSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.25, retry_limit=1, heartbeat_interval=0.015,
            heartbeat_timeout=0.5, poll_interval=0.001,
        )
        stop_result = {}
        repeated_stop_result = {}
        try:
            self.assertTrue(worker.wait_ready(0.5))
            active_on = worker.submit("FRONT_PUMP_ON", coalesce_key="front_pump")
            self.assertTrue(active_on_written.wait(0.3))
            deadline = time.monotonic() + 0.3
            while instances[0].ping_count == 0 and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertGreater(instances[0].ping_count, 0)

            stopper = threading.Thread(
                target=lambda: stop_result.setdefault("result", worker.stop(0.8))
            )
            stop_started_at = time.monotonic()
            stopper.start()
            self.assertTrue(safe_off_started.wait(0.3))
            ping_count_at_off = instances[0].ping_count
            repeated_stopper = threading.Thread(
                target=lambda: repeated_stop_result.setdefault(
                    "result", worker.stop(0.01)
                )
            )
            repeated_stopper.start()
            repeated_stopper.join(0.1)
            self.assertFalse(repeated_stopper.is_alive())
            self.assertFalse(
                repeated_stop_result["result"],
                "a short non-owner stop may time out, but must not finalize shutdown",
            )
            self.assertTrue(stopper.is_alive(), "the original stop owner still waits for OFF ACK")
            self.assertIsNotNone(worker._safety, "the owner’s SAFE_OFF transaction must remain intact")
            time.sleep(0.05)
            self.assertEqual(
                instances[0].ping_count, ping_count_at_off,
                "shutdown must stop extending the MCU watchdog before SAFE_OFF ACK wait",
            )
            self.assertFalse(delayed_off_ack_sent.is_set())
            stopper.join(1.0)
            repeated_stopper.join(1.0)
            self.assertFalse(stopper.is_alive())
            self.assertFalse(repeated_stopper.is_alive())
            self.assertTrue(stop_result["result"])
            self.assertFalse(repeated_stop_result["result"])
            self.assertTrue(delayed_off_ack_sent.is_set())
            self.assertLess(time.monotonic() - stop_started_at, 0.8)
            self.assertFalse(active_on.result(timeout=0.1).success)
            self.assertEqual(instances[0].commands.count("FRONT_PUMP_ON"), 1)
            self.assertEqual(instances[0].close_count, 1)
            self.assertTrue(worker.stop(0.1), "repeated stop should be idempotent")
            self.assertEqual(instances[0].close_count, 1)
        finally:
            if worker._thread.is_alive():
                worker.stop(0.5, safe_off=False)

    def test_stop_without_safe_off_is_bounded_and_repeated_call_closes_once(self):
        module = self.worker_module()
        instances = []
        command_written = threading.Event()

        class PendingCommandSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.close_count = 0
                self.ping_count = 0
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    self.ping_count += 1
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                    ))
                elif frame[0] == "CMD":
                    command_written.set()
                return super().write(data)

            def close(self):
                self.close_count += 1

        worker = module.SerialWorker(
            PendingCommandSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.1, retry_limit=2, heartbeat_interval=0.015,
            heartbeat_timeout=0.5, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            pending = worker.submit("START")
            self.assertTrue(command_written.wait(0.3))
            started = time.monotonic()
            self.assertTrue(worker.stop(0.25, safe_off=False))
            self.assertLess(time.monotonic() - started, 0.3)
            self.assertFalse(pending.result(timeout=0.1).success)
            self.assertTrue(worker.stop(0.1, safe_off=False))
            self.assertEqual(instances[0].close_count, 1)
        finally:
            if worker._thread.is_alive():
                worker.stop(0.5, safe_off=False)

    def test_fault_latch_cancels_active_and_queued_commands_before_safe_off(self):
        module = self.worker_module()
        instances = []
        active_on_written = threading.Event()
        release_active_on_write = threading.Event()
        active_on_write_completed = threading.Event()
        first_off_written = threading.Event()
        active_on_completed_at = []
        first_off_at = []
        fault_latch_returned_at = []
        ack_timeout = 0.1

        class FaultLatchSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.commands = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    self.pending.extend(encode_frame(
                        "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                    ))
                elif frame[0] == "CMD":
                    opcode = frame[3]
                    self.commands.append(opcode)
                    if opcode == "FRONT_PUMP_ON":
                        active_on_written.set()
                        release_active_on_write.wait(0.3)
                        # Leave the active command unacknowledged so fault
                        # latching must retire it without retrying.
                    else:
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], opcode, "ACCEPTED", "1"
                        ))
                result = super().write(data)
                if frame[0] == "CMD" and frame[3] == "FRONT_PUMP_ON":
                    active_on_completed_at.append(time.monotonic())
                    active_on_write_completed.set()
                elif frame[0] == "CMD" and frame[3] == "FRONT_CLEANER_OFF":
                    first_off_at.append(time.monotonic())
                    first_off_written.set()
                return result

        worker = module.SerialWorker(
            FaultLatchSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=ack_timeout, retry_limit=2, heartbeat_interval=0.015,
            poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            active_on = worker.submit("FRONT_PUMP_ON")
            self.assertTrue(active_on_written.wait(0.3))
            queued_start = worker.submit("START")
            queued_capture = worker.submit("CAPTURE_OK")
            queued_on = worker.submit("SIDE_PUMP_ON")

            latch_result = {}
            latch_done = threading.Event()
            def latch_worker_fault():
                latch_result["future"] = worker.latch_fault()
                fault_latch_returned_at.append(time.monotonic())
                latch_done.set()

            latch_thread = threading.Thread(target=latch_worker_fault)
            latch_thread.start()
            time.sleep(0.02)
            self.assertFalse(latch_done.is_set(), "fault latch must serialize with an in-progress CMD write")
            release_active_on_write.set()
            self.assertTrue(latch_done.wait(0.3))
            latch_thread.join(0.2)
            self.assertTrue(active_on_write_completed.wait(0.1))
            self.assertTrue(first_off_written.wait(0.1), "fault latch must dispatch SAFE_OFF")
            self.assertLess(
                first_off_at[0] - active_on_completed_at[0], ack_timeout,
                "SAFE_OFF must start before the active ON's first ACK deadline",
            )
            self.assertLess(
                first_off_at[0] - fault_latch_returned_at[0], ack_timeout,
                "SAFE_OFF must dispatch within one ACK window after fault latch returns",
            )
            safe_off = latch_result["future"]
            self.assertTrue(active_on.done())
            self.assertFalse(active_on.result().success)
            self.assertFalse(queued_start.result(timeout=0.1).success)
            self.assertFalse(queued_capture.result(timeout=0.1).success)
            self.assertFalse(queued_on.result(timeout=0.1).success)
            self.assertFalse(worker.submit("CAPTURE_OK").result(timeout=0.1).success)
            self.assertTrue(safe_off.result(timeout=0.5).success)
            retry_deadline = active_on_completed_at[0] + ack_timeout + 0.05
            time.sleep(max(0.0, retry_deadline - time.monotonic()))
            self.assertGreaterEqual(time.monotonic() - active_on_completed_at[0], ack_timeout)
            self.assertEqual(instances[0].commands.count("FRONT_PUMP_ON"), 1)
            self.assertEqual(instances[0].commands.count("START"), 0)
            self.assertNotIn("CAPTURE_OK", instances[0].commands)
            self.assertNotIn("SIDE_PUMP_ON", instances[0].commands)
            self.assertEqual(instances[0].commands[1:], [
                "FRONT_CLEANER_OFF", "FRONT_PUMP_OFF",
                "SIDE_CLEANER_OFF", "SIDE_PUMP_OFF",
            ])
        finally:
            release_active_on_write.set()
            worker.stop(0.5, safe_off=False)

    def test_cmd_ack_window_starts_after_slow_successful_write(self):
        module = self.worker_module()
        instances = []

        class SlowSuccessfulCommandSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.command_writes = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    self.command_writes.append(data)
                    time.sleep(0.08)

                    def delayed_ack():
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                        ))

                    threading.Timer(0.02, delayed_ack).start()
                return super().write(data)

        worker = module.SerialWorker(
            SlowSuccessfulCommandSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.05, retry_limit=1, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            result = worker.submit("START").result(timeout=0.3)
            self.assertTrue(result.success, result.error)
            self.assertEqual(len(instances[0].command_writes), 1)
        finally:
            worker.stop(0.5)

    def test_ping_ack_window_starts_after_slow_successful_write(self):
        module = self.worker_module()
        instances, first_ping_acked = [], threading.Event()

        class SlowSuccessfulPingSerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.ping_writes = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    self.ping_writes.append(data)
                    if len(self.ping_writes) == 1:
                        time.sleep(0.08)

                        def delayed_ack():
                            self.pending.extend(encode_frame(
                                "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                            ))
                            first_ping_acked.set()

                        threading.Timer(0.02, delayed_ack).start()
                    else:
                        self.pending.extend(encode_frame(
                            "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                        ))
                return super().write(data)

        worker = module.SerialWorker(
            SlowSuccessfulPingSerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.05, retry_limit=1, heartbeat_interval=0.2,
            heartbeat_timeout=0.5, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            self.assertTrue(first_ping_acked.wait(0.5))
            time.sleep(0.02)
            self.assertEqual(
                len(instances[0].ping_writes), 1,
                "a slow but successful PING write must still get a full ACK window",
            )
        finally:
            worker.stop(0.5)

    def test_cmd_retry_ack_window_starts_after_slow_successful_retry_write(self):
        module = self.worker_module()
        instances = []

        class SlowCommandRetrySerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.command_writes = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "CMD":
                    self.command_writes.append(data)
                    if len(self.command_writes) == 2:
                        time.sleep(0.08)

                        def delayed_ack():
                            self.pending.extend(encode_frame(
                                "ACK", frame[1], frame[2], frame[3], "ACCEPTED", "1"
                            ))

                        threading.Timer(0.02, delayed_ack).start()
                return super().write(data)

        worker = module.SerialWorker(
            SlowCommandRetrySerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.05, retry_limit=2, heartbeat_interval=0.5,
            heartbeat_timeout=1.0, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            result = worker.submit("START").result(timeout=0.5)
            self.assertTrue(result.success, result.error)
            self.assertEqual(len(instances[0].command_writes), 2)
            self.assertEqual(instances[0].command_writes[0], instances[0].command_writes[1])
        finally:
            worker.stop(0.5)

    def test_ping_retry_ack_window_starts_after_slow_successful_retry_write(self):
        module = self.worker_module()
        instances, retry_ack = [], threading.Event()

        class SlowPingRetrySerial(FakeSerial):
            def __init__(self, **_kwargs):
                super().__init__()
                self.ping_writes = []
                instances.append(self)

            def write(self, data):
                frame = decode_frame(data)
                if frame[0] == "HELLO":
                    self.pending.extend(encode_frame("READY", frame[1], "2"))
                elif frame[0] == "PING":
                    self.ping_writes.append(data)
                    if len(self.ping_writes) == 2:
                        time.sleep(0.08)

                        def delayed_ack():
                            self.pending.extend(encode_frame(
                                "ACK", frame[1], frame[2], "PING", "ACCEPTED", "1"
                            ))
                            retry_ack.set()

                        threading.Timer(0.02, delayed_ack).start()
                return super().write(data)

        worker = module.SerialWorker(
            SlowPingRetrySerial, serial_kwargs={}, session=SESSION,
            ack_timeout=0.05, retry_limit=2, heartbeat_interval=0.05,
            heartbeat_timeout=0.5, poll_interval=0.001,
        )
        try:
            self.assertTrue(worker.wait_ready(0.5))
            self.assertTrue(retry_ack.wait(0.5))
            time.sleep(0.02)
            self.assertEqual(len(instances[0].ping_writes), 2)
            self.assertEqual(instances[0].ping_writes[0], instances[0].ping_writes[1])
        finally:
            worker.stop(0.5)


if __name__ == "__main__":
    unittest.main()
