"""LED/초음파 전용 DBG1 세션. cap=8에는 주행·브러시·펌프 권한이 없다."""
from __future__ import annotations

import re
import time
import uuid

from .debug_protocol import parse_ready


COLORS = {'off': (0, 0, 0), 'red': (255, 0, 0), 'green': (0, 255, 0),
          'blue': (0, 0, 255), 'white': (255, 255, 255),
          'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255)}


class SensorDebugSession:
    def __init__(self, uart, *, clock=time.monotonic, session_id=None):
        self.uart, self.clock = uart, clock
        self.session_id = session_id or uuid.uuid4().hex
        if not re.fullmatch('[0-9a-f]{32}', self.session_id):
            raise ValueError('Invalid session ID')
        self.ready = self.failed = self.closed = False
        self.error = ''
        self.sequence = 0
        self.feedback = None
        self._buffer = bytearray()
        self._pending_led = None
        self._started = self._received = clock()
        self._sent = float('-inf')
        self._write(f'DBG1_HELLO {self.session_id} 8')

    def _write(self, text):
        frame = (text + '\r\n').encode('ascii')
        if self.uart.write(frame) != len(frame):
            raise IOError('Incomplete sensor UART write')

    def command_led(self, command):
        # ROS 콜백에서는 최신 요청 한 개만 보관한다. UART는 tick에서 최대 5Hz 송신하므로
        # CLI 연타가 STM의 한 줄 명령함을 넘치게 하거나 제어 루프를 막지 않는다.
        if not self.ready or self.closed or self.failed:
            raise ValueError('Sensor session is not ready')
        if command == 'auto':
            self._pending_led = ('auto', 0, 0, 0)
        elif command in ('blue', 'green'):
            self._pending_led = ('breathe', *COLORS[command])
        elif command == 'red':
            self._pending_led = ('blink', *COLORS[command])
        elif command in COLORS:
            self._pending_led = ('manual', *COLORS[command])
        elif re.fullmatch(r'[0-9]{1,3},[0-9]{1,3},[0-9]{1,3}', command):
            rgb = tuple(map(int, command.split(',')))
            if max(rgb) > 255:
                raise ValueError('RGB values must be 0..255')
            self._pending_led = ('manual', *rgb)
        else:
            raise ValueError('Use auto/off/red/green/blue/white/yellow/cyan/magenta or R,G,B')

    def _line(self, line):
        parts = line.split()
        if len(parts) < 2 or parts[1] != self.session_id:
            return
        kind = parts[0]
        if kind in ('DBG1_REJECT', 'DBG1_ERROR', 'DBG1_TIMEOUT'):
            raise ValueError('STM sensor debug: ' + ' '.join(parts[2:] or [kind]))
        if kind == 'DBG1_READY':
            if self.ready:
                raise ValueError('Unexpected duplicate sensor READY')
            parse_ready(parts, 8)
            self.ready = True
            return
        if kind != 'DBG1_SENSOR' or not self.ready:
            return
        if len(parts) != 13 or not all(p.isascii() and p.isdigit() for p in parts[2:12]):
            raise ValueError('Malformed sensor feedback')
        seq, tick, distance, valid, detected, held, auto, red, green, blue = map(int, parts[2:12])
        status = parts[12]
        if (not 0 <= seq <= self.sequence or not 0 <= tick <= 0xffffffff or distance > 10000
                or any(flag not in (0, 1) for flag in (valid, detected, held, auto))
                or max(red, green, blue) > 255 or status not in ('WAITING', 'ECHO', 'NO_ECHO')
                or bool(valid) != (status == 'ECHO') or (not valid and distance != 0)):
            raise ValueError('Invalid sensor feedback values')
        if self.feedback is not None:
            # 같은 샘플·지연 프레임으로 통신 감시 기한을 늘리지 않는다. HAL tick 순환은 허용한다.
            delta = (tick - self.feedback['mcu_millis']) & 0xffffffff
            if delta == 0 or delta >= 0x80000000 or seq < self.feedback['sequence']:
                return
        self.feedback = dict(sequence=seq, mcu_millis=tick, distance_mm=distance if valid else None,
                             valid=bool(valid), detected=bool(detected), held=bool(held),
                             led_auto=bool(auto), requested_rgb=[red, green, blue], range_status=status)
        self._received = self.clock()

    def tick(self):
        if self.closed or self.failed:
            return
        try:
            # 읽기·대기열 크기도 제한. 카메라/추론과 별도 프로세스이며 read는 nonblocking이다.
            self._buffer.extend(self.uart.read(min(4096, self.uart.in_waiting)))
            if len(self._buffer) > 8192:
                raise ValueError('Sensor UART receive overflow')
            while b'\n' in self._buffer:
                line, _, rest = self._buffer.partition(b'\n')
                self._buffer = bytearray(rest)
                self._line(line.decode('ascii'))
            now = self.clock()
            if not self.ready:
                if now - self._started > 2.0:
                    self.fail('No sensor handshake; compatible cap=8 STM firmware is required')
                return
            if now - self._received > 0.6:
                self.fail('Sensor feedback timeout')
                return
            if now - self._sent >= 0.2:
                self.sequence += 1
                if self.sequence > 0xffffffff:
                    raise ValueError('Sensor sequence exhausted; restart session')
                if self._pending_led is None:
                    self._write(f'DBG1_SET {self.session_id} {self.sequence} 0 0 0')
                else:
                    mode, r, g, b = self._pending_led
                    self._write(f'DBG1_LED {self.session_id} {self.sequence} {mode} {r} {g} {b}')
                    self._pending_led = None
                self._sent = now
        except (ValueError, OSError) as exc:
            self.fail(str(exc))

    def fail(self, reason):
        self.failed, self.error = True, reason
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self._pending_led = None
            self._write(f'DBG1_CLOSE {self.session_id}')


class Rr2SensorDebugSession:
    """LED/ultrasonic RR2 state with worker-owned serial and a 200 ms lease."""

    def __init__(self, worker, *, clock=time.monotonic):
        self.worker, self.clock = worker, clock
        self.session_id = worker.session_id
        self.failed = self.closed = False
        self.error = ''
        self.sequence = 0
        self.feedback = None
        self._pending_led = None
        self._pending = None
        self._started = self._received = clock()
        self._next_send = self._started

    @property
    def ready(self):
        return self.worker.ready and not self.failed and not self.closed

    def command_led(self, command):
        if not self.ready:
            raise ValueError('Sensor RR2 session is not ready')
        if command == 'auto':
            self._pending_led = ('auto', 0, 0, 0)
        elif command in ('blue', 'green'):
            self._pending_led = ('breathe', *COLORS[command])
        elif command == 'red':
            self._pending_led = ('blink', *COLORS[command])
        elif command in COLORS:
            self._pending_led = ('manual', *COLORS[command])
        elif re.fullmatch(r'[0-9]{1,3},[0-9]{1,3},[0-9]{1,3}', command):
            rgb = tuple(map(int, command.split(',')))
            if max(rgb) > 255:
                raise ValueError('RGB values must be 0..255')
            self._pending_led = ('manual', *rgb)
        else:
            raise ValueError('Use auto/off/red/green/blue/white/yellow/cyan/magenta or R,G,B')
        self._next_send = self.clock()

    def _status(self, result):
        fields = result.payload
        if not fields or fields[0] != 'ULTRASONIC':
            return
        if len(fields) != 11 or fields[10] not in ('WAITING', 'ECHO', 'NO_ECHO'):
            raise ValueError('Malformed RR2 ULTRASONIC status')
        values = fields[1:10]
        if any(not value.isascii() or not value.isdecimal() for value in values):
            raise ValueError('Malformed RR2 ULTRASONIC values')
        tick, distance, valid, detected, held, auto, red, green, blue = map(int, values)
        status = fields[10]
        if (tick > 0xffffffff or distance > 10000 or
                any(flag not in (0, 1) for flag in (valid, detected, held, auto)) or
                max(red, green, blue) > 255 or bool(valid) != (status == 'ECHO') or
                not valid and distance != 0):
            raise ValueError('Invalid RR2 ULTRASONIC status values')
        previous = self.feedback
        if previous is not None:
            delta = (tick - previous['mcu_millis']) & 0xffffffff
            if delta == 0 or delta >= 0x80000000:
                return
        self.feedback = dict(sequence=result.sequence, mcu_millis=tick,
                             distance_mm=distance if valid else None, valid=bool(valid),
                             detected=bool(detected), held=bool(held),
                             led_auto=bool(auto), requested_rgb=[red, green, blue],
                             range_status=status)
        self._received = self.clock()

    def tick(self):
        if self.failed or self.closed:
            return
        try:
            if not self.worker.ready:
                if self.feedback is None and not self.worker.startup_error and self.clock() - self._started < 2.0:
                    return
                raise RuntimeError(self.worker.startup_error or 'RR2 sensor session lost')
            for result in self.worker.drain_results(32):
                if result.kind == 'STATUS':
                    self._status(result)
                elif result.kind == 'EVENT' and result.payload and result.payload[0] in ('DEBUG_FAULT', 'DEBUG_TIMEOUT'):
                    raise RuntimeError('STM sensor event: ' + ' '.join(result.payload))
            now = self.clock()
            if self.feedback is None and now - self._started >= 2.0:
                raise RuntimeError('RR2 sensor feedback was not received')
            if self.feedback is not None and now - self._received >= .6:
                raise RuntimeError('RR2 sensor feedback timed out')
            if self._pending is not None:
                if not self._pending.done():
                    return
                outcome = self._pending.result()
                self._pending = None
                if not outcome.success:
                    raise RuntimeError('RR2 sensor command failed: ' + outcome.error)
                self.sequence = outcome.sequence or self.sequence
            if self.feedback is None or now < self._next_send:
                return
            if self._pending_led is None:
                # STOP refreshes Debug_RR2Stop's command lease without changing
                # LED mode or resetting breathe/blink animation phase.
                self._pending = self.worker.submit('STOP', force=True)
            else:
                mode, red, green, blue = self._pending_led
                self._pending = self.worker.submit(
                    'LED', (mode, str(red), str(green), str(blue)),
                    coalesce_key='debug_led')
                self._pending_led = None
            self._next_send = now + .2
            if self._pending.done() and not self._pending.result().success:
                raise RuntimeError('RR2 sensor queue rejected command: ' + self._pending.result().error)
        except (ValueError, RuntimeError, OSError) as exc:
            self.fail(str(exc))

    def fail(self, reason):
        if self.failed:
            return
        self.failed, self.error = True, reason
        self._pending_led = None
        self.worker.latch_fault()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._pending_led = None
        self.worker.submit('STOP', force=True)
