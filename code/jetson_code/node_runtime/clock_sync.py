"""Kernel clock synchronization state for START guards (read-only adjtimex).

RTC 배터리가 없는 Jetson은 부팅 직후 시계가 과거에 머물다가 NTP가 붙는 순간 크게 점프한다.
임무 도중 점프하면 ROS 신선도 검사가 fault를 내므로, 동기화 전에는 START를 보내지 않는다.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys

# Shared by mission_node (response message) and remote_control_node (result code).
CLOCK_UNSYNCED_MESSAGE = "Jetson clock is not NTP-synchronized; START was not sent."

# adjtimex() returns TIME_ERROR while the kernel's STA_UNSYNC flag is set,
# which is what `timedatectl` reports as "System clock synchronized: no".
TIME_ERROR = 5
# struct timex is under 256 bytes on every Linux ABI; modes=0 reads only.
_TIMEX_BUFFER_BYTES = 512


def clock_synchronized() -> bool | None:
    """True/False from the kernel on Linux; None where the state cannot be read."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        state = libc.adjtimex(ctypes.create_string_buffer(_TIMEX_BUFFER_BYTES))
    except (OSError, AttributeError):
        return None
    if state < 0:
        return None
    return state != TIME_ERROR
