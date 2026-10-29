"""ROS clock conversion and common QoS; no hardware or global rclpy context."""
from __future__ import annotations

import math

from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

qos_sensor = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
qos_events = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=32,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
qos_state = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def stamp_seconds(stamp) -> float:
    if not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError("Invalid ROS nanosecond timestamp.")
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000


def stamp_from_seconds(value: float) -> Time:
    if not math.isfinite(value) or value < 0:
        raise ValueError("ROS timestamp must be finite and nonnegative.")
    nanoseconds = int(value * 1_000_000_000)
    seconds, nanos = divmod(nanoseconds, 1_000_000_000)
    return Time(sec=seconds, nanosec=nanos)


def source_monotonic(stamp, ros_now_seconds: float, monotonic_now: float,
                     max_age: float) -> float:
    """Preserve source age without comparing different machines' monotonic clocks."""
    age = ros_now_seconds - stamp_seconds(stamp)
    if not all(math.isfinite(v) for v in (age, monotonic_now, max_age)):
        raise ValueError("Nonfinite observation time.")
    if max_age < 0 or age < 0 or age > max_age:
        raise ValueError("Observation is stale or from the future.")
    return monotonic_now - age
