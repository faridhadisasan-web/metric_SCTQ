import math
from typing import Tuple


def compute_displacement(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Compute dx, dy displacement between two points."""
    return p2[0] - p1[0], p2[1] - p1[1]


def compute_speed(dx: float, dy: float) -> float:
    """Compute Euclidean speed from dx and dy."""
    return math.hypot(dx, dy)


def compute_heading(dx: float, dy: float) -> float:
    """Compute heading angle using atan2. Result is in radians."""
    return math.atan2(dy, dx)


def angular_difference(theta1: float, theta2: float) -> float:
    """Compute shortest angular difference between two angles in radians."""
    diff = (theta2 - theta1) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return abs(diff)


def unwrap_angles(angles: list[float]) -> list[float]:
    """Unwrap a list of angles to remove 2*pi jumps."""
    if not angles:
        return []

    unwrapped = [angles[0]]
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        unwrapped.append(unwrapped[-1] + diff)
    return unwrapped
