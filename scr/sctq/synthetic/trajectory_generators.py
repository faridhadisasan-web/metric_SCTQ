import math
from typing import List, Tuple

import numpy as np

from sctq.types import TrackPoint


def generate_linear_trajectory(
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    start_frame: int,
    num_frames: int,
    w: float = 50.0,
    h: float = 100.0,
    class_id: int = 0,
) -> List[TrackPoint]:
    """Generate a simple linear trajectory."""
    points = []
    x, y = start_pos
    vx, vy = velocity
    for i in range(num_frames):
        points.append(
            TrackPoint(frame_idx=start_frame + i, cx=x, cy=y, w=w, h=h, conf=1.0, class_id=class_id)
        )
        x += vx
        y += vy
    return points


def generate_curved_trajectory(
    start_pos: Tuple[float, float],
    initial_velocity: Tuple[float, float],
    angular_velocity: float,
    start_frame: int,
    num_frames: int,
    w: float = 50.0,
    h: float = 100.0,
    class_id: int = 0,
) -> List[TrackPoint]:
    """Generate a curved trajectory with constant angular velocity."""
    points = []
    x, y = start_pos
    speed = math.hypot(*initial_velocity)
    heading = math.atan2(initial_velocity[1], initial_velocity[0])

    for i in range(num_frames):
        points.append(
            TrackPoint(frame_idx=start_frame + i, cx=x, cy=y, w=w, h=h, conf=1.0, class_id=class_id)
        )
        heading += angular_velocity
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)

    return points


def generate_crossing_trajectories(
    start_pos1: Tuple[float, float],
    start_pos2: Tuple[float, float],
    velocity1: Tuple[float, float],
    velocity2: Tuple[float, float],
    start_frame: int,
    num_frames: int,
    w: float = 50.0,
    h: float = 100.0,
) -> Tuple[List[TrackPoint], List[TrackPoint]]:
    """Generate two intersecting linear trajectories to test tracker fragmentation and ID consistency."""
    traj1 = generate_linear_trajectory(start_pos1, velocity1, start_frame, num_frames, w, h, 0)
    traj2 = generate_linear_trajectory(start_pos2, velocity2, start_frame, num_frames, w, h, 0)
    return traj1, traj2


def generate_stop_and_go_trajectory(
    start_pos: Tuple[float, float],
    velocity: Tuple[float, float],
    start_frame: int,
    num_frames: int,
    stop_start_ratio: float = 0.4,
    stop_end_ratio: float = 0.6,
    w: float = 50.0,
    h: float = 100.0,
    class_id: int = 0,
) -> List[TrackPoint]:
    """Generate a trajectory that moves, stops for a period, and then moves again."""
    points = []
    x, y = start_pos
    vx, vy = velocity

    stop_start = int(num_frames * stop_start_ratio)
    stop_end = int(num_frames * stop_end_ratio)

    for i in range(num_frames):
        points.append(
            TrackPoint(frame_idx=start_frame + i, cx=x, cy=y, w=w, h=h, conf=1.0, class_id=class_id)
        )

        # Only update position if not in the stop window
        if not (stop_start <= i <= stop_end):
            x += vx
            y += vy

    return points
