import random
from typing import Any, Dict, List

from sctq.synthetic.object_models import SyntheticObject
from sctq.synthetic.trajectory_generators import (
    generate_crossing_trajectories,
    generate_curved_trajectory,
    generate_linear_trajectory,
    generate_stop_and_go_trajectory,
)
from sctq.types import DetectionRaw, FrameDetections


class SyntheticSceneGenerator:
    """Generates synthetic scenes with configurable motion patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config.get("scene_width", 1920)
        self.height = config.get("scene_height", 1080)
        self.fps = config.get("fps", 30)
        self.duration_sec = config.get("duration_sec", 10)
        self.total_frames = self.fps * self.duration_sec
        self.objects: List[SyntheticObject] = []

    def generate(self) -> None:
        """Generate a new scene."""
        self.objects.clear()

        gen_cfg = self.config.get("generation", {})
        obj_range = gen_cfg.get("objects_per_scene", [5, 15])
        num_objects = random.randint(obj_range[0], obj_range[1])
        motion_types = gen_cfg.get("motion_types", ["linear", "curved", "crossing", "stop_and_go"])

        object_id_counter = 1

        # We might increment object_id_counter by 2 if we create a crossing pair
        i = 0
        while i < num_objects:
            motion = random.choice(motion_types)
            start_frame = random.randint(0, self.total_frames // 2)
            num_frames = random.randint(self.total_frames // 4, self.total_frames - start_frame)

            # Start near edges for realism, but keep it simple for now
            start_x = random.uniform(100, self.width - 100)
            start_y = random.uniform(100, self.height - 100)

            # Random velocity
            vx = random.uniform(-10, 10)
            vy = random.uniform(-10, 10)

            if motion == "crossing" and i < num_objects - 1:
                # Need another object to cross with
                start_x2 = start_x + random.uniform(-200, 200)
                start_y2 = start_y + random.uniform(-200, 200)
                vx2 = random.uniform(-10, 10)
                vy2 = random.uniform(-10, 10)

                traj1, traj2 = generate_crossing_trajectories(
                    (start_x, start_y),
                    (start_x2, start_y2),
                    (vx, vy),
                    (vx2, vy2),
                    start_frame,
                    num_frames,
                )
                self.objects.append(
                    SyntheticObject(object_id=object_id_counter, class_id=0, trajectory=traj1)
                )
                self.objects.append(
                    SyntheticObject(object_id=object_id_counter + 1, class_id=0, trajectory=traj2)
                )
                object_id_counter += 2
                i += 2
                continue

            elif motion == "curved":
                angular_vel = random.uniform(-0.1, 0.1)
                traj = generate_curved_trajectory(
                    (start_x, start_y), (vx, vy), angular_vel, start_frame, num_frames, class_id=0
                )
            elif motion == "stop_and_go":
                traj = generate_stop_and_go_trajectory(
                    (start_x, start_y), (vx, vy), start_frame, num_frames, class_id=0
                )
            else:
                # Fallback to linear
                traj = generate_linear_trajectory(
                    (start_x, start_y), (vx, vy), start_frame, num_frames, class_id=0
                )

            self.objects.append(
                SyntheticObject(object_id=object_id_counter, class_id=0, trajectory=traj)
            )
            object_id_counter += 1
            i += 1

    def get_detections(self, frame_idx: int) -> FrameDetections:
        """Get ideal ground-truth detections for a specific frame."""
        dets = []
        for obj in self.objects:
            if obj.start_frame <= frame_idx <= obj.end_frame:
                # Find point
                for pt in obj.trajectory:
                    if pt.frame_idx == frame_idx:
                        # Add bounds check so objects don't leave the scene entirely
                        if 0 <= pt.cx <= self.width and 0 <= pt.cy <= self.height:
                            dets.append(
                                DetectionRaw(
                                    frame_idx=frame_idx,
                                    det_id=obj.object_id,
                                    cx=pt.cx,
                                    cy=pt.cy,
                                    w=pt.w,
                                    h=pt.h,
                                    conf=1.0,
                                    class_id=pt.class_id,
                                )
                            )
                        break
        return FrameDetections(frame_idx=frame_idx, detections=dets)

    def get_all_detections(self) -> List[FrameDetections]:
        """Get detections for all frames."""
        all_dets = []
        for f in range(self.total_frames):
            all_dets.append(self.get_detections(f))
        return all_dets
