import csv
from pathlib import Path
from typing import Dict, List

from sctq.types import DetectionRaw, FrameDetections


class MOTReader:
    """Reads MOT-like detections from CSV-ish text files."""

    @staticmethod
    def read_detections(file_path: str) -> List[FrameDetections]:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Could not read MOT file {file_path}: file not found")

        detections_by_frame: Dict[int, List[DetectionRaw]] = {}
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "," if sample.count(",") >= sample.count(";") else ";"
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                if len(row) < 6:
                    # fallback for whitespace separated files
                    row = " ".join(row).replace(",", " ").split()
                if len(row) < 6:
                    continue
                try:
                    frame_idx = int(float(row[0]))
                    det_id = int(float(row[1])) if row[1] not in {"", "-1"} else -1
                    bb_left = float(row[2])
                    bb_top = float(row[3])
                    bb_width = float(row[4])
                    bb_height = float(row[5])
                    conf = float(row[6]) if len(row) > 6 and row[6] not in {"", "-1"} else 1.0
                    class_id = int(float(row[7])) if len(row) > 7 and row[7] not in {"", "-1"} else 0
                    visibility = float(row[8]) if len(row) > 8 and row[8] not in {"", "-1"} else 1.0
                except ValueError:
                    continue

                detections_by_frame.setdefault(frame_idx, []).append(
                    DetectionRaw(
                        frame_idx=frame_idx,
                        det_id=det_id,
                        cx=bb_left + bb_width / 2.0,
                        cy=bb_top + bb_height / 2.0,
                        w=bb_width,
                        h=bb_height,
                        conf=conf,
                        class_id=class_id,
                        visibility=visibility,
                    )
                )

        return [FrameDetections(frame_idx=frame_idx, detections=dets) for frame_idx, dets in sorted(detections_by_frame.items())]
