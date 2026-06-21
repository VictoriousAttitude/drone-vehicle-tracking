"""Optional annotated-video output (boxes + track IDs) for visual QA.

A frame-indexed lookup of box annotations is built as a pure function
(``overlay_index``), so the geometry/labelling logic is unit-testable without
OpenCV; the thin :func:`render_overlay` then opens the source video and burns
each frame's boxes on with cv2. Boxes drawn are the genuine detection boxes the
tracker carried through (``TrackPoint.bbox_xyxy``), not synthesised ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.telemetry.models import Track

# Distinct BGR colours (OpenCV channel order) cycled per track id.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (75, 25, 230),
    (75, 180, 60),
    (216, 99, 67),
    (49, 130, 245),
    (180, 30, 145),
    (244, 212, 66),
    (230, 50, 240),
    (69, 235, 191),
)


@dataclass(frozen=True, slots=True)
class BoxAnnotation:
    """One box to draw on one frame: pixel bbox, label parts and BGR colour."""

    track_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    color: tuple[int, int, int]


def _color_for(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def overlay_index(tracks: Sequence[Track]) -> dict[int, list[BoxAnnotation]]:
    """Map each frame index to the box annotations to draw on it.

    Track points without a ``bbox_xyxy`` (e.g. injected geo-only points) carry no
    box and are skipped.
    """
    index: dict[int, list[BoxAnnotation]] = {}
    for track in tracks:
        color = _color_for(track.track_id)
        for point in track.points:
            if point.bbox_xyxy is None:
                continue
            index.setdefault(point.frame_index, []).append(
                BoxAnnotation(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    bbox_xyxy=point.bbox_xyxy,
                    color=color,
                )
            )
    return index


def _draw_box(frame: npt.NDArray[np.uint8], ann: BoxAnnotation) -> None:
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in ann.bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), ann.color, 2)
    label = f"#{ann.track_id} {ann.class_name}"
    cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ann.color, 1)


def render_overlay(
    video_path: str | Path, tracks: Sequence[Track], output_path: str | Path
) -> None:
    """Burn track boxes and IDs onto the source video for sanity checking."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    index = overlay_index(tracks)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    try:
        position = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for ann in index.get(position + 1, []):
                _draw_box(frame, ann)
            writer.write(frame)
            position += 1
    finally:
        capture.release()
        writer.release()
