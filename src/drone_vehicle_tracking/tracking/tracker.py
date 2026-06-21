"""Multi-object tracking via the roboflow ``trackers`` ByteTrack implementation.

ByteTrack consumes detections from any detector, keeping detection and tracking
decoupled. The tracker emits pixel-space tracks (ground-contact point = bbox
bottom-centre); geo-referencing is applied downstream. Detections are exchanged
through supervision's ``Detections`` container, which ``trackers`` consumes and
returns annotated with ``tracker_id``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drone_vehicle_tracking.telemetry.models import Detection, Track, TrackPoint


def bbox_bottom_center(bbox: Sequence[float]) -> tuple[float, float]:
    """Ground-contact pixel of a vehicle: horizontal centre, bottom edge."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


class ByteTrackVehicleTracker:
    """Associates per-frame detections into stable tracks using ByteTrack."""

    def __init__(self, min_track_length: int) -> None:
        import supervision as sv
        from trackers import ByteTrackTracker

        self._sv = sv
        self._tracker = ByteTrackTracker()
        self._min_track_length = min_track_length
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._points: dict[int, list[TrackPoint]] = {}
        self._class_of: dict[int, str] = {}

    def _class_id(self, name: str) -> int:
        if name not in self._name_to_id:
            cid = len(self._name_to_id)
            self._name_to_id[name] = cid
            self._id_to_name[cid] = name
        return self._name_to_id[name]

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        sv = self._sv
        if detections:
            det = sv.Detections(
                xyxy=np.array([d.bbox_xyxy for d in detections], dtype=float),
                confidence=np.array([d.confidence for d in detections], dtype=float),
                class_id=np.array([self._class_id(d.class_name) for d in detections], dtype=int),
            )
        else:
            det = sv.Detections.empty()
        tracked = self._tracker.update(det)
        if tracked.tracker_id is None:
            return
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            if tid < 0:  # unconfirmed detection, not yet promoted to a stable track
                continue
            box = tracked.xyxy[i]
            point = TrackPoint(
                frame_index=frame_index,
                pixel_xy=bbox_bottom_center(box),
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
            self._points.setdefault(tid, []).append(point)
            if tracked.class_id is not None:
                self._class_of.setdefault(tid, self._id_to_name[int(tracked.class_id[i])])

    def finalize(self) -> list[Track]:
        tracks = [
            Track(track_id=tid, class_name=self._class_of.get(tid, "unknown"), points=points)
            for tid, points in self._points.items()
            if len(points) >= self._min_track_length
        ]
        return sorted(tracks, key=lambda t: t.track_id)
