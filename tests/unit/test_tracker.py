import pytest

from drone_vehicle_tracking.interfaces import Tracker
from drone_vehicle_tracking.telemetry.models import Detection
from drone_vehicle_tracking.tracking.tracker import (
    ByteTrackVehicleTracker,
    bbox_bottom_center,
)


def _det(frame: int, x: float, y: float, cls: str = "car") -> Detection:
    return Detection(frame, (x, y, x + 24.0, y + 24.0), 0.9, cls)


def test_bbox_bottom_center() -> None:
    assert bbox_bottom_center((10.0, 20.0, 30.0, 50.0)) == (20.0, 50.0)


def test_single_moving_object_yields_one_stable_track() -> None:
    pytest.importorskip("supervision")
    tracker = ByteTrackVehicleTracker(min_track_length=5)
    for f in range(25):
        tracker.update(f, [_det(f, 100.0 + f * 4.0, 200.0)])
    tracks = tracker.finalize()

    assert len(tracks) == 1
    track = tracks[0]
    assert track.class_name == "car"
    assert len(track.points) >= 5
    xs = [p.pixel_xy[0] for p in track.points]
    assert xs == sorted(xs)  # motion is monotonic to the right
    assert all(p.geo is None for p in track.points)  # geo attached later


def test_short_track_is_filtered_out() -> None:
    pytest.importorskip("supervision")
    tracker = ByteTrackVehicleTracker(min_track_length=10)
    for f in range(3):
        tracker.update(f, [_det(f, 100.0 + f * 4.0, 200.0)])
    assert tracker.finalize() == []


def test_empty_frames_do_not_crash() -> None:
    pytest.importorskip("supervision")
    tracker = ByteTrackVehicleTracker(min_track_length=1)
    tracker.update(0, [])
    tracker.update(1, [_det(1, 100.0, 200.0)])
    tracker.finalize()  # must not raise


def test_tracker_satisfies_protocol() -> None:
    pytest.importorskip("supervision")
    assert isinstance(ByteTrackVehicleTracker(min_track_length=1), Tracker)
