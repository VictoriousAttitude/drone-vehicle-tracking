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
    pytest.importorskip("trackers")
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
    pytest.importorskip("trackers")
    tracker = ByteTrackVehicleTracker(min_track_length=10)
    for f in range(3):
        tracker.update(f, [_det(f, 100.0 + f * 4.0, 200.0)])
    assert tracker.finalize() == []


def test_empty_frames_do_not_crash() -> None:
    pytest.importorskip("trackers")
    tracker = ByteTrackVehicleTracker(min_track_length=1)
    tracker.update(0, [])
    tracker.update(1, [_det(1, 100.0, 200.0)])
    tracker.finalize()  # must not raise


def test_tracker_satisfies_protocol() -> None:
    pytest.importorskip("trackers")
    assert isinstance(ByteTrackVehicleTracker(min_track_length=1), Tracker)


def test_update_ignores_result_without_tracker_ids() -> None:
    sv = pytest.importorskip("supervision")
    pytest.importorskip("trackers")
    tracker = ByteTrackVehicleTracker(min_track_length=1)
    # Simulate the underlying tracker returning detections with no assigned ids.
    tracker._tracker.update = lambda det: sv.Detections.empty()  # type: ignore[method-assign]
    tracker.update(0, [_det(0, 100.0, 200.0)])
    assert tracker.finalize() == []


def test_update_handles_tracks_without_class_id() -> None:
    import numpy as np

    sv = pytest.importorskip("supervision")
    pytest.importorskip("trackers")
    tracker = ByteTrackVehicleTracker(min_track_length=1)
    # tracker_id present (one confirmed track) but class_id absent.
    tracked = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 50.0]]),
        tracker_id=np.array([7]),
    )
    tracker._tracker.update = lambda det: tracked  # type: ignore[method-assign]
    tracker.update(0, [_det(0, 100.0, 200.0)])
    (track,) = tracker.finalize()
    assert track.track_id == 7
    assert track.class_name == "unknown"  # no class_id -> default
    assert track.points[0].pixel_xy == (20.0, 50.0)
    assert track.points[0].bbox_xyxy == (10.0, 20.0, 30.0, 50.0)  # genuine box kept
