import pytest

from drone_vehicle_tracking.telemetry.models import Track, TrackPoint
from drone_vehicle_tracking.tracking.quality import (
    filter_tracks_by_confidence,
    track_mean_confidence,
)


def _track(track_id: int, confidences: list[float | None]) -> Track:
    return Track(
        track_id=track_id,
        class_name="car",
        points=[TrackPoint(i, (0.0, 0.0), confidence=c) for i, c in enumerate(confidences)],
    )


def test_mean_confidence_averages_present_values() -> None:
    assert track_mean_confidence(_track(1, [0.8, 0.6])) == pytest.approx(0.7)


def test_mean_confidence_ignores_missing_values() -> None:
    assert track_mean_confidence(_track(1, [0.9, None])) == pytest.approx(0.9)


def test_mean_confidence_is_none_without_any_values() -> None:
    assert track_mean_confidence(_track(1, [None, None])) is None


def test_filter_disabled_keeps_everything() -> None:
    tracks = [_track(1, [0.1]), _track(2, [0.9])]
    assert filter_tracks_by_confidence(tracks, 0.0) == tracks


def test_filter_drops_low_mean_keeps_high() -> None:
    low = _track(1, [0.2, 0.2])
    high = _track(2, [0.8, 0.8])
    kept = filter_tracks_by_confidence([low, high], 0.5)
    assert [t.track_id for t in kept] == [2]


def test_filter_keeps_tracks_without_confidence() -> None:
    unknown = _track(1, [None, None])  # cannot be assessed -> kept
    kept = filter_tracks_by_confidence([unknown], 0.5)
    assert [t.track_id for t in kept] == [1]
