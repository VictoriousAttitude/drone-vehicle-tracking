from datetime import datetime, timedelta

import pytest

from drone_vehicle_tracking.telemetry.models import GeoPoint, Track, TrackPoint
from drone_vehicle_tracking.visualization.map_viz import render_map


def _track(
    track_id: int,
    coords: list[tuple[float, float]],
    times: list[datetime] | None = None,
    confidences: list[float] | None = None,
) -> Track:
    points = [
        TrackPoint(
            frame_index=i,
            pixel_xy=(0.0, 0.0),
            geo=GeoPoint(lat, lon),
            timestamp=None if times is None else times[i],
            confidence=None if confidences is None else confidences[i],
        )
        for i, (lat, lon) in enumerate(coords)
    ]
    return Track(track_id=track_id, class_name="car", points=points)


def test_render_map_writes_html(tmp_path) -> None:
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    moving = _track(
        1,
        [(48.0, 25.0), (48.001, 25.0), (48.002, 25.0)],
        times=[t0, t0 + timedelta(seconds=5), t0 + timedelta(seconds=10)],
        confidences=[0.8, 0.7, 0.9],
    )
    stationary = _track(2, [(48.01, 25.01), (48.01, 25.01)])  # no timestamps/confidence
    out = tmp_path / "map.html"
    render_map([moving, stationary], out)
    html = out.read_text()
    assert out.exists()
    assert "leaflet" in html.lower()
    # Both the moving palette colour and the stationary grey should be present.
    assert "#e6194b" in html
    assert "#808080" in html
    # The timed moving track shows a speed; the untimed stationary one does not.
    assert "km/h" in html
    # The moving track carries confidence; the stationary one does not.
    assert "mean confidence" in html


def test_render_map_shows_position_accuracy_when_given(tmp_path) -> None:
    moving = _track(1, [(48.0, 25.0), (48.001, 25.0), (48.002, 25.0)])
    out = tmp_path / "map.html"
    render_map([moving], out, position_error_m=2.5)
    assert "position accuracy" in out.read_text()


def test_render_map_skips_tracks_without_geo(tmp_path) -> None:
    no_geo = Track(
        track_id=1,
        class_name="car",
        points=[TrackPoint(frame_index=0, pixel_xy=(0.0, 0.0), geo=None)],
    )
    with pytest.raises(ValueError):
        render_map([no_geo], tmp_path / "map.html")


def test_render_map_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        render_map([], tmp_path / "map.html")
