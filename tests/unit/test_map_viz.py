import pytest

from drone_vehicle_tracking.telemetry.models import GeoPoint, Track, TrackPoint
from drone_vehicle_tracking.visualization.map_viz import render_map


def _track(track_id: int, coords: list[tuple[float, float]]) -> Track:
    points = [
        TrackPoint(frame_index=i, pixel_xy=(0.0, 0.0), geo=GeoPoint(lat, lon))
        for i, (lat, lon) in enumerate(coords)
    ]
    return Track(track_id=track_id, class_name="car", points=points)


def test_render_map_writes_html(tmp_path) -> None:
    moving = _track(1, [(48.0, 25.0), (48.001, 25.0), (48.002, 25.0)])
    stationary = _track(2, [(48.01, 25.01), (48.01, 25.01)])
    out = tmp_path / "map.html"
    render_map([moving, stationary], out)
    html = out.read_text()
    assert out.exists()
    assert "leaflet" in html.lower()
    # Both the moving palette colour and the stationary grey should be present.
    assert "#e6194b" in html
    assert "#808080" in html


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
