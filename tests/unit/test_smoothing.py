import pytest

from drone_vehicle_tracking.geo.metrics import path_length_m, track_geo_points
from drone_vehicle_tracking.geo.smoothing import smooth_track, smooth_tracks
from drone_vehicle_tracking.telemetry.models import GeoPoint, Track, TrackPoint


def _geo_track(lons: list[float], lat: float = 0.0, track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        class_name="car",
        points=[TrackPoint(i, (0.0, 0.0), GeoPoint(lat, lon)) for i, lon in enumerate(lons)],
    )


def test_window_one_is_identity() -> None:
    track = _geo_track([0.0, 1.0, 2.0])
    assert smooth_track(track, 1) is track  # disabled -> same object, no copy


def test_centred_average_smooths_interior_and_anchors_endpoints() -> None:
    track = _geo_track([0.0, 0.0, 6.0, 0.0, 0.0])
    out = smooth_track(track, 3)
    lons = [p.geo.longitude for p in out.points]
    assert lons[0] == 0.0  # first fix anchored (half-window 0)
    assert lons[-1] == 0.0  # last fix anchored
    assert lons[1] == pytest.approx(2.0)  # spike spread across the 3-pt window
    assert lons[2] == pytest.approx(2.0)
    assert lons[3] == pytest.approx(2.0)
    assert all(p.geo.latitude == 0.0 for p in out.points)  # constant axis untouched


def test_last_endpoint_is_anchored_even_when_its_neighbour_differs() -> None:
    # Right-end anchoring: the final fix must survive unchanged even when its
    # neighbour differs sharply, guarding the symmetric window shrink at the end.
    track = _geo_track([0.0, 0.0, 0.0, 6.0])
    out = smooth_track(track, 3)
    lons = [p.geo.longitude for p in out.points]
    assert lons[-1] == pytest.approx(6.0)  # anchored, not pulled toward its neighbour
    assert lons[0] == pytest.approx(0.0)


def test_smoothing_shortens_a_jittery_path() -> None:
    jittery = _geo_track([0.0, 0.0010, -0.0010, 0.0010, -0.0010, 0.0])
    smoothed = smooth_track(jittery, 3)
    assert path_length_m(track_geo_points(smoothed)) < path_length_m(track_geo_points(jittery))


def test_points_without_geo_are_preserved_and_skipped() -> None:
    track = Track(
        track_id=2,
        class_name="car",
        points=[
            TrackPoint(0, (0.0, 0.0), GeoPoint(0.0, 0.0)),
            TrackPoint(1, (0.0, 0.0), geo=None),  # gap: no telemetry this frame
            TrackPoint(2, (0.0, 0.0), GeoPoint(0.0, 6.0)),
            TrackPoint(3, (0.0, 0.0), GeoPoint(0.0, 0.0)),
        ],
    )
    out = smooth_track(track, 3)
    assert out.points[1].geo is None  # gap left untouched
    # The three geo points form the smoothing sequence; the middle one is averaged.
    assert out.points[2].geo.longitude == pytest.approx(2.0)


def test_smooth_tracks_maps_over_all_tracks() -> None:
    a = _geo_track([0.0, 6.0, 0.0], track_id=1)
    b = _geo_track([0.0, 9.0, 0.0], track_id=2)
    out_a, out_b = smooth_tracks([a, b], 3)
    assert out_a.points[1].geo.longitude == pytest.approx(2.0)
    assert out_b.points[1].geo.longitude == pytest.approx(3.0)
