from datetime import datetime, timedelta

import pytest

from drone_vehicle_tracking.geo.metrics import (
    geo_centroid,
    net_displacement_m,
    path_length_m,
    reprojection_scatter_m,
    track_geo_points,
    track_speed,
)
from drone_vehicle_tracking.telemetry.models import GeoPoint, Track, TrackPoint


def _track_with_geo(coords: list[tuple[float, float]]) -> Track:
    points = [
        TrackPoint(frame_index=i, pixel_xy=(0.0, 0.0), geo=GeoPoint(lat, lon))
        for i, (lat, lon) in enumerate(coords)
    ]
    return Track(track_id=1, class_name="car", points=points)


def test_track_geo_points_filters_none() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(frame_index=0, pixel_xy=(0.0, 0.0), geo=GeoPoint(48.0, 25.0)),
            TrackPoint(frame_index=1, pixel_xy=(1.0, 1.0), geo=None),
            TrackPoint(frame_index=2, pixel_xy=(2.0, 2.0), geo=GeoPoint(48.1, 25.1)),
        ],
    )
    geo = track_geo_points(track)
    assert len(geo) == 2
    assert geo[0] == GeoPoint(48.0, 25.0)
    assert geo[1] == GeoPoint(48.1, 25.1)


def test_net_displacement_one_degree_latitude() -> None:
    # 0.001 deg latitude ~ 111 m anywhere on the ellipsoid.
    points = [GeoPoint(48.0, 25.0), GeoPoint(48.001, 25.0)]
    assert net_displacement_m(points) == net_displacement_m(points)
    assert abs(net_displacement_m(points) - 111.0) < 1.0


def test_net_displacement_requires_two_points() -> None:
    assert net_displacement_m([]) == 0.0
    assert net_displacement_m([GeoPoint(48.0, 25.0)]) == 0.0


def test_net_displacement_ignores_intermediate_points() -> None:
    # Out-and-back: large path, ~zero net displacement.
    points = [GeoPoint(48.0, 25.0), GeoPoint(48.001, 25.0), GeoPoint(48.0, 25.0)]
    assert net_displacement_m(points) < 0.01


def test_path_length_sums_segments() -> None:
    points = [GeoPoint(48.0, 25.0), GeoPoint(48.001, 25.0), GeoPoint(48.002, 25.0)]
    # Two ~111 m segments.
    assert abs(path_length_m(points) - 222.0) < 2.0


def test_path_length_at_least_net_displacement() -> None:
    track = _track_with_geo([(48.0, 25.0), (48.001, 25.001), (48.0, 25.002)])
    geo = track_geo_points(track)
    assert path_length_m(geo) >= net_displacement_m(geo)


def test_geo_centroid_is_arithmetic_mean() -> None:
    c = geo_centroid([GeoPoint(48.0, 25.0), GeoPoint(48.2, 25.4)])
    assert c.latitude == pytest.approx(48.1)
    assert c.longitude == pytest.approx(25.2)


def test_reprojection_scatter_empty_is_zero() -> None:
    stats = reprojection_scatter_m([])
    assert stats.count == 0
    assert stats.rms_m == 0.0
    assert stats.max_m == 0.0


def test_reprojection_scatter_identical_points_is_zero() -> None:
    pts = [GeoPoint(48.0, 25.0)] * 5
    stats = reprojection_scatter_m(pts)
    assert stats.count == 5
    assert stats.rms_m == pytest.approx(0.0, abs=1e-6)
    assert stats.max_m == pytest.approx(0.0, abs=1e-6)


def test_track_speed_mean_distance_over_time() -> None:
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(0, (0.0, 0.0), GeoPoint(48.0, 25.0), t0),
            TrackPoint(1, (0.0, 0.0), geo=None),  # no geo -> skipped
            TrackPoint(2, (0.0, 0.0), GeoPoint(48.001, 25.0), t0 + timedelta(seconds=10)),
        ],
    )
    speed = track_speed(track)
    assert speed is not None
    assert speed.duration_s == 10.0
    assert speed.distance_m == pytest.approx(111.0, abs=1.0)  # ~111 m
    assert speed.mean_speed_mps == pytest.approx(11.1, abs=0.1)
    assert speed.mean_speed_kmh == pytest.approx(speed.mean_speed_mps * 3.6)


def test_track_speed_spans_first_to_last_timed_point() -> None:
    # Three timed points: the duration must span first->last, not first->second,
    # so a mid-track timestamp cannot shorten the measured timespan.
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(0, (0.0, 0.0), GeoPoint(48.0, 25.0), t0),
            TrackPoint(1, (0.0, 0.0), GeoPoint(48.001, 25.0), t0 + timedelta(seconds=1)),
            TrackPoint(2, (0.0, 0.0), GeoPoint(48.002, 25.0), t0 + timedelta(seconds=20)),
        ],
    )
    speed = track_speed(track)
    assert speed is not None
    assert speed.duration_s == pytest.approx(20.0)  # first->last, not the 1 s to point 2


def test_track_speed_accepts_subsecond_duration() -> None:
    # A positive sub-second timespan must still yield a speed (guards the > 0 boundary).
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(0, (0.0, 0.0), GeoPoint(48.0, 25.0), t0),
            TrackPoint(1, (0.0, 0.0), GeoPoint(48.0001, 25.0), t0 + timedelta(milliseconds=500)),
        ],
    )
    speed = track_speed(track)
    assert speed is not None
    assert speed.duration_s == pytest.approx(0.5)


def test_track_speed_none_without_timestamps() -> None:
    track = _track_with_geo([(48.0, 25.0), (48.001, 25.0)])  # geo but no timestamps
    assert track_speed(track) is None


def test_track_speed_none_when_duration_not_positive() -> None:
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(0, (0.0, 0.0), GeoPoint(48.0, 25.0), t0),
            TrackPoint(1, (0.0, 0.0), GeoPoint(48.001, 25.0), t0),  # same timestamp
        ],
    )
    assert track_speed(track) is None


def test_reprojection_scatter_known_spread() -> None:
    # Two points symmetric about the centroid, ~111 m apart in latitude.
    # Each lies ~55.5 m from the centroid -> RMS == max == ~55.5 m.
    pts = [GeoPoint(48.0, 25.0), GeoPoint(48.001, 25.0)]
    stats = reprojection_scatter_m(pts)
    assert stats.count == 2
    assert stats.rms_m <= stats.max_m
    assert stats.rms_m == pytest.approx(stats.max_m, rel=1e-4)  # symmetric to ~um
    assert abs(stats.max_m - 55.5) < 1.0
