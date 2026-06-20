from drone_vehicle_tracking.geo.metrics import (
    net_displacement_m,
    path_length_m,
    track_geo_points,
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
