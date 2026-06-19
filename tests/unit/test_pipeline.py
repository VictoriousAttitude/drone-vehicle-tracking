from datetime import datetime

from drone_vehicle_tracking.interfaces import Projector
from drone_vehicle_tracking.pipeline import georeference_tracks, tracks_to_geojson
from drone_vehicle_tracking.telemetry.models import (
    GeoPoint,
    TelemetryFrame,
    Track,
    TrackPoint,
)


class FakeProjector:
    """Encodes the telemetry it was given into the output so tests can assert
    that each point was projected with its own frame's pose."""

    def pixel_to_geo(self, pixel_xy: tuple[float, float], telemetry: TelemetryFrame) -> GeoPoint:
        return GeoPoint(
            latitude=telemetry.latitude + pixel_xy[1],
            longitude=telemetry.longitude + pixel_xy[0],
        )


def _telemetry(frame: int, lat: float, lon: float) -> TelemetryFrame:
    return TelemetryFrame(
        frame_index=frame,
        timestamp=datetime(2024, 1, 1),
        latitude=lat,
        longitude=lon,
        rel_alt=80.0,
        abs_alt=180.0,
        gimbal_yaw=0.0,
        gimbal_pitch=-90.0,
        gimbal_roll=0.0,
        focal_len=24.0,
    )


def test_fake_projector_satisfies_protocol() -> None:
    assert isinstance(FakeProjector(), Projector)


def test_georeference_uses_each_points_own_frame() -> None:
    telemetry = {
        1: _telemetry(1, lat=48.0, lon=25.0),
        2: _telemetry(2, lat=49.0, lon=26.0),
    }
    track = Track(
        track_id=7,
        class_name="car",
        points=[
            TrackPoint(frame_index=1, pixel_xy=(10.0, 20.0)),
            TrackPoint(frame_index=2, pixel_xy=(0.0, 0.0)),
        ],
    )
    (out,) = georeference_tracks([track], telemetry, FakeProjector())
    assert out.points[0].geo == GeoPoint(latitude=48.0 + 20.0, longitude=25.0 + 10.0)
    assert out.points[1].geo == GeoPoint(latitude=49.0, longitude=26.0)


def test_georeference_leaves_geo_none_when_telemetry_missing() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[TrackPoint(frame_index=99, pixel_xy=(1.0, 1.0))],
    )
    (out,) = georeference_tracks([track], {}, FakeProjector())
    assert out.points[0].geo is None


def test_tracks_to_geojson_structure_and_axis_order() -> None:
    track = Track(
        track_id=3,
        class_name="truck",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(latitude=48.1, longitude=25.2)),
            TrackPoint(2, (0.0, 0.0), GeoPoint(latitude=48.3, longitude=25.4)),
        ],
    )
    fc = tracks_to_geojson([track])
    assert fc["type"] == "FeatureCollection"
    feature = fc["features"][0]  # type: ignore[index]
    assert feature["geometry"]["type"] == "LineString"
    assert feature["geometry"]["coordinates"] == [[25.2, 48.1], [25.4, 48.3]]
    assert feature["properties"]["track_id"] == 3
    assert feature["properties"]["class_name"] == "truck"


def test_tracks_to_geojson_omits_tracks_with_fewer_than_two_geo_points() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(latitude=48.0, longitude=25.0)),
            TrackPoint(2, (0.0, 0.0), geo=None),
        ],
    )
    fc = tracks_to_geojson([track])
    assert fc["features"] == []
