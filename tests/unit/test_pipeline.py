import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from drone_vehicle_tracking import pipeline
from drone_vehicle_tracking.interfaces import Detector, Projector, Tracker
from drone_vehicle_tracking.pipeline import georeference_tracks, run, tracks_to_geojson
from drone_vehicle_tracking.telemetry.models import (
    Detection,
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


class _FakeDetector:
    """Returns one detection per frame; satisfies the Detector Protocol."""

    def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]:
        return [Detection(frame_index, (0.0, 0.0, 10.0, 10.0), 0.9, "car")]


class _FakeTracker:
    """Yields a fixed two-point track regardless of input."""

    def __init__(self, frame_indices: tuple[int, int]) -> None:
        self._frame_indices = frame_indices
        self.updates = 0

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        self.updates += 1

    def finalize(self) -> list[Track]:
        return [
            Track(
                track_id=1,
                class_name="car",
                points=[
                    TrackPoint(frame_index=fi, pixel_xy=(5.0, 9.0)) for fi in self._frame_indices
                ],
            )
        ]


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(_FakeDetector(), Detector)
    assert isinstance(_FakeTracker((1, 2)), Tracker)


def _write_config(tmp_path: Path, output_dir: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "detection:\n"
        "  model: unused.pt\n"
        "  conf_threshold: 0.25\n"
        "  imgsz: 1280\n"
        "  classes: [car]\n"
        "tracking:\n"
        "  min_track_length: 1\n"
        "camera:\n"
        "  model: mavic_3t_wide\n"
        "projection:\n"
        "  altitude_source: rel_alt\n"
        "io:\n"
        "  frame_stride: 1\n"
        f"  output_dir: {output_dir}\n"
        "visualization:\n"
        "  map_html: map.html\n"
        "  moving_min_displacement_m: 3.0\n"
    )
    return cfg


def test_run_end_to_end_with_injected_components(tmp_path, make_srt) -> None:
    srt = make_srt([(48.000, 25.000), (48.010, 25.000)])  # ~1.1 km north -> moving
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir)
    frames = [(1, np.zeros((4, 4, 3), dtype=np.uint8)), (2, np.zeros((4, 4, 3), dtype=np.uint8))]
    tracker = _FakeTracker((1, 2))

    tracks = run(
        "ignored.mp4",
        srt,
        cfg,
        detector=_FakeDetector(),
        tracker=tracker,
        projector=FakeProjector(),
        frames=frames,
    )

    assert tracker.updates == 2
    assert len(tracks) == 1
    geojson = json.loads((out_dir / "tracks.geojson").read_text())
    assert len(geojson["features"]) == 1
    assert (out_dir / "map.html").exists()


def test_run_skips_map_when_no_geo(tmp_path, make_srt) -> None:
    srt = make_srt([(48.0, 25.0), (48.0, 25.0)])
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir)
    frames = [(900, np.zeros((4, 4, 3), dtype=np.uint8))]  # no telemetry for frame 900
    tracks = run(
        "ignored.mp4",
        srt,
        cfg,
        detector=_FakeDetector(),
        tracker=_FakeTracker((900, 900)),
        projector=FakeProjector(),
        frames=frames,
    )

    assert all(p.geo is None for t in tracks for p in t.points)
    assert (out_dir / "tracks.geojson").exists()
    assert not (out_dir / "map.html").exists()


def test_cli_main_invokes_run(monkeypatch, capsys) -> None:
    import drone_vehicle_tracking.cli as cli

    captured: dict[str, object] = {}

    def fake_run(video: str, srt: str, config: str) -> list[Track]:
        captured["args"] = (video, srt, config)
        return [Track(1, "car", []), Track(2, "car", [])]

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv", ["dvt", "--video", "v.mp4", "--srt", "v.srt", "--config", "c.yaml"]
    )
    cli.main()

    assert captured["args"] == ("v.mp4", "v.srt", "c.yaml")
    assert "2 geo-referenced vehicle tracks" in capsys.readouterr().out
