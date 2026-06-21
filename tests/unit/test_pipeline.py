import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from drone_vehicle_tracking import pipeline
from drone_vehicle_tracking.geo.camera import MAVIC_3T_WIDE
from drone_vehicle_tracking.geo.error_budget import AccuracyModel
from drone_vehicle_tracking.interfaces import Detector, Projector, Tracker
from drone_vehicle_tracking.pipeline import georeference_tracks, run, tracks_to_geojson
from drone_vehicle_tracking.telemetry.models import (
    Detection,
    GeoPoint,
    TelemetryFrame,
    Track,
    TrackPoint,
)
from drone_vehicle_tracking.tracking.tracker import bbox_bottom_center

# RFC 7946-shaped schema for the GeoJSON this pipeline emits: a FeatureCollection
# of LineString features, each coordinate a [lon, lat] pair. Locks the output
# contract so a structural regression fails a test instead of a consumer.
_GEOJSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["type", "features"],
    "properties": {
        "type": {"const": "FeatureCollection"},
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "geometry", "properties"],
                "properties": {
                    "type": {"const": "Feature"},
                    "geometry": {
                        "type": "object",
                        "required": ["type", "coordinates"],
                        "properties": {
                            "type": {"const": "LineString"},
                            "coordinates": {
                                "type": "array",
                                "minItems": 2,
                                "items": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "items": {"type": "number"},
                                },
                            },
                        },
                    },
                    "properties": {"type": "object"},
                },
            },
        },
    },
}


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
    # The frame's telemetry timestamp is attached alongside the geo coordinate.
    assert out.points[0].timestamp == datetime(2024, 1, 1)
    # No accuracy model supplied -> no per-point error is computed.
    assert out.points[0].position_error_m is None


def _accuracy_model() -> AccuracyModel:
    return AccuracyModel(
        camera=MAVIC_3T_WIDE,
        altitude_source="rel_alt",
        tilt_error_deg=0.1,
        altitude_relative_error=0.01,
        focal_relative_error=0.01,
        yaw_error_deg=0.5,
        gnss_error_m=3.0,
    )


def test_georeference_attaches_geometry_aware_error_when_accuracy_given() -> None:
    telemetry = {1: _telemetry(1, lat=48.0, lon=25.0)}
    _fx, _fy, cx, cy = MAVIC_3T_WIDE.intrinsics()
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(frame_index=1, pixel_xy=(cx, cy)),  # nadir -> floor-dominated
            TrackPoint(frame_index=1, pixel_xy=(0.0, 0.0)),  # corner -> larger
        ],
    )
    (out,) = georeference_tracks([track], telemetry, FakeProjector(), _accuracy_model())
    center_error = out.points[0].position_error_m
    corner_error = out.points[1].position_error_m
    assert center_error is not None and corner_error is not None
    assert corner_error > center_error >= 3.0  # at least the GNSS floor, grows off-nadir


def test_georeference_leaves_geo_none_when_telemetry_missing() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[TrackPoint(frame_index=99, pixel_xy=(1.0, 1.0))],
    )
    (out,) = georeference_tracks([track], {}, FakeProjector())
    assert out.points[0].geo is None
    assert out.points[0].timestamp is None


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
    # No timestamps on these points, so speed is not computable.
    assert feature["properties"]["mean_speed_kmh"] is None
    # No confidence on these points either.
    assert feature["properties"]["mean_confidence"] is None
    # Self-reported accuracy is unset when not supplied.
    assert feature["properties"]["position_error_m"] is None


def test_tracks_to_geojson_records_position_error_when_given() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(latitude=48.0, longitude=25.0)),
            TrackPoint(2, (0.0, 0.0), GeoPoint(latitude=48.1, longitude=25.0)),
        ],
    )
    fc = tracks_to_geojson([track], position_error_m=2.5)
    assert fc["features"][0]["properties"]["position_error_m"] == 2.5  # type: ignore[index]


def test_tracks_to_geojson_prefers_per_point_error_over_fallback() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(48.0, 25.0), position_error_m=4.2),
            TrackPoint(2, (0.0, 0.0), GeoPoint(48.1, 25.0), position_error_m=3.1),
        ],
    )
    fc = tracks_to_geojson([track], position_error_m=3.0)
    # The track's worst-case (max) per-point error wins over the 3.0 fallback.
    assert fc["features"][0]["properties"]["position_error_m"] == 4.2  # type: ignore[index]


def test_tracks_to_geojson_includes_mean_speed_when_timed() -> None:
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    track = Track(
        track_id=5,
        class_name="car",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(latitude=48.0, longitude=25.0), t0),
            TrackPoint(
                2, (0.0, 0.0), GeoPoint(latitude=48.001, longitude=25.0), t0.replace(second=10)
            ),
        ],
    )
    fc = tracks_to_geojson([track])
    speed = fc["features"][0]["properties"]["mean_speed_kmh"]  # type: ignore[index]
    assert isinstance(speed, float)
    assert speed == pytest.approx(40.0, abs=1.0)  # ~111 m in 10 s -> ~40 km/h


def test_tracks_to_geojson_includes_mean_confidence_when_present() -> None:
    track = Track(
        track_id=8,
        class_name="car",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(48.0, 25.0), confidence=0.8),
            TrackPoint(2, (0.0, 0.0), GeoPoint(48.1, 25.0), confidence=0.6),
        ],
    )
    fc = tracks_to_geojson([track])
    assert fc["features"][0]["properties"]["mean_confidence"] == pytest.approx(0.7)  # type: ignore[index]


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


def _write_config(
    tmp_path: Path,
    output_dir: Path,
    smoothing_window: int | None = None,
    min_track_confidence: float | None = None,
) -> Path:
    cfg = tmp_path / "config.yaml"
    processing = ""
    if smoothing_window is not None or min_track_confidence is not None:
        processing = "processing:\n"
        if smoothing_window is not None:
            processing += f"  smoothing_window: {smoothing_window}\n"
        if min_track_confidence is not None:
            processing += f"  min_track_confidence: {min_track_confidence}\n"
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
        f"{processing}"
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
    # Self-reported accuracy is geometry-aware (computed per point), and is always
    # at least the 3.0 m GNSS floor (default config, no accuracy/export sections).
    assert geojson["features"][0]["properties"]["position_error_m"] >= 3.0
    assert (out_dir / "map.html").exists()


def test_run_writes_cot_when_path_given(tmp_path, make_srt) -> None:
    from xml.etree.ElementTree import fromstring

    srt = make_srt([(48.0, 25.0), (48.010, 25.0)])  # ~1.1 km north -> moving
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir)
    cot = tmp_path / "tracks.cot"
    frames = [(i, np.zeros((4, 4, 3), dtype=np.uint8)) for i in (1, 2)]

    run(
        "ignored.mp4",
        srt,
        cfg,
        detector=_FakeDetector(),
        tracker=_FakeTracker((1, 2)),
        projector=FakeProjector(),
        frames=frames,
        cot_path=cot,
    )

    root = fromstring(cot.read_text())  # parses -> well-formed XML
    (event,) = root.findall("event")
    assert event.get("version") == "2.0"
    assert event.get("uid") == "dvt-vehicle-1"
    # ce is the track's geometry-aware accuracy, at least the 3.0 m GNSS floor.
    assert float(event.find("point").get("ce")) >= 3.0


class _CenterBoxDetector:
    """Emits one box per frame whose bottom-centre is the image centre (960, 540)
    for the 1920x1080 mavic_3t_wide model."""

    def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]:
        return [Detection(frame_index, (955.0, 530.0, 965.0, 540.0), 0.9, "car")]


class _PassThroughTracker:
    """Turns each frame's first detection into a track point at the detection's
    bbox bottom-centre, so the real detector->projector chain is exercised."""

    def __init__(self) -> None:
        self._points: list[TrackPoint] = []

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        det = detections[0]
        self._points.append(
            TrackPoint(frame_index=frame_index, pixel_xy=bbox_bottom_center(det.bbox_xyxy))
        )

    def finalize(self) -> list[Track]:
        return [Track(track_id=1, class_name="car", points=self._points)]


def test_run_geo_references_center_detection_to_drone_position(tmp_path, make_srt) -> None:
    """End-to-end with the REAL projector and a known answer: a detection at the
    image centre under a nadir gimbal must geo-reference to the drone's own
    position, so the output coordinate is verifiable, not just well-formed."""
    lat0, lon0 = 48.267013, 25.914562
    lat1, lon1 = 48.268013, 25.914562  # ~111 m north -> a moving track
    srt = make_srt([(lat0, lon0), (lat1, lon1)])
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir)
    frames = [(i, np.zeros((4, 4, 3), dtype=np.uint8)) for i in (1, 2)]

    # projector is NOT injected -> run() builds the real NadirProjector from config.
    run(
        "ignored.mp4",
        srt,
        cfg,
        detector=_CenterBoxDetector(),
        tracker=_PassThroughTracker(),
        frames=frames,
    )

    geojson = json.loads((out_dir / "tracks.geojson").read_text())
    coords = geojson["features"][0]["geometry"]["coordinates"]
    assert coords[0][0] == pytest.approx(lon0, abs=1e-6)
    assert coords[0][1] == pytest.approx(lat0, abs=1e-6)
    assert coords[1][0] == pytest.approx(lon1, abs=1e-6)
    assert coords[1][1] == pytest.approx(lat1, abs=1e-6)
    # The on-disk artefact is valid GeoJSON, not only well-shaped in memory.
    jsonschema.validate(instance=geojson, schema=_GEOJSON_SCHEMA)


def test_tracks_to_geojson_validates_against_geojson_schema() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[
            TrackPoint(1, (0.0, 0.0), GeoPoint(latitude=48.1, longitude=25.2)),
            TrackPoint(2, (0.0, 0.0), GeoPoint(latitude=48.3, longitude=25.4)),
        ],
    )
    jsonschema.validate(instance=tracks_to_geojson([track]), schema=_GEOJSON_SCHEMA)


def test_empty_feature_collection_is_valid_geojson() -> None:
    jsonschema.validate(instance=tracks_to_geojson([]), schema=_GEOJSON_SCHEMA)


class _SpikeTracker:
    """Yields a 3-point track whose middle pixel is offset, so the FakeProjector
    produces a one-point latitude spike for smoothing to flatten."""

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        pass

    def finalize(self) -> list[Track]:
        return [
            Track(
                track_id=1,
                class_name="car",
                points=[
                    TrackPoint(frame_index=1, pixel_xy=(0.0, 0.0)),
                    TrackPoint(frame_index=2, pixel_xy=(0.0, 6.0)),  # the spike
                    TrackPoint(frame_index=3, pixel_xy=(0.0, 0.0)),
                ],
            )
        ]


def test_run_applies_geo_smoothing(tmp_path, make_srt) -> None:
    srt = make_srt([(0.0, 25.0), (0.0, 25.0), (0.0, 25.0)])  # flat telemetry
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir, smoothing_window=3)
    frames = [(i, np.zeros((4, 4, 3), dtype=np.uint8)) for i in (1, 2, 3)]

    run(
        "ignored.mp4",
        srt,
        cfg,
        detector=_FakeDetector(),
        tracker=_SpikeTracker(),
        projector=FakeProjector(),
        frames=frames,
    )

    geojson = json.loads((out_dir / "tracks.geojson").read_text())
    coords = geojson["features"][0]["geometry"]["coordinates"]
    lats = [lat for _lon, lat in coords]
    assert lats[0] == 0.0 and lats[2] == 0.0  # endpoints anchored
    assert lats[1] == pytest.approx(2.0)  # spike (6) averaged with two zero neighbours


class _TwoTrackTracker:
    """Yields a confident and a weak track so the quality filter can drop one."""

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        pass

    def finalize(self) -> list[Track]:
        return [
            Track(
                track_id=1,
                class_name="car",
                points=[TrackPoint(fi, (5.0, 9.0), confidence=0.9) for fi in (1, 2)],
            ),
            Track(
                track_id=2,
                class_name="car",
                points=[TrackPoint(fi, (5.0, 9.0), confidence=0.1) for fi in (1, 2)],
            ),
        ]


def test_run_drops_low_confidence_tracks(tmp_path, make_srt) -> None:
    srt = make_srt([(48.0, 25.0), (48.010, 25.0)])
    out_dir = tmp_path / "out"
    cfg = _write_config(tmp_path, out_dir, min_track_confidence=0.5)
    frames = [(i, np.zeros((4, 4, 3), dtype=np.uint8)) for i in (1, 2)]

    tracks = run(
        "ignored.mp4",
        srt,
        cfg,
        detector=_FakeDetector(),
        tracker=_TwoTrackTracker(),
        projector=FakeProjector(),
        frames=frames,
    )

    assert [t.track_id for t in tracks] == [1]  # the 0.1-mean track was filtered out
    geojson = json.loads((out_dir / "tracks.geojson").read_text())
    assert [f["properties"]["track_id"] for f in geojson["features"]] == [1]


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

    def fake_run(video: str, srt: str, config: str, *, cot_path: str | None = None) -> list[Track]:
        captured["args"] = (video, srt, config)
        captured["cot_path"] = cot_path
        return [Track(1, "car", []), Track(2, "car", [])]

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv", ["dvt", "--video", "v.mp4", "--srt", "v.srt", "--config", "c.yaml"]
    )
    cli.main()

    assert captured["args"] == ("v.mp4", "v.srt", "c.yaml")
    assert captured["cot_path"] is None
    assert "2 geo-referenced vehicle tracks" in capsys.readouterr().out


def test_cli_main_benchmark_prints_report(monkeypatch, capsys) -> None:
    import drone_vehicle_tracking.cli as cli
    import drone_vehicle_tracking.perf as perf
    from drone_vehicle_tracking.perf import StageTimings

    def fake_benchmark(video: str, srt: str, config: str) -> StageTimings:
        return StageTimings(
            frames=6, wall_s=2.0, decode_s=0.1, detect_s=1.5, track_s=0.3, project_s=0.05
        )

    monkeypatch.setattr(perf, "benchmark", fake_benchmark)
    monkeypatch.setattr("sys.argv", ["dvt", "--video", "v.mp4", "--srt", "v.srt", "--benchmark"])
    cli.main()

    out = capsys.readouterr().out
    assert "throughput" in out
    assert "3.00 fps" in out


def test_cli_main_overlay_renders_annotated_video(monkeypatch, capsys) -> None:
    import drone_vehicle_tracking.cli as cli
    import drone_vehicle_tracking.visualization.video_overlay as video_overlay

    tracks = [Track(1, "car", [])]
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "run", lambda video, srt, config, cot_path=None: tracks)

    def fake_render(video: str, trks: list[Track], output: str) -> None:
        captured["args"] = (video, trks, output)

    monkeypatch.setattr(video_overlay, "render_overlay", fake_render)
    monkeypatch.setattr(
        "sys.argv",
        ["dvt", "--video", "v.mp4", "--srt", "v.srt", "--overlay", "out.mp4"],
    )
    cli.main()

    assert captured["args"] == ("v.mp4", tracks, "out.mp4")
    assert "Wrote annotated video: out.mp4" in capsys.readouterr().out


def test_cli_main_cot_passes_path_to_run(monkeypatch, capsys) -> None:
    import drone_vehicle_tracking.cli as cli

    captured: dict[str, object] = {}

    def fake_run(video: str, srt: str, config: str, *, cot_path: str | None = None) -> list[Track]:
        captured["cot_path"] = cot_path
        return [Track(1, "car", [])]

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv", ["dvt", "--video", "v.mp4", "--srt", "v.srt", "--cot", "out.cot"]
    )
    cli.main()

    assert captured["cot_path"] == "out.cot"
    assert "Wrote CoT events: out.cot" in capsys.readouterr().out
