from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from drone_vehicle_tracking import perf
from drone_vehicle_tracking.interfaces import Detector, Projector, Tracker
from drone_vehicle_tracking.perf import StageTimings, benchmark
from drone_vehicle_tracking.telemetry.models import (
    Detection,
    GeoPoint,
    TelemetryFrame,
    Track,
    TrackPoint,
)


def test_fps_is_frames_over_wall_time() -> None:
    timings = StageTimings(
        frames=30, wall_s=2.0, decode_s=0.1, detect_s=1.5, track_s=0.2, project_s=0.05
    )
    assert timings.fps == 15.0


def test_fps_is_zero_when_no_wall_time() -> None:
    assert StageTimings(0, 0.0, 0.0, 0.0, 0.0, 0.0).fps == 0.0


def test_format_report_lists_every_stage() -> None:
    report = StageTimings(8, 4.0, 0.5, 3.0, 0.3, 0.2).format_report()
    for token in ("frames processed", "fps", "decode", "detect", "track", "project"):
        assert token in report


class _StepClock:
    """Deterministic clock advancing a fixed step per call (replaces perf_counter)."""

    def __init__(self, step: float = 0.001) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


class _FakeDetector:
    def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]:
        return [Detection(frame_index, (0.0, 0.0, 10.0, 10.0), 0.9, "car")]


class _FakeTracker:
    def __init__(self, frame_indices: tuple[int, ...]) -> None:
        self._frame_indices = frame_indices

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        pass

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


class _FakeProjector:
    def pixel_to_geo(self, pixel_xy: tuple[float, float], telemetry: TelemetryFrame) -> GeoPoint:
        return GeoPoint(latitude=telemetry.latitude, longitude=telemetry.longitude)


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(_FakeDetector(), Detector)
    assert isinstance(_FakeTracker((1,)), Tracker)
    assert isinstance(_FakeProjector(), Projector)


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


def test_benchmark_times_each_stage_with_injected_fakes(tmp_path, make_srt, monkeypatch) -> None:
    monkeypatch.setattr(perf, "_clock", _StepClock())
    srt = make_srt([(48.000, 25.000), (48.010, 25.000)])
    cfg = _write_config(tmp_path, tmp_path / "out")
    frames = [(1, np.zeros((4, 4, 3), dtype=np.uint8)), (2, np.zeros((4, 4, 3), dtype=np.uint8))]

    timings = benchmark(
        "ignored.mp4",
        srt,
        cfg,
        detector=_FakeDetector(),
        tracker=_FakeTracker((1, 2)),
        projector=_FakeProjector(),
        frames=frames,
    )

    assert timings.frames == 2
    assert timings.wall_s > 0
    assert timings.fps == pytest.approx(timings.frames / timings.wall_s)
    # Every stage ran, so each accrued some of the deterministic clock's ticks.
    assert timings.decode_s > 0
    assert timings.detect_s > 0
    assert timings.track_s > 0
    assert timings.project_s > 0  # two geo-located points were projected
