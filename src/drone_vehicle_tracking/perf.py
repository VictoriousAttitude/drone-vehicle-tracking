"""Performance benchmarking: per-stage wall-clock timing of the full pipeline.

Each swappable pipeline stage (decode, detect, track, project) is wrapped in a
light timing decorator and the real :func:`pipeline.run` is driven through them.
Because every stage is injected via the same Protocols the pipeline already
uses, benchmarking measures the exact production path and adds no branches to
the pipeline itself.

The split is the engineering payload: on a survey clip the detector dominates,
which is what justifies trading frames (``frame_stride``) for throughput.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.config import load_config
from drone_vehicle_tracking.interfaces import Detector, Projector, Tracker
from drone_vehicle_tracking.pipeline import (
    _build_detector,
    _build_projector,
    _build_tracker,
    iter_video_frames,
    run,
)
from drone_vehicle_tracking.telemetry.models import Detection, GeoPoint, TelemetryFrame, Track

# Module-level clock indirection so tests can substitute a deterministic clock.
_clock = time.perf_counter


@dataclass
class _Accumulator:
    """Mutable per-stage time totals shared by the timing wrappers (seconds)."""

    decode_s: float = 0.0
    detect_s: float = 0.0
    track_s: float = 0.0
    project_s: float = 0.0
    frames: int = 0


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Wall-clock cost of one pipeline run, split by stage (seconds)."""

    frames: int
    wall_s: float
    decode_s: float
    detect_s: float
    track_s: float
    project_s: float

    @property
    def fps(self) -> float:
        """Frames processed per second of total wall time."""
        return self.frames / self.wall_s if self.wall_s > 0 else 0.0

    def format_report(self) -> str:
        """Render a human-readable per-stage timing report."""
        return "\n".join(
            [
                f"frames processed : {self.frames}",
                f"total wall time  : {self.wall_s:.3f} s",
                f"throughput       : {self.fps:.2f} fps",
                f"  decode         : {self.decode_s:.3f} s",
                f"  detect         : {self.detect_s:.3f} s",
                f"  track          : {self.track_s:.3f} s",
                f"  project        : {self.project_s:.3f} s",
            ]
        )


class _TimedDetector:
    """Detector wrapper that accumulates inference time."""

    def __init__(self, inner: Detector, acc: _Accumulator) -> None:
        self._inner = inner
        self._acc = acc

    def detect(self, frame_index: int, image: npt.NDArray[np.uint8]) -> list[Detection]:
        start = _clock()
        result = self._inner.detect(frame_index, image)
        self._acc.detect_s += _clock() - start
        return result


class _TimedTracker:
    """Tracker wrapper that accumulates association time (update + finalize)."""

    def __init__(self, inner: Tracker, acc: _Accumulator) -> None:
        self._inner = inner
        self._acc = acc

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None:
        start = _clock()
        self._inner.update(frame_index, detections)
        self._acc.track_s += _clock() - start

    def finalize(self) -> list[Track]:
        start = _clock()
        result = self._inner.finalize()
        self._acc.track_s += _clock() - start
        return result


class _TimedProjector:
    """Projector wrapper that accumulates geo-referencing time."""

    def __init__(self, inner: Projector, acc: _Accumulator) -> None:
        self._inner = inner
        self._acc = acc

    def pixel_to_geo(self, pixel_xy: tuple[float, float], telemetry: TelemetryFrame) -> GeoPoint:
        start = _clock()
        result = self._inner.pixel_to_geo(pixel_xy, telemetry)
        self._acc.project_s += _clock() - start
        return result


def _timed_frames(
    frames: Iterable[tuple[int, npt.NDArray[np.uint8]]], acc: _Accumulator
) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
    """Wrap a frame iterable, timing the per-frame decode and counting frames."""
    iterator = iter(frames)
    while True:
        start = _clock()
        try:
            item = next(iterator)
        except StopIteration:
            return
        acc.decode_s += _clock() - start
        acc.frames += 1
        yield item


def benchmark(
    video_path: str | Path,
    srt_path: str | Path,
    config_path: str | Path,
    *,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    projector: Projector | None = None,
    frames: Iterable[tuple[int, npt.NDArray[np.uint8]]] | None = None,
) -> StageTimings:
    """Run the full pipeline once and return its per-stage wall-clock timing.

    The default components are built when not injected; tests inject fakes to
    exercise this without the CV stack. The pipeline still writes its normal
    outputs, so total wall time reflects the real end-to-end cost.
    """
    config = load_config(config_path)
    if projector is None:
        projector = _build_projector(config)
    if detector is None:
        detector = _build_detector(config)
    if tracker is None:
        tracker = _build_tracker(config)
    if frames is None:
        frames = iter_video_frames(video_path, config.frame_stride)

    acc = _Accumulator()
    start = _clock()
    run(
        video_path,
        srt_path,
        config_path,
        detector=_TimedDetector(detector, acc),
        tracker=_TimedTracker(tracker, acc),
        projector=_TimedProjector(projector, acc),
        frames=_timed_frames(frames, acc),
    )
    wall = _clock() - start
    return StageTimings(
        frames=acc.frames,
        wall_s=wall,
        decode_s=acc.decode_s,
        detect_s=acc.detect_s,
        track_s=acc.track_s,
        project_s=acc.project_s,
    )
