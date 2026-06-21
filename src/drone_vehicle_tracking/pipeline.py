"""End-to-end orchestration: telemetry + video -> geo-referenced tracks -> output.

This module wires the stages together. The heavy CV imports (OpenCV, the YOLO
detector, the tracker) are deferred into :func:`run`, so the pure orchestration
helpers (``georeference_tracks``, ``tracks_to_geojson``) stay importable and
unit-testable without the ``[cv]`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.config import PipelineConfig, load_config
from drone_vehicle_tracking.geo.camera import CAMERA_REGISTRY
from drone_vehicle_tracking.geo.error_budget import AccuracyModel
from drone_vehicle_tracking.geo.metrics import track_position_error_m, track_speed
from drone_vehicle_tracking.geo.projection import NadirProjector
from drone_vehicle_tracking.geo.smoothing import smooth_tracks
from drone_vehicle_tracking.interfaces import Detector, Projector, Tracker
from drone_vehicle_tracking.telemetry.models import TelemetryFrame, Track
from drone_vehicle_tracking.telemetry.srt_parser import parse_srt
from drone_vehicle_tracking.tracking.quality import (
    filter_tracks_by_confidence,
    track_mean_confidence,
)


def _build_projector(config: PipelineConfig) -> Projector:
    """Construct the default nadir projector from config."""
    return NadirProjector(CAMERA_REGISTRY[config.camera_model], config.altitude_source)


def _build_detector(config: PipelineConfig) -> Detector:
    """Construct the default YOLO detector (lazily importing the CV stack)."""
    from drone_vehicle_tracking.detection.detector import YoloVehicleDetector

    return YoloVehicleDetector(config.model, config.conf_threshold, config.classes, config.imgsz)


def _build_tracker(config: PipelineConfig) -> Tracker:
    """Construct the default ByteTrack tracker (lazily importing the CV stack)."""
    from drone_vehicle_tracking.tracking.tracker import ByteTrackVehicleTracker

    return ByteTrackVehicleTracker(config.min_track_length)


def _build_accuracy_model(config: PipelineConfig) -> AccuracyModel:
    """Construct the per-point accuracy model from config (GNSS floor = position_error_m)."""
    return AccuracyModel(
        camera=CAMERA_REGISTRY[config.camera_model],
        altitude_source=config.altitude_source,
        tilt_error_deg=config.tilt_error_deg,
        altitude_relative_error=config.altitude_relative_error,
        focal_relative_error=config.focal_relative_error,
        yaw_error_deg=config.yaw_error_deg,
        gnss_error_m=config.position_error_m,
    )


def georeference_tracks(
    tracks: list[Track],
    telemetry_by_index: Mapping[int, TelemetryFrame],
    projector: Projector,
    accuracy: AccuracyModel | None = None,
) -> list[Track]:
    """Attach a WGS84 ``geo`` coordinate to every track point with telemetry.

    Each point is projected using the telemetry of its own frame, so per-frame
    drone motion and gimbal yaw are respected. Points whose frame has no matching
    telemetry are left with ``geo=None`` rather than dropped. When ``accuracy`` is
    given, each projected point also gets a geometry-aware ``position_error_m``.
    """
    out: list[Track] = []
    for track in tracks:
        points = []
        for point in track.points:
            telemetry = telemetry_by_index.get(point.frame_index)
            if telemetry is None:
                points.append(point)
                continue
            geo = projector.pixel_to_geo(point.pixel_xy, telemetry)
            error = accuracy.error_for(point.pixel_xy, telemetry) if accuracy is not None else None
            points.append(
                replace(point, geo=geo, timestamp=telemetry.timestamp, position_error_m=error)
            )
        out.append(Track(track_id=track.track_id, class_name=track.class_name, points=points))
    return out


def tracks_to_geojson(
    tracks: list[Track], position_error_m: float | None = None
) -> dict[str, object]:
    """Serialise geo-referenced tracks to a GeoJSON ``FeatureCollection``.

    Each track becomes a ``LineString`` of ``[lon, lat]`` vertices (GeoJSON axis
    order). Tracks with fewer than two geo-located points are omitted. Each
    feature's ``position_error_m`` is the track's worst-case geometry-aware
    accuracy when its points carry one, else the ``position_error_m`` fallback;
    ``None`` leaves it unset.
    """
    features: list[dict[str, object]] = []
    for track in tracks:
        coords = [
            [point.geo.longitude, point.geo.latitude]
            for point in track.points
            if point.geo is not None
        ]
        if len(coords) < 2:
            continue
        speed = track_speed(track)
        confidence = track_mean_confidence(track)
        track_error = track_position_error_m(track)
        error = track_error if track_error is not None else position_error_m
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "num_points": len(coords),
                    "mean_speed_kmh": (
                        round(speed.mean_speed_kmh, 1) if speed is not None else None
                    ),
                    "mean_confidence": (round(confidence, 3) if confidence is not None else None),
                    "position_error_m": round(error, 1) if error is not None else None,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def iter_video_frames(
    video_path: str | Path, frame_stride: int = 1
) -> Iterator[tuple[int, npt.NDArray[np.uint8]]]:
    """Yield ``(frame_index, image)`` pairs from a video file.

    ``frame_index`` is 1-based to align with the DJI SRT ``FrameCnt``. Only every
    ``frame_stride``-th frame is yielded.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        position = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if position % frame_stride == 0:
                yield position + 1, frame
            position += 1
    finally:
        capture.release()


def run(
    video_path: str | Path,
    srt_path: str | Path,
    config_path: str | Path,
    *,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    projector: Projector | None = None,
    frames: Iterable[tuple[int, npt.NDArray[np.uint8]]] | None = None,
    cot_path: str | Path | None = None,
) -> list[Track]:
    """Run detect -> track -> geo-reference for one flight and write outputs.

    Returns the geo-referenced tracks and writes ``tracks.geojson`` (and, when
    any point is geo-located, ``map.html``) into the configured output directory.
    When ``cot_path`` is given, also writes a Cursor-on-Target XML file for TAK.

    The ``detector``, ``tracker``, ``projector`` and ``frames`` arguments default
    to the production implementations (lazily importing the heavy CV stack) but
    can be injected, which keeps the orchestration testable without that stack.
    """
    config = load_config(config_path)
    telemetry_by_index = {frame.frame_index: frame for frame in parse_srt(srt_path)}

    if projector is None:
        projector = _build_projector(config)
    if detector is None:
        detector = _build_detector(config)
    if tracker is None:
        tracker = _build_tracker(config)
    if frames is None:
        frames = iter_video_frames(video_path, config.frame_stride)

    for frame_index, image in frames:
        tracker.update(frame_index, detector.detect(frame_index, image))

    kept = filter_tracks_by_confidence(tracker.finalize(), config.min_track_confidence)
    tracks = georeference_tracks(kept, telemetry_by_index, projector, _build_accuracy_model(config))
    tracks = smooth_tracks(tracks, config.smoothing_window)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geojson = tracks_to_geojson(tracks, config.position_error_m)
    (output_dir / "tracks.geojson").write_text(json.dumps(geojson, indent=2))

    if any(point.geo is not None for track in tracks for point in track.points):
        from drone_vehicle_tracking.visualization.map_viz import render_map

        render_map(
            tracks,
            output_dir / config.map_html,
            config.moving_min_displacement_m,
            config.position_error_m,
        )

    if cot_path is not None:
        from drone_vehicle_tracking.export.cot import write_cot

        write_cot(
            tracks,
            cot_path,
            cot_type=config.cot_type,
            stale_seconds=config.cot_stale_seconds,
            position_error_m=config.position_error_m,
        )
    return tracks
