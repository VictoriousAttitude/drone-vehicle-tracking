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

from drone_vehicle_tracking.config import load_config
from drone_vehicle_tracking.geo.camera import CAMERA_REGISTRY
from drone_vehicle_tracking.geo.projection import NadirProjector
from drone_vehicle_tracking.interfaces import Detector, Projector, Tracker
from drone_vehicle_tracking.telemetry.models import TelemetryFrame, Track
from drone_vehicle_tracking.telemetry.srt_parser import parse_srt


def georeference_tracks(
    tracks: list[Track],
    telemetry_by_index: Mapping[int, TelemetryFrame],
    projector: Projector,
) -> list[Track]:
    """Attach a WGS84 ``geo`` coordinate to every track point with telemetry.

    Each point is projected using the telemetry of its own frame, so per-frame
    drone motion and gimbal yaw are respected. Points whose frame has no matching
    telemetry are left with ``geo=None`` rather than dropped.
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
            points.append(replace(point, geo=geo))
        out.append(Track(track_id=track.track_id, class_name=track.class_name, points=points))
    return out


def tracks_to_geojson(tracks: list[Track]) -> dict[str, object]:
    """Serialise geo-referenced tracks to a GeoJSON ``FeatureCollection``.

    Each track becomes a ``LineString`` of ``[lon, lat]`` vertices (GeoJSON axis
    order). Tracks with fewer than two geo-located points are omitted.
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
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "num_points": len(coords),
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
) -> list[Track]:
    """Run detect -> track -> geo-reference for one flight and write outputs.

    Returns the geo-referenced tracks and writes ``tracks.geojson`` (and, when
    any point is geo-located, ``map.html``) into the configured output directory.

    The ``detector``, ``tracker``, ``projector`` and ``frames`` arguments default
    to the production implementations (lazily importing the heavy CV stack) but
    can be injected, which keeps the orchestration testable without that stack.
    """
    config = load_config(config_path)
    telemetry_by_index = {frame.frame_index: frame for frame in parse_srt(srt_path)}

    if projector is None:
        projector = NadirProjector(CAMERA_REGISTRY[config.camera_model], config.altitude_source)
    if detector is None:
        from drone_vehicle_tracking.detection.detector import YoloVehicleDetector

        detector = YoloVehicleDetector(
            config.model, config.conf_threshold, config.classes, config.imgsz
        )
    if tracker is None:
        from drone_vehicle_tracking.tracking.tracker import ByteTrackVehicleTracker

        tracker = ByteTrackVehicleTracker(config.min_track_length)
    if frames is None:
        frames = iter_video_frames(video_path, config.frame_stride)

    for frame_index, image in frames:
        tracker.update(frame_index, detector.detect(frame_index, image))

    tracks = georeference_tracks(tracker.finalize(), telemetry_by_index, projector)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tracks.geojson").write_text(json.dumps(tracks_to_geojson(tracks), indent=2))

    if any(point.geo is not None for track in tracks for point in track.points):
        from drone_vehicle_tracking.visualization.map_viz import render_map

        render_map(tracks, output_dir / config.map_html, config.moving_min_displacement_m)
    return tracks
