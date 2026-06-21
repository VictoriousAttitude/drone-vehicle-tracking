"""Shared, immutable domain contracts passed between pipeline stages.

Keeping these as small frozen dataclasses (no behaviour) decouples the stages:
the parser, detector, tracker, projector and visualizer only depend on these
types, never on each other's implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """Drone pose for a single video frame, parsed from the DJI SRT."""

    frame_index: int  # 1-based FrameCnt, aligned 1:1 to a decoded video frame
    timestamp: datetime
    latitude: float  # WGS84 degrees, drone position
    longitude: float
    rel_alt: float  # metres above takeoff point
    abs_alt: float  # metres above sea level
    gimbal_yaw: float  # degrees (camera heading)
    gimbal_pitch: float  # degrees; ~ -90 == nadir (straight down)
    gimbal_roll: float
    focal_len: float  # 35mm-equivalent focal length, mm


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """A WGS84 ground coordinate."""

    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class Detection:
    """A single vehicle detection in image space."""

    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """One position of a tracked vehicle.

    The tracker produces ``pixel_xy`` (ground-contact point), ``bbox_xyxy`` (the
    detection box, kept for annotated-video overlay) and ``confidence`` (the
    detector score, kept for track-quality filtering); the geo-referencing stage
    then attaches ``geo`` and ``timestamp`` from that frame's telemetry.
    """

    frame_index: int
    pixel_xy: tuple[float, float]
    geo: GeoPoint | None = None
    timestamp: datetime | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    confidence: float | None = None


@dataclass
class Track:
    """A single vehicle's trajectory across frames."""

    track_id: int
    class_name: str
    points: list[TrackPoint]
