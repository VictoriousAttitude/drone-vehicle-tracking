"""Structural interfaces (Protocols) for the swappable pipeline stages.

Stages depend only on these Protocols, never on concrete implementations, so a
YOLO detector can be swapped for RT-DETR, or ByteTrack for BoT-SORT, without
touching the orchestration. Tests use trivial fakes that satisfy these
Protocols.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from drone_vehicle_tracking.telemetry.models import Detection, GeoPoint, TelemetryFrame, Track


@runtime_checkable
class Detector(Protocol):
    """Detects vehicles in a single image frame."""

    def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    """Associates per-frame detections into stable, ID'd tracks."""

    def update(self, frame_index: int, detections: Sequence[Detection]) -> None: ...

    def finalize(self) -> list[Track]: ...


@runtime_checkable
class Projector(Protocol):
    """Maps an image pixel to a WGS84 ground coordinate for a given frame pose."""

    def pixel_to_geo(
        self, pixel_xy: tuple[float, float], telemetry: TelemetryFrame
    ) -> GeoPoint: ...
