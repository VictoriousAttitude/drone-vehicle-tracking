"""The pipeline-facing stabilization stage (implements ``interfaces.Stabilizer``).

Streams frames: only the previous grayscale frame is retained, so memory stays
constant regardless of flight length. Registration results are accumulated and
turned into corrected telemetry in one pass by
:func:`drone_vehicle_tracking.geo.fusion.fuse_telemetry`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.geo.camera import CameraModel
from drone_vehicle_tracking.geo.fusion import FramePair, fuse_telemetry
from drone_vehicle_tracking.stabilization.registration import estimate_transform, to_gray
from drone_vehicle_tracking.telemetry.models import TelemetryFrame


class TelemetryStabilizer:
    """Registers consecutive frames and fuses the motion into the telemetry."""

    def __init__(
        self, camera: CameraModel, altitude_source: str = "rel_alt", window: int = 61
    ) -> None:
        if altitude_source not in ("rel_alt", "abs_alt"):
            raise ValueError(f"Unknown altitude_source: {altitude_source!r}")
        self._camera = camera
        self._altitude_source = altitude_source
        self._window = window
        self._prev: tuple[int, npt.NDArray[np.uint8]] | None = None
        self._pairs: list[FramePair] = []

    def observe(self, frame_index: int, image: npt.NDArray[np.uint8]) -> None:
        """Register ``image`` against the previously observed frame."""
        gray = to_gray(image)
        if self._prev is not None:
            prev_index, prev_gray = self._prev
            transform = estimate_transform(prev_gray, gray)
            if transform is not None:
                self._pairs.append(FramePair(prev_index, frame_index, transform))
        self._prev = (frame_index, gray)

    def corrected(
        self, telemetry_by_index: Mapping[int, TelemetryFrame]
    ) -> dict[int, TelemetryFrame]:
        """Fused telemetry for every frame inside a registration segment."""
        return fuse_telemetry(
            self._pairs, telemetry_by_index, self._camera, self._altitude_source, self._window
        )
