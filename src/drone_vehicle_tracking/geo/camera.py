"""Camera models and ground-sampling-distance math.

The horizontal FOV here is derived from the published DJI Mavic 3T wide-camera
spec (1/2" CMOS, 24mm-equivalent, DFOV 84 deg). It is a nominal starting point;
``hfov_deg`` should be refined by empirical calibration against a ground feature
of known size to reach the 1 m accuracy target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Pinhole camera description sufficient for nadir ground projection."""

    name: str
    sensor_width_mm: float
    sensor_height_mm: float
    image_width_px: int
    image_height_px: int
    hfov_deg: float

    def gsd(self, altitude_m: float) -> float:
        """Ground sampling distance (metres per pixel) for a nadir view."""
        ground_width_m = 2.0 * altitude_m * math.tan(math.radians(self.hfov_deg / 2.0))
        return ground_width_m / self.image_width_px

    @property
    def focal_length_px(self) -> float:
        """Focal length in pixels, derived from HFOV (square-pixel assumption)."""
        return (self.image_width_px / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))

    def intrinsics(self) -> tuple[float, float, float, float]:
        """Pinhole intrinsics ``(fx, fy, cx, cy)`` in pixels.

        Assumes square pixels (fx == fy) and a principal point at the image
        centre — adequate for a dewarped DJI feed; refine via calibration if
        sub-pixel accuracy at the frame edges is required.
        """
        f = self.focal_length_px
        return f, f, self.image_width_px / 2.0, self.image_height_px / 2.0


# DJI Mavic 3T, wide camera. HFOV derived from DFOV 84 deg on a 4:3 1/2" sensor.
MAVIC_3T_WIDE = CameraModel(
    name="DJI Mavic 3T (wide)",
    sensor_width_mm=6.4,
    sensor_height_mm=4.8,
    image_width_px=1920,
    image_height_px=1080,
    hfov_deg=71.5,
)

CAMERA_REGISTRY: dict[str, CameraModel] = {
    "mavic_3t_wide": MAVIC_3T_WIDE,
}
