"""Analytical ground-accuracy error budget for the nadir projection.

This propagates the dominant sensor uncertainties of a telemetry-driven flat-
ground projection into ground error (metres), so the accuracy claim rests on
numbers with a known cause rather than adjectives. For a near-nadir model the
dominant terms are:

* **GNSS horizontal error** -> 1:1 ground error. This is the absolute-accuracy
  floor and, without RTK/GCPs, it dominates; it does not depend on the geometry.
* **Gimbal angle error** (pitch/roll) -> ``altitude * tan(error)`` lateral shift,
  independent of where in the image the pixel lies (it tilts the whole ray bundle).
* **Altitude error** and **focal-length / HFOV miscalibration** -> both *scale* the
  ground offset, so their error is proportional to the pixel's radial ground
  distance and vanishes at the nadir point.
* **Heading (yaw) error** -> *rotates* the ground-offset vector; the resulting
  displacement is the chord ``2 * offset * sin(error / 2)``, also proportional to
  the radial ground distance and zero at nadir.

Independent error sources combine in quadrature (root-sum-square).

:class:`AccuracyModel` binds these terms to a camera and a set of error
coefficients so the budget can be evaluated *per projected pixel* at runtime --
turning the static table below into the self-reported accuracy surfaced on each
track (CoT ``ce``, GeoJSON, map).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from drone_vehicle_tracking.geo.camera import CameraModel
from drone_vehicle_tracking.telemetry.models import TelemetryFrame


def tilt_error_m(altitude_m: float, tilt_error_deg: float) -> float:
    """Lateral ground shift from a residual gimbal pitch/roll error."""
    return altitude_m * math.tan(math.radians(tilt_error_deg))


def scale_error_m(ground_offset_m: float, relative_error: float) -> float:
    """Ground error from a proportional scale error (altitude or focal length).

    ``relative_error`` is a fraction, e.g. ``0.01`` for a 1 % altitude or
    focal-length uncertainty.
    """
    return abs(ground_offset_m * relative_error)


def heading_error_m(ground_offset_m: float, yaw_error_deg: float) -> float:
    """Tangential ground error from a heading (yaw) error, as the rotation chord."""
    return 2.0 * ground_offset_m * abs(math.sin(math.radians(yaw_error_deg) / 2.0))


def root_sum_square(values: Iterable[float]) -> float:
    """Quadrature sum of independent error terms."""
    return math.sqrt(sum(v * v for v in values))


def point_error_m(
    altitude_m: float,
    ground_offset_m: float,
    *,
    tilt_error_deg: float,
    altitude_relative_error: float,
    focal_relative_error: float,
    yaw_error_deg: float,
    gnss_error_m: float,
) -> float:
    """Geometry-aware ground accuracy (metres) at a single projected pixel.

    Combines the budget terms in quadrature for this point's own altitude and
    radial ground offset ``r`` from nadir, then RSS'd with the GNSS floor. The
    scale and heading terms vanish at nadir (``ground_offset_m == 0``), so the
    result there is the floor (plus any residual tilt); it grows toward the image
    edge and with altitude. The result is therefore always at least the floor.
    """
    return root_sum_square(
        [
            tilt_error_m(altitude_m, tilt_error_deg),
            scale_error_m(ground_offset_m, altitude_relative_error),
            scale_error_m(ground_offset_m, focal_relative_error),
            heading_error_m(ground_offset_m, yaw_error_deg),
            gnss_error_m,
        ]
    )


@dataclass(frozen=True, slots=True)
class AccuracyModel:
    """Per-point ground-accuracy estimator bound to a camera and error coefficients.

    The radial ground offset is derived from the pixel geometry
    (``radius_px * GSD``, with ``GSD = altitude / focal_px``), not from the
    projected coordinate, so the estimate depends only on the camera, the frame's
    altitude and the pixel -- never on the projection's geodetic conversion.
    """

    camera: CameraModel
    altitude_source: str
    tilt_error_deg: float
    altitude_relative_error: float
    focal_relative_error: float
    yaw_error_deg: float
    gnss_error_m: float

    def error_for(self, pixel_xy: tuple[float, float], telemetry: TelemetryFrame) -> float:
        """Self-reported horizontal accuracy (metres) for one pixel in this frame."""
        altitude = float(getattr(telemetry, self.altitude_source))
        fx, fy, cx, cy = self.camera.intrinsics()
        offset_east = (pixel_xy[0] - cx) * altitude / fx
        offset_north = (pixel_xy[1] - cy) * altitude / fy
        ground_offset = math.hypot(offset_east, offset_north)
        return point_error_m(
            altitude,
            ground_offset,
            tilt_error_deg=self.tilt_error_deg,
            altitude_relative_error=self.altitude_relative_error,
            focal_relative_error=self.focal_relative_error,
            yaw_error_deg=self.yaw_error_deg,
            gnss_error_m=self.gnss_error_m,
        )
