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
"""

from __future__ import annotations

import math
from collections.abc import Iterable


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
