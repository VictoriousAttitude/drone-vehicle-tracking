import math

import pytest

from drone_vehicle_tracking.geo.error_budget import (
    heading_error_m,
    root_sum_square,
    scale_error_m,
    tilt_error_m,
)


def test_tilt_error_matches_hand_derived_figure() -> None:
    # Same 0.1 deg off-nadir proof used in the projection tests: ~0.178 m at 102 m.
    assert tilt_error_m(102.0, 0.1) == pytest.approx(0.178, abs=0.005)


def test_tilt_error_zero_when_perfectly_level() -> None:
    assert tilt_error_m(100.0, 0.0) == 0.0


def test_tilt_error_scales_with_altitude() -> None:
    assert tilt_error_m(200.0, 0.2) == pytest.approx(2.0 * tilt_error_m(100.0, 0.2))


def test_scale_error_is_proportional_to_offset() -> None:
    assert scale_error_m(50.0, 0.02) == pytest.approx(1.0)
    assert scale_error_m(0.0, 0.02) == 0.0  # zero at nadir
    assert scale_error_m(-30.0, 0.1) == pytest.approx(3.0)  # magnitude only


def test_heading_error_is_rotation_chord() -> None:
    assert heading_error_m(50.0, 0.0) == 0.0  # zero at nadir / no yaw error
    # Rotating a 1 m radial offset by 90 deg moves the point by sqrt(2).
    assert heading_error_m(1.0, 90.0) == pytest.approx(math.sqrt(2.0))
    # Small-angle: ~ offset * radians(error).
    assert heading_error_m(100.0, 0.5) == pytest.approx(100.0 * math.radians(0.5), rel=1e-3)


def test_root_sum_square_combines_in_quadrature() -> None:
    assert root_sum_square([3.0, 4.0]) == pytest.approx(5.0)
    assert root_sum_square([]) == 0.0
    assert root_sum_square([2.0]) == pytest.approx(2.0)
