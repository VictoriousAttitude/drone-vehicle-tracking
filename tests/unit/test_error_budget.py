import math
from datetime import datetime

import pytest

from drone_vehicle_tracking.geo.camera import MAVIC_3T_WIDE
from drone_vehicle_tracking.geo.error_budget import (
    AccuracyModel,
    heading_error_m,
    point_error_m,
    root_sum_square,
    scale_error_m,
    tilt_error_m,
)
from drone_vehicle_tracking.telemetry.models import TelemetryFrame


def _telemetry(altitude_m: float) -> TelemetryFrame:
    return TelemetryFrame(
        frame_index=1,
        timestamp=datetime(2024, 1, 1),
        latitude=48.0,
        longitude=25.0,
        rel_alt=altitude_m,
        abs_alt=altitude_m + 100.0,
        gimbal_yaw=0.0,
        gimbal_pitch=-90.0,
        gimbal_roll=0.0,
        focal_len=24.0,
    )


def _model(gnss_error_m: float = 3.0, tilt_error_deg: float = 0.1) -> AccuracyModel:
    return AccuracyModel(
        camera=MAVIC_3T_WIDE,
        altitude_source="rel_alt",
        tilt_error_deg=tilt_error_deg,
        altitude_relative_error=0.01,
        focal_relative_error=0.01,
        yaw_error_deg=0.5,
        gnss_error_m=gnss_error_m,
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


def test_point_error_reduces_to_gnss_floor_at_nadir_with_no_tilt() -> None:
    # r=0 (nadir) and zero tilt -> every geometry term vanishes; only the floor.
    error = point_error_m(
        80.0,
        0.0,
        tilt_error_deg=0.0,
        altitude_relative_error=0.01,
        focal_relative_error=0.01,
        yaw_error_deg=0.5,
        gnss_error_m=3.0,
    )
    assert error == pytest.approx(3.0)


def test_point_error_grows_with_radial_offset_and_stays_above_floor() -> None:
    near = point_error_m(
        80.0,
        0.0,
        tilt_error_deg=0.1,
        altitude_relative_error=0.01,
        focal_relative_error=0.01,
        yaw_error_deg=0.5,
        gnss_error_m=3.0,
    )
    far = point_error_m(
        80.0,
        85.0,
        tilt_error_deg=0.1,
        altitude_relative_error=0.01,
        focal_relative_error=0.01,
        yaw_error_deg=0.5,
        gnss_error_m=3.0,
    )
    assert far > near >= 3.0  # geometry adds on top of the GNSS floor


def test_accuracy_model_center_pixel_is_floor_dominated() -> None:
    model = _model(tilt_error_deg=0.0)
    _fx, _fy, cx, cy = MAVIC_3T_WIDE.intrinsics()
    # A pixel at the principal point has zero radial offset -> just the floor.
    assert model.error_for((cx, cy), _telemetry(80.0)) == pytest.approx(3.0)


def test_accuracy_model_edge_pixel_exceeds_center() -> None:
    model = _model()
    _fx, _fy, cx, cy = MAVIC_3T_WIDE.intrinsics()
    center = model.error_for((cx, cy), _telemetry(80.0))
    corner = model.error_for((0.0, 0.0), _telemetry(80.0))
    assert corner > center >= 3.0
