import math
from datetime import datetime

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from drone_vehicle_tracking.geo.camera import MAVIC_3T_WIDE
from drone_vehicle_tracking.geo.projection import (
    NadirProjector,
    enu_to_geo,
    nadir_gsd_offset,
    pixel_to_local_enu,
)
from drone_vehicle_tracking.telemetry.models import TelemetryFrame

CAM = MAVIC_3T_WIDE
CX, CY = CAM.image_width_px / 2.0, CAM.image_height_px / 2.0


def _tele(
    yaw: float = 0.0,
    pitch: float = -90.0,
    roll: float = 0.0,
    lat: float = 48.0,
    lon: float = 25.0,
) -> TelemetryFrame:
    return TelemetryFrame(
        frame_index=1,
        timestamp=datetime(2024, 1, 1),
        latitude=lat,
        longitude=lon,
        rel_alt=100.0,
        abs_alt=300.0,
        gimbal_yaw=yaw,
        gimbal_pitch=pitch,
        gimbal_roll=roll,
        focal_len=24.0,
    )


# --- Known-geometry tests (hand-derived expected values) ---


def test_center_pixel_has_zero_ground_offset() -> None:
    east, north = pixel_to_local_enu((CX, CY), _tele(), CAM, 100.0)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_pixel_right_of_centre_maps_east_at_yaw_zero() -> None:
    gsd = CAM.gsd(100.0)
    east, north = pixel_to_local_enu((CX + 100, CY), _tele(yaw=0.0), CAM, 100.0)
    assert east == pytest.approx(100 * gsd, rel=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_pixel_above_centre_maps_north_at_yaw_zero() -> None:
    gsd = CAM.gsd(100.0)
    # Smaller v == higher in the image == further North.
    east, north = pixel_to_local_enu((CX, CY - 100), _tele(yaw=0.0), CAM, 100.0)
    assert north == pytest.approx(100 * gsd, rel=1e-9)
    assert east == pytest.approx(0.0, abs=1e-9)


def test_yaw_90_rotates_image_right_towards_south() -> None:
    # Heading East (yaw=90): the image-right direction (East at yaw 0) rotates
    # to point South under a clockwise compass convention.
    _, north = pixel_to_local_enu((CX + 100, CY), _tele(yaw=90.0), CAM, 100.0)
    gsd = CAM.gsd(100.0)
    assert north == pytest.approx(-100 * gsd, rel=1e-6)


# --- General ray-cast must reduce to the simple GSD model at nadir ---


def test_general_projector_matches_nadir_oracle() -> None:
    for u in (0, 320, 960, 1600, 1919):
        for v in (0, 270, 540, 810, 1079):
            for yaw in (0.0, 37.0, 90.0, 180.0, -123.0):
                tele = _tele(yaw=yaw, pitch=-90.0)
                got = pixel_to_local_enu((u, v), tele, CAM, 102.0)
                expected = nadir_gsd_offset((u, v), tele, CAM, 102.0)
                assert np.allclose(got, expected, atol=1e-6)


# --- Scaling / physical invariants ---


def test_offset_scales_linearly_with_altitude_even_off_nadir() -> None:
    tele = _tele(yaw=20.0, pitch=-80.0)
    o100 = np.array(pixel_to_local_enu((1500, 300), tele, CAM, 100.0))
    o200 = np.array(pixel_to_local_enu((1500, 300), tele, CAM, 200.0))
    assert np.allclose(o200, 2.0 * o100, rtol=1e-9)


def test_horizon_ray_has_no_ground_intersection() -> None:
    with pytest.raises(ValueError):
        pixel_to_local_enu((CX, CY), _tele(pitch=0.0), CAM, 100.0)


# --- Proof of the error-budget figure: a 0.1 deg tilt off nadir ---


def test_tenth_degree_tilt_shifts_centre_by_known_amount() -> None:
    h = 102.0
    expected = h * math.tan(math.radians(0.1))  # ~0.178 m
    east, north = pixel_to_local_enu((CX, CY), _tele(yaw=0.0, pitch=-89.9), CAM, h)
    assert math.hypot(east, north) == pytest.approx(expected, rel=1e-6)
    assert expected == pytest.approx(0.178, abs=0.005)


# --- Property-based invariants ---


@settings(max_examples=200)
@given(yaw=st.floats(min_value=-180.0, max_value=180.0))
def test_yaw_preserves_ground_offset_magnitude(yaw: float) -> None:
    ref = math.hypot(*pixel_to_local_enu((1400, 250), _tele(yaw=0.0), CAM, 100.0))
    rot = math.hypot(*pixel_to_local_enu((1400, 250), _tele(yaw=yaw), CAM, 100.0))
    assert rot == pytest.approx(ref, rel=1e-9)


@settings(max_examples=200)
@given(
    h1=st.floats(min_value=20.0, max_value=300.0),
    h2=st.floats(min_value=20.0, max_value=300.0),
)
def test_offset_magnitude_monotonic_in_altitude(h1: float, h2: float) -> None:
    tele = _tele(yaw=15.0, pitch=-85.0)
    m1 = math.hypot(*pixel_to_local_enu((1700, 200), tele, CAM, h1))
    m2 = math.hypot(*pixel_to_local_enu((1700, 200), tele, CAM, h2))
    if h1 < h2:
        assert m1 < m2
    elif h1 > h2:
        assert m1 > m2
    else:
        assert m1 == pytest.approx(m2)


# --- Geodetic conversion cross-checked against an independent geodesic solver ---


def test_enu_to_geo_distance_matches_geodesic() -> None:
    from pyproj import Geod

    lat0, lon0 = 48.267013, 25.914562
    east, north = 123.4, -56.7
    point = enu_to_geo(east, north, lat0, lon0)
    geod = Geod(ellps="WGS84")
    _, _, dist = geod.inv(lon0, lat0, point.longitude, point.latitude)
    assert dist == pytest.approx(math.hypot(east, north), rel=1e-3)


def test_enu_to_geo_axes_point_the_right_way() -> None:
    # Pin the sign of both axes (a distance-only check cannot): +north must raise
    # latitude and +east must raise longitude. The off-axis component is only the
    # small grid-north vs true-north convergence (this works in the UTM grid frame).
    base = enu_to_geo(0.0, 0.0, 48.0, 25.0)
    north = enu_to_geo(0.0, 50.0, 48.0, 25.0)
    east = enu_to_geo(50.0, 0.0, 48.0, 25.0)
    assert north.latitude > base.latitude
    assert east.longitude > base.longitude
    assert abs(north.longitude - base.longitude) < 1e-4  # convergence only
    assert abs(east.latitude - base.latitude) < 1e-4


def test_projector_center_pixel_returns_drone_position() -> None:
    tele = _tele(lat=48.267013, lon=25.914562)
    point = NadirProjector(CAM).pixel_to_geo((CX, CY), tele)
    assert point.latitude == pytest.approx(48.267013, abs=1e-7)
    assert point.longitude == pytest.approx(25.914562, abs=1e-7)


def test_same_ground_feature_reprojects_consistently_across_poses() -> None:
    """With perfect telemetry, one fixed ground point imaged from two different
    drone positions must project to the same WGS84 coordinate. This isolates the
    projection geometry: its cross-frame error is zero, so any real-world scatter
    is sensor (GNSS/attitude) noise, not the maths.

    The drone is moved in the projector's own ENU/UTM frame (via ``enu_to_geo``)
    rather than along a true-north geodesic, so the two legs share a grid axis and
    meridian convergence (grid-north vs true-north) does not leak into the result.
    """
    from drone_vehicle_tracking.geo.metrics import reprojection_scatter_m

    h = 100.0
    gsd = CAM.gsd(h)
    d = 20.0  # drone moves 20 m north (grid) between the two frames

    # Frame A: drone over the feature -> feature is at the centre pixel.
    tele_a = _tele(yaw=0.0, lat=48.0, lon=25.0)
    point_a = NadirProjector(CAM).pixel_to_geo((CX, CY), tele_a)

    # Frame B: drone d metres north -> the same feature is d/gsd pixels *south*
    # (larger v) of centre.
    drone_b = enu_to_geo(0.0, d, 48.0, 25.0)
    tele_b = _tele(yaw=0.0, lat=drone_b.latitude, lon=drone_b.longitude)
    point_b = NadirProjector(CAM).pixel_to_geo((CX, CY + d / gsd), tele_b)

    scatter = reprojection_scatter_m([point_a, point_b])
    assert scatter.max_m < 0.01  # sub-centimetre: geometry adds no cross-frame error


def test_projector_rejects_unknown_altitude_source() -> None:
    with pytest.raises(ValueError, match="altitude_source"):
        NadirProjector(CAM, altitude_source="bogus")


def test_projector_uses_abs_alt_when_configured() -> None:
    tele = _tele()
    far = NadirProjector(CAM, "abs_alt").pixel_to_geo((CX + 100, CY), tele)
    near = NadirProjector(CAM, "rel_alt").pixel_to_geo((CX + 100, CY), tele)
    # abs_alt (300) is higher than rel_alt (100) -> larger ground offset for the
    # same pixel, so the projected longitude lands further east.
    assert far.longitude > near.longitude
