"""Tests for the visual ego-motion + telemetry fusion.

The oracle tests build pixel correspondences analytically from a *known* pair
of poses (via the closed-form nadir inverse projection) and assert that the
solver recovers that pose exactly, even when the current frame's telemetry
reports a wrong position and yaw. The filter tests feed exact visual motion
plus noisy telemetry and assert the noise is suppressed while segment
endpoints stay telemetry-anchored.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from drone_vehicle_tracking.geo.camera import MAVIC_3T_WIDE, CameraModel
from drone_vehicle_tracking.geo.fusion import (
    FramePair,
    FrameTransform,
    fuse_telemetry,
    solve_pose,
)
from drone_vehicle_tracking.geo.projection import enu_to_geo, pixel_to_local_enu, utm_epsg
from drone_vehicle_tracking.telemetry.models import TelemetryFrame

_LAT0, _LON0 = 48.1, 25.2
_ALT = 80.0
_TS = datetime(2024, 11, 24, 17, 38, 3)

# Non-collinear pixels used to build synthetic transforms from known poses.
_PIXELS = ((200.0, 150.0), (1700.0, 200.0), (900.0, 950.0))


def _tel(frame_index: int, lat: float, lon: float, yaw: float) -> TelemetryFrame:
    return TelemetryFrame(
        frame_index=frame_index,
        timestamp=_TS,
        latitude=lat,
        longitude=lon,
        rel_alt=_ALT,
        abs_alt=_ALT + 100.0,
        gimbal_yaw=yaw,
        gimbal_pitch=-90.0,
        gimbal_roll=0.0,
        focal_len=24.0,
    )


def _ground_to_pixel(
    ground_en: tuple[float, float], yaw_deg: float, camera: CameraModel, altitude: float
) -> tuple[float, float]:
    """Closed-form nadir inverse of the projector (ENU offset from drone -> pixel)."""
    fx, fy, cx, cy = camera.intrinsics()
    a = math.radians(-yaw_deg)
    east, north = ground_en
    east0 = math.cos(a) * east + math.sin(a) * north
    north0 = -math.sin(a) * east + math.cos(a) * north
    return cx + east0 * fx / altitude, cy - north0 * fy / altitude


def _fit_transform(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> FrameTransform:
    """Exact affine fit through pixel correspondences (least squares)."""
    rows, rhs = [], []
    for (u, v), (x, y) in pairs:
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0])
        rows.append([0.0, 0.0, 0.0, u, v, 1.0])
        rhs.extend([x, y])
    sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    a, b, tx, c, d, ty = (float(s) for s in sol)
    return FrameTransform(matrix=(a, b, tx, c, d, ty))


def _transform_between(
    prev_pos: tuple[float, float],
    prev_yaw: float,
    curr_pos: tuple[float, float],
    curr_yaw: float,
    camera: CameraModel,
) -> FrameTransform:
    """Pixel transform between two known poses (positions in a shared ENU frame)."""
    prev_tel = _tel(0, _LAT0, _LON0, prev_yaw)  # lat/lon unused by pixel_to_local_enu
    pairs = []
    for p in _PIXELS:
        offset = pixel_to_local_enu(p, prev_tel, camera, _ALT)
        ground = (prev_pos[0] + offset[0], prev_pos[1] + offset[1])
        from_curr = (ground[0] - curr_pos[0], ground[1] - curr_pos[1])
        pairs.append((p, _ground_to_pixel(from_curr, curr_yaw, camera, _ALT)))
    return _fit_transform(pairs)


def _to_enu(lat: float, lon: float) -> tuple[float, float]:
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg(_LAT0, _LON0)}", always_xy=True)
    e0, n0 = to_utm.transform(_LON0, _LAT0)
    e1, n1 = to_utm.transform(lon, lat)
    return float(e1 - e0), float(n1 - n0)


def test_frame_transform_applies_affine() -> None:
    transform = FrameTransform(matrix=(1.0, 0.0, 5.0, 0.0, 1.0, -3.0))
    assert transform.apply((2.0, 2.0)) == (7.0, -1.0)


def test_ground_to_pixel_inverts_the_projector() -> None:
    tel = _tel(1, _LAT0, _LON0, 33.0)
    for pixel in _PIXELS:
        ground = pixel_to_local_enu(pixel, tel, MAVIC_3T_WIDE, _ALT)
        back = _ground_to_pixel(ground, 33.0, MAVIC_3T_WIDE, _ALT)
        assert back == pytest.approx(pixel, abs=1e-9)


def test_solve_pose_recovers_known_motion_despite_wrong_telemetry() -> None:
    """Oracle: the solver must return the *true* pose change to numerical precision.

    The current frame's telemetry deliberately lies about position (unchanged)
    and yaw (37 instead of the true 40): the solver may use only its angles and
    altitude, so position noise cannot leak into the increment and the yaw is
    corrected from the imagery.
    """
    true_delta, true_yaw = (5.0, -3.0), 40.0
    transform = _transform_between((0.0, 0.0), 30.0, true_delta, true_yaw, MAVIC_3T_WIDE)
    prev = _tel(1, _LAT0, _LON0, 30.0)
    curr_reported = _tel(2, _LAT0, _LON0, 37.0)  # wrong position, wrong yaw

    east, north, yaw = solve_pose(transform, prev, curr_reported, MAVIC_3T_WIDE)

    assert east == pytest.approx(true_delta[0], abs=1e-6)
    assert north == pytest.approx(true_delta[1], abs=1e-6)
    assert yaw == pytest.approx(true_yaw, abs=1e-6)


def test_solve_pose_identity_transform_means_no_motion() -> None:
    identity = FrameTransform(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    prev = _tel(1, _LAT0, _LON0, 12.0)
    curr = _tel(2, _LAT0, _LON0, 12.0)
    east, north, yaw = solve_pose(identity, prev, curr, MAVIC_3T_WIDE)
    assert (east, north) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert yaw == pytest.approx(12.0, abs=1e-9)


def test_fuse_telemetry_suppresses_gnss_noise_and_anchors_endpoints() -> None:
    """Exact visual motion + noisy GNSS: fused error must shrink several-fold."""
    n = 41
    rng = np.random.default_rng(7)
    truth = [(0.5 * i, 0.0) for i in range(n)]
    noise = rng.normal(0.0, 2.0, size=(n, 2))
    telemetry: dict[int, TelemetryFrame] = {}
    for i in range(n):
        geo = enu_to_geo(truth[i][0] + noise[i][0], truth[i][1] + noise[i][1], _LAT0, _LON0)
        telemetry[i + 1] = _tel(i + 1, geo.latitude, geo.longitude, 0.0)
    pairs = [
        FramePair(i + 1, i + 2, _transform_between(truth[i], 0.0, truth[i + 1], 0.0, MAVIC_3T_WIDE))
        for i in range(n - 1)
    ]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=21)

    assert set(fused) == set(telemetry)
    errors = []
    for i in range(n):
        e, north = _to_enu(fused[i + 1].latitude, fused[i + 1].longitude)
        errors.append((e - truth[i][0]) ** 2 + (north - truth[i][1]) ** 2)
    fused_rms = math.sqrt(sum(errors) / n)
    raw_rms = math.sqrt(float((noise**2).sum()) / n)
    assert fused_rms < 0.5 * raw_rms
    # Segment endpoints stay anchored to telemetry (the absolute datum is GNSS).
    assert fused[1].latitude == pytest.approx(telemetry[1].latitude, abs=1e-9)
    assert fused[n].longitude == pytest.approx(telemetry[n].longitude, abs=1e-9)


def test_fuse_telemetry_corrects_a_yaw_spike_the_imagery_contradicts() -> None:
    """Stationary drone, identity transforms, one 5-degree telemetry yaw spike.

    Dead-reckoned yaw stays at 10; the spike survives only through the smoothed
    residual: mean([0, 0, 5, 0, 0]) = 1 at the centre, so fused yaw is 11.
    """
    yaws = [10.0, 10.0, 15.0, 10.0, 10.0]
    telemetry = {i + 1: _tel(i + 1, _LAT0, _LON0, yaw) for i, yaw in enumerate(yaws)}
    identity = FrameTransform(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    pairs = [FramePair(i, i + 1, identity) for i in range(1, 5)]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=5)

    assert fused[3].gimbal_yaw == pytest.approx(11.0, abs=1e-9)
    assert fused[1].gimbal_yaw == pytest.approx(10.0, abs=1e-9)
    assert fused[3].latitude == pytest.approx(_LAT0, abs=1e-9)


def test_fuse_telemetry_dead_reckons_zigzag_position_the_telemetry_missed() -> None:
    """Zigzag visual motion vs flat telemetry: known-answer interior fusion.

    Telemetry says the drone never moved; the imagery says it zigzagged
    east (0,1,0,-1,0) and north (0,-1,0,1,0). At frame 2 the shrunken
    residual window is [0,-1,0] -> mean -1/3, so fused east = 1 - 1/3 = 2/3
    (north mirrored). This pins the *sign* of the dead-reckoned increments,
    which the noise test cannot: there the truth motion is linear and a
    centred moving average reproduces linear ramps exactly, so a sign flip
    still passes the RMS bound.
    """
    truth = [(0.0, 0.0), (1.0, -1.0), (0.0, 0.0), (-1.0, 1.0), (0.0, 0.0)]
    telemetry = {i + 1: _tel(i + 1, _LAT0, _LON0, 10.0) for i in range(5)}
    pairs = [
        FramePair(
            i + 1, i + 2, _transform_between(truth[i], 10.0, truth[i + 1], 10.0, MAVIC_3T_WIDE)
        )
        for i in range(4)
    ]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=5)

    east, north = _to_enu(fused[2].latitude, fused[2].longitude)
    assert east == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert north == pytest.approx(-2.0 / 3.0, abs=1e-6)


def test_fuse_telemetry_dead_reckons_zigzag_yaw_the_telemetry_missed() -> None:
    """Visual yaw zigzag (10,11,10,9,10) vs constant telemetry yaw 10.

    Dead-reckoned yaw follows the imagery (the previous frame's telemetry
    yaw error cancels inside the increment), so the residual at frame 2 is
    smoothed over [0,-1,0] -> fused yaw = 11 - 1/3. Pins the yaw-increment
    sign the identity-transform spike test cannot (its increments are zero).
    """
    truth_yaw = [10.0, 11.0, 10.0, 9.0, 10.0]
    telemetry = {i + 1: _tel(i + 1, _LAT0, _LON0, 10.0) for i in range(5)}
    pairs = [
        FramePair(
            i + 1,
            i + 2,
            _transform_between(
                (0.0, 0.0), truth_yaw[i], (0.0, 0.0), truth_yaw[i + 1], MAVIC_3T_WIDE
            ),
        )
        for i in range(4)
    ]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=5)

    assert fused[2].gimbal_yaw == pytest.approx(11.0 - 1.0 / 3.0, abs=1e-6)
    east, north = _to_enu(fused[2].latitude, fused[2].longitude)
    assert (east, north) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_fuse_telemetry_window_of_two_is_active() -> None:
    """window=2 (half-width 1) must correct, not silently disable."""
    yaws = [10.0, 10.0, 16.0, 10.0, 10.0]
    telemetry = {i + 1: _tel(i + 1, _LAT0, _LON0, yaw) for i, yaw in enumerate(yaws)}
    identity = FrameTransform(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    pairs = [FramePair(i, i + 1, identity) for i in range(1, 5)]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=2)

    assert set(fused) == set(telemetry)
    # half-width 1: residual [0,0,6,0,0] -> centre mean(0,6,0) = 2 -> yaw 12.
    assert fused[3].gimbal_yaw == pytest.approx(12.0, abs=1e-9)


def test_fuse_telemetry_window_of_one_disables_correction() -> None:
    identity = FrameTransform(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    telemetry = {1: _tel(1, _LAT0, _LON0, 0.0), 2: _tel(2, _LAT0, _LON0, 0.0)}
    assert fuse_telemetry([FramePair(1, 2, identity)], telemetry, MAVIC_3T_WIDE, window=1) == {}


def test_fuse_telemetry_no_pairs_returns_nothing() -> None:
    assert fuse_telemetry([], {1: _tel(1, _LAT0, _LON0, 0.0)}, MAVIC_3T_WIDE, window=5) == {}


def test_missing_telemetry_splits_the_chain_into_segments() -> None:
    identity = FrameTransform(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    telemetry = {i: _tel(i, _LAT0, _LON0, 0.0) for i in (1, 2, 4, 5)}  # no frame 3
    pairs = [FramePair(i, i + 1, identity) for i in range(1, 5)]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=5)

    assert set(fused) == {1, 2, 4, 5}


def test_registration_gap_splits_the_chain_into_segments() -> None:
    identity = FrameTransform(matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
    telemetry = {i: _tel(i, _LAT0, _LON0, 0.0) for i in (1, 2, 3, 7, 8)}
    pairs = [FramePair(1, 2, identity), FramePair(2, 3, identity), FramePair(7, 8, identity)]

    fused = fuse_telemetry(pairs, telemetry, MAVIC_3T_WIDE, window=5)

    assert set(fused) == {1, 2, 3, 7, 8}
