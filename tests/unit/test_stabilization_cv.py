"""Tests for frame registration and the telemetry stabilizer (real OpenCV).

Registration correctness is asserted convention-free: the recovered transform
is applied to sample pixels and compared against the known warp. The stabilizer
known-answer test feeds identical frames (true motion = zero) plus a synthetic
GNSS spike and asserts the fused position suppresses the spike by exactly the
moving-average factor -- through the *real* cv2 registration.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from drone_vehicle_tracking.geo.camera import MAVIC_3T_WIDE  # noqa: E402
from drone_vehicle_tracking.interfaces import Stabilizer  # noqa: E402
from drone_vehicle_tracking.stabilization.registration import (  # noqa: E402
    estimate_transform,
    to_gray,
)
from drone_vehicle_tracking.stabilization.stabilizer import TelemetryStabilizer  # noqa: E402
from drone_vehicle_tracking.telemetry.models import TelemetryFrame  # noqa: E402

_LAT0, _LON0 = 48.1, 25.2


def _textured(width: int = 320, height: int = 240) -> np.ndarray:
    """Deterministic feature-rich grayscale image (scattered bright squares)."""
    rng = np.random.default_rng(3)
    image = np.zeros((height, width), dtype=np.uint8)
    for _ in range(60):
        x = int(rng.integers(10, width - 20))
        y = int(rng.integers(10, height - 20))
        image[y : y + 8, x : x + 8] = int(rng.integers(60, 255))
    return image


def _tel(frame_index: int, lat: float, lon: float) -> TelemetryFrame:
    return TelemetryFrame(
        frame_index=frame_index,
        timestamp=datetime(2024, 11, 24, 17, 38, 3),
        latitude=lat,
        longitude=lon,
        rel_alt=80.0,
        abs_alt=180.0,
        gimbal_yaw=0.0,
        gimbal_pitch=-90.0,
        gimbal_roll=0.0,
        focal_len=24.0,
    )


def test_to_gray_converts_bgr_and_passes_through_gray() -> None:
    gray = _textured()
    assert to_gray(gray) is gray
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert to_gray(bgr).shape == gray.shape


@pytest.mark.parametrize(
    "warp",
    [
        np.array([[1.0, 0.0, 5.5], [0.0, 1.0, -3.25]]),  # subpixel translation
        cv2.getRotationMatrix2D((160.0, 120.0), 2.0, 1.0),  # 2 degree rotation
    ],
    ids=["translation", "rotation"],
)
def test_estimate_transform_recovers_known_warp(warp) -> None:
    prev = _textured()
    curr = cv2.warpAffine(prev, np.asarray(warp, dtype=np.float64), (320, 240))

    transform = estimate_transform(prev, curr)

    assert transform is not None
    for pixel in ((80.0, 60.0), (240.0, 60.0), (160.0, 180.0)):
        expected = (
            warp[0, 0] * pixel[0] + warp[0, 1] * pixel[1] + warp[0, 2],
            warp[1, 0] * pixel[0] + warp[1, 1] * pixel[1] + warp[1, 2],
        )
        assert transform.apply(pixel) == pytest.approx(expected, abs=0.5)


def test_estimate_transform_returns_none_on_textureless_image() -> None:
    uniform = np.zeros((240, 320), dtype=np.uint8)
    assert estimate_transform(uniform, uniform) is None


def test_estimate_transform_returns_none_on_too_few_corners() -> None:
    nearly_uniform = np.zeros((240, 320), dtype=np.uint8)
    nearly_uniform[100:120, 100:120] = 255  # a single square: ~4 corners
    assert estimate_transform(nearly_uniform, nearly_uniform) is None


def test_estimate_transform_returns_none_when_tracking_is_lost(monkeypatch) -> None:
    prev = _textured()

    def _lost(prev_img, curr_img, corners, flow):  # noqa: ARG001
        status = np.zeros((len(corners), 1), dtype=np.uint8)
        return corners.copy(), status, np.zeros_like(status, dtype=np.float32)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", _lost)
    assert estimate_transform(prev, prev) is None


def test_estimate_transform_returns_none_on_degenerate_fit(monkeypatch) -> None:
    prev = _textured()
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", lambda *a, **k: (None, None))
    assert estimate_transform(prev, prev) is None


def test_telemetry_stabilizer_satisfies_protocol_and_validates_altitude_source() -> None:
    assert isinstance(TelemetryStabilizer(MAVIC_3T_WIDE), Stabilizer)
    with pytest.raises(ValueError, match="altitude_source"):
        TelemetryStabilizer(MAVIC_3T_WIDE, altitude_source="bogus")


def test_stabilizer_suppresses_gnss_spike_on_static_imagery() -> None:
    """Known answer through the real registration.

    Three identical frames mean the true motion is exactly zero, so the
    dead-reckoned position is constant. Telemetry reports a +2e-5 degree
    (about 2.2 m) latitude spike on the middle frame; a window-3 moving
    average of the residual keeps only a third of it.
    """
    frame = cv2.cvtColor(_textured(), cv2.COLOR_GRAY2BGR)
    spike = 2e-5
    telemetry = {
        1: _tel(1, _LAT0, _LON0),
        2: _tel(2, _LAT0 + spike, _LON0),
        3: _tel(3, _LAT0, _LON0),
    }
    stabilizer = TelemetryStabilizer(MAVIC_3T_WIDE, window=3)
    for index in (1, 2, 3):
        stabilizer.observe(index, frame)

    corrected = stabilizer.corrected(telemetry)

    assert set(corrected) == {1, 2, 3}
    assert corrected[2].latitude == pytest.approx(_LAT0 + spike / 3.0, abs=5e-7)
    assert corrected[1].latitude == pytest.approx(_LAT0, abs=5e-7)  # endpoint anchored


def test_stabilizer_with_failed_registration_corrects_nothing() -> None:
    uniform = np.zeros((240, 320, 3), dtype=np.uint8)
    textured = cv2.cvtColor(_textured(), cv2.COLOR_GRAY2BGR)
    stabilizer = TelemetryStabilizer(MAVIC_3T_WIDE, window=3)
    stabilizer.observe(1, uniform)  # no previous frame yet
    stabilizer.observe(2, textured)  # registration against uniform fails

    assert stabilizer.corrected({1: _tel(1, _LAT0, _LON0), 2: _tel(2, _LAT0, _LON0)}) == {}
