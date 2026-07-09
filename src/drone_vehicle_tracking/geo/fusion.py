"""Visual ego-motion + telemetry fusion (a complementary filter on the pose).

Without RTK the drone's GNSS fix jitters by metres frame to frame, and that
jitter passes straight through the projector into every geo-referenced track.
Consecutive nadir frames, however, overlap almost completely, so the drone's
*relative* motion between frames can be measured from the imagery itself at
sub-pixel (centimetre-level) precision. The two signals are complementary:

* telemetry is absolute but noisy at high frequency (GNSS jitter),
* visual ego-motion is precise at high frequency but drifts when integrated.

The fusion therefore dead-reckons the visual increments and blends them with
telemetry through a centred moving average of the residual::

    fused = dead_reckoned + MA(telemetry - dead_reckoned)

The moving-average window shrinks symmetrically at segment ends (same scheme
as :mod:`drone_vehicle_tracking.geo.smoothing`), so the first and last fused
poses stay anchored to telemetry: the absolute datum remains GNSS-bound while
the high-frequency jitter is replaced by the visual measurement. Registration
failures split the flight into independent segments; frames outside any
segment keep their raw telemetry.

Geometry: :func:`solve_pose` recovers the inter-frame pose change from a pixel
similarity transform by projecting sample pixels through the *existing*
:func:`~drone_vehicle_tracking.geo.projection.pixel_to_local_enu` for both
frames and solving the resulting 2D rigid alignment in closed form (Kabsch).
Reusing the projector guarantees the fusion shares its coordinate conventions
exactly; the oracle tests recover a known synthetic pose to numerical
precision. Altitude is not fused (the barometric altimeter is already smooth
compared with GNSS horizontal noise).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from drone_vehicle_tracking.geo.camera import CameraModel
from drone_vehicle_tracking.geo.projection import enu_to_geo, pixel_to_local_enu, utm_epsg
from drone_vehicle_tracking.geo.smoothing import _moving_average
from drone_vehicle_tracking.telemetry.models import TelemetryFrame

# Non-collinear sample pixels (offsets from the principal point) through which
# the pixel-space transform is converted into ground-space correspondences.
_SAMPLE_OFFSETS = ((0.0, 0.0), (300.0, 0.0), (0.0, 300.0))


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Similarity transform mapping previous-frame pixels to current-frame pixels.

    ``matrix`` is the row-major 2x3 affine ``(a, b, tx, c, d, ty)`` as returned
    by ``cv2.estimateAffinePartial2D``: ``(u, v) -> (a*u + b*v + tx, c*u + d*v + ty)``.
    """

    matrix: tuple[float, float, float, float, float, float]

    def apply(self, pixel_xy: tuple[float, float]) -> tuple[float, float]:
        """Map a previous-frame pixel to its current-frame position."""
        a, b, tx, c, d, ty = self.matrix
        u, v = pixel_xy
        return a * u + b * v + tx, c * u + d * v + ty


@dataclass(frozen=True, slots=True)
class FramePair:
    """A successful registration between two consecutively processed frames."""

    prev_index: int
    curr_index: int
    transform: FrameTransform


def _altitude(telemetry: TelemetryFrame, altitude_source: str) -> float:
    return float(getattr(telemetry, altitude_source))


def solve_pose(
    transform: FrameTransform,
    prev: TelemetryFrame,
    curr: TelemetryFrame,
    camera: CameraModel,
    altitude_source: str = "rel_alt",
) -> tuple[float, float, float]:
    """Recover the current frame's pose from the previous pose and the transform.

    Returns ``(east_m, north_m, yaw_deg)``: the current drone position as an ENU
    offset from the previous drone position, and the corrected gimbal yaw. Only
    the *angles and altitude* of ``curr`` are consumed (its GNSS lat/lon is not),
    so the recovered increment is immune to GNSS noise.

    Ground features are fixed: a pixel ``p`` in the previous frame and its
    image ``transform(p)`` in the current frame see the same ground point. Both
    are projected to ENU offsets with the frame's own pose; the rigid transform
    aligning the two point sets *is* the pose correction (2D Kabsch).
    """
    _, _, cx, cy = camera.intrinsics()
    ground = []
    offsets = []
    for du, dv in _SAMPLE_OFFSETS:
        p = (cx + du, cy + dv)
        q = transform.apply(p)
        ground.append(pixel_to_local_enu(p, prev, camera, _altitude(prev, altitude_source)))
        offsets.append(pixel_to_local_enu(q, curr, camera, _altitude(curr, altitude_source)))
    a = np.asarray(ground, dtype=np.float64)
    b = np.asarray(offsets, dtype=np.float64)
    a_c, b_c = a.mean(axis=0), b.mean(axis=0)
    m = (b - b_c).T @ (a - a_c)
    theta = math.atan2(m[0, 1] - m[1, 0], m[0, 0] + m[1, 1])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    east = a_c[0] - (cos_t * b_c[0] - sin_t * b_c[1])
    north = a_c[1] - (sin_t * b_c[0] + cos_t * b_c[1])
    # An extra CCW world rotation of the offsets corresponds to a *smaller*
    # compass (CW) yaw; see rotation_camera_to_world's sign convention.
    return float(east), float(north), curr.gimbal_yaw - math.degrees(theta)


def _segments(
    pairs: Sequence[FramePair], telemetry_by_index: Mapping[int, TelemetryFrame]
) -> Iterator[list[FramePair]]:
    """Split registrations into chains of contiguous pairs with telemetry."""
    current: list[FramePair] = []
    for pair in pairs:
        if pair.prev_index not in telemetry_by_index or pair.curr_index not in telemetry_by_index:
            if current:
                yield current
            current = []
        elif current and pair.prev_index != current[-1].curr_index:
            yield current
            current = [pair]
        else:
            current.append(pair)
    if current:
        yield current


def _to_enu(
    frames: Sequence[TelemetryFrame], anchor: TelemetryFrame
) -> tuple[list[float], list[float]]:
    """Telemetry positions as ENU offsets (metres) from ``anchor``, one UTM grid."""
    from pyproj import Transformer

    epsg = utm_epsg(anchor.latitude, anchor.longitude)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e0, n0 = to_utm.transform(anchor.longitude, anchor.latitude)
    easts, norths = to_utm.transform(
        [frame.longitude for frame in frames], [frame.latitude for frame in frames]
    )
    return [float(e - e0) for e in easts], [float(n - n0) for n in norths]


def _fuse_segment(
    segment: list[FramePair],
    telemetry_by_index: Mapping[int, TelemetryFrame],
    camera: CameraModel,
    altitude_source: str,
    half_width: int,
) -> dict[int, TelemetryFrame]:
    frames = [telemetry_by_index[segment[0].prev_index]]
    frames += [telemetry_by_index[pair.curr_index] for pair in segment]
    anchor = frames[0]
    tel_e, tel_n = _to_enu(frames, anchor)

    dr_e, dr_n, dr_yaw = [tel_e[0]], [tel_n[0]], [frames[0].gimbal_yaw]
    for pair, prev_frame, curr_frame in zip(segment, frames[:-1], frames[1:], strict=True):
        de, dn, yaw = solve_pose(pair.transform, prev_frame, curr_frame, camera, altitude_source)
        dr_e.append(dr_e[-1] + de)
        dr_n.append(dr_n[-1] + dn)
        dr_yaw.append(dr_yaw[-1] + (yaw - prev_frame.gimbal_yaw))

    res_e = _moving_average([t - d for t, d in zip(tel_e, dr_e, strict=True)], half_width)
    res_n = _moving_average([t - d for t, d in zip(tel_n, dr_n, strict=True)], half_width)
    raw_res_yaw = [f.gimbal_yaw - d for f, d in zip(frames, dr_yaw, strict=True)]
    res_yaw = _moving_average(
        [float(r) for r in np.degrees(np.unwrap(np.radians(raw_res_yaw)))], half_width
    )

    corrected: dict[int, TelemetryFrame] = {}
    for frame, de_dr, dn_dr, dy_dr, re, rn, ry in zip(
        frames, dr_e, dr_n, dr_yaw, res_e, res_n, res_yaw, strict=True
    ):
        geo = enu_to_geo(de_dr + re, dn_dr + rn, anchor.latitude, anchor.longitude)
        corrected[frame.frame_index] = replace(
            frame, latitude=geo.latitude, longitude=geo.longitude, gimbal_yaw=dy_dr + ry
        )
    return corrected


def fuse_telemetry(
    pairs: Sequence[FramePair],
    telemetry_by_index: Mapping[int, TelemetryFrame],
    camera: CameraModel,
    altitude_source: str = "rel_alt",
    window: int = 61,
) -> dict[int, TelemetryFrame]:
    """Blend telemetry with visual ego-motion; returns the corrected frames.

    ``window`` is the full moving-average window (in processed frames) applied
    to the telemetry-minus-dead-reckoning residual; ``window <= 1`` makes the
    blend an identity, so nothing is corrected and ``{}`` is returned. Only
    frames inside a registration segment appear in the result; callers merge it
    over the raw telemetry.
    """
    if window <= 1:
        return {}
    half_width = window // 2
    corrected: dict[int, TelemetryFrame] = {}
    for segment in _segments(pairs, telemetry_by_index):
        corrected.update(
            _fuse_segment(segment, telemetry_by_index, camera, altitude_source, half_width)
        )
    return corrected
