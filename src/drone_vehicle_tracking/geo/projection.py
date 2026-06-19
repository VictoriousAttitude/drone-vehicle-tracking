"""Geo-referencing: image pixels -> WGS84 ground coordinates.

The model is a general pinhole ray cast onto a flat ground plane, valid for any
gimbal orientation (not only perfect nadir). The transformation chain is:

    pixel (u, v)
      -> camera ray            d_cam = K^-1 [u, v, 1]
      -> world ray (ENU)       d_world = R(yaw, pitch, roll) . d_cam
      -> ground intersection   t = -H / d_world.z ;  (E, N) = t . d_world[:2]
      -> geographic            (E, N) added to the drone's UTM position -> lat/lon

Design:
* ``pixel_to_local_enu`` is pure NumPy (no pyproj, no I/O) so the hard geometry
  is fully unit-testable and CI-light.
* ``enu_to_geo`` isolates the (heavier) geodetic conversion behind a thin layer.
* At perfect nadir (pitch=-90, roll=0) the ray cast provably reduces to the
  simple ``offset = pixel_delta * GSD`` model -- this is asserted in the tests
  and ``nadir_gsd_offset`` provides that independent oracle.

Coordinate conventions (documented, and the right knobs to confirm against a
known ground feature during calibration):
* World frame is local ENU (x=East, y=North, z=Up).
* Camera frame is OpenCV style (x=right, y=down, z=forward/optical axis).
* DJI ``gimbal_yaw`` is treated as a compass heading (clockwise from North);
  ``gimbal_pitch`` is negative downward (-90 == nadir); ``gimbal_roll`` about
  the optical axis.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.geo.camera import CameraModel
from drone_vehicle_tracking.telemetry.models import GeoPoint, TelemetryFrame

# Camera->world basis at zero yaw/pitch/roll: optical axis -> North, image
# right -> East, image down -> Down. Maps d_cam=[a,b,c] -> (a, c, -b).
_R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _rot_x(theta_rad: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(theta_rad: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_camera_to_world(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> npt.NDArray[np.float64]:
    """Rotation matrix mapping a camera-frame direction to world ENU.

    Intrinsic sequence from the level-North reference ``_R0``: yaw about world
    Up, then pitch about the camera's right axis, then roll about the optical
    axis. DJI yaw is a clockwise compass heading, hence the sign flip into the
    counter-clockwise (mathematical) rotation about Up.
    """
    r_up = _rot_z(math.radians(-yaw_deg))
    rotation = r_up @ _R0 @ _rot_x(math.radians(pitch_deg)) @ _rot_z(math.radians(roll_deg))
    return cast("npt.NDArray[np.float64]", rotation)


def pixel_to_local_enu(
    pixel_xy: tuple[float, float],
    telemetry: TelemetryFrame,
    camera: CameraModel,
    altitude_m: float,
) -> tuple[float, float]:
    """Project a pixel to a local ENU ground offset (metres) from the drone.

    Raises:
        ValueError: If the ray does not point below the horizon (no ground hit).
    """
    fx, fy, cx, cy = camera.intrinsics()
    u, v = pixel_xy
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_world = (
        rotation_camera_to_world(
            telemetry.gimbal_yaw, telemetry.gimbal_pitch, telemetry.gimbal_roll
        )
        @ d_cam
    )
    if d_world[2] >= -1e-9:
        raise ValueError(
            "Camera ray does not intersect the ground plane below the drone "
            f"(gimbal_pitch={telemetry.gimbal_pitch})."
        )
    t = -altitude_m / d_world[2]
    return float(t * d_world[0]), float(t * d_world[1])


def nadir_gsd_offset(
    pixel_xy: tuple[float, float],
    telemetry: TelemetryFrame,
    camera: CameraModel,
    altitude_m: float,
) -> tuple[float, float]:
    """Simple-model ground offset, valid only at perfect nadir (pitch=-90, roll=0).

    Independent oracle used to cross-check the general ray-cast projector.
    """
    fx, fy, cx, cy = camera.intrinsics()
    gsd_x, gsd_y = altitude_m / fx, altitude_m / fy
    east0 = (pixel_xy[0] - cx) * gsd_x
    north0 = -(pixel_xy[1] - cy) * gsd_y
    a = math.radians(-telemetry.gimbal_yaw)
    east = math.cos(a) * east0 - math.sin(a) * north0
    north = math.sin(a) * east0 + math.cos(a) * north0
    return east, north


def enu_to_geo(east_m: float, north_m: float, lat0: float, lon0: float) -> GeoPoint:
    """Convert a local ENU offset (metres) around (lat0, lon0) to WGS84.

    Uses the local UTM zone for a metric, near-isometric tangent plane.
    """
    from pyproj import Transformer

    zone = int((lon0 + 180.0) // 6.0) + 1
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    e0, n0 = to_utm.transform(lon0, lat0)
    lon, lat = to_wgs.transform(e0 + east_m, n0 + north_m)
    return GeoPoint(latitude=float(lat), longitude=float(lon))


class NadirProjector:
    """Projects image pixels onto the ground plane and returns WGS84 coordinates."""

    def __init__(self, camera: CameraModel, altitude_source: str = "rel_alt") -> None:
        if altitude_source not in ("rel_alt", "abs_alt"):
            raise ValueError(f"Unknown altitude_source: {altitude_source!r}")
        self._camera = camera
        self._altitude_source = altitude_source

    def _altitude(self, telemetry: TelemetryFrame) -> float:
        return float(getattr(telemetry, self._altitude_source))

    def pixel_to_geo(self, pixel_xy: tuple[float, float], telemetry: TelemetryFrame) -> GeoPoint:
        """Map a pixel (u, v) to a WGS84 ground coordinate for this frame."""
        altitude = self._altitude(telemetry)
        east, north = pixel_to_local_enu(pixel_xy, telemetry, self._camera, altitude)
        return enu_to_geo(east, north, telemetry.latitude, telemetry.longitude)
