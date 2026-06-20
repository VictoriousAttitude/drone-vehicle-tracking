"""Geometric metrics over geo-referenced tracks.

Pure and dependency-light (geodesic distance only), so it runs in CI and is
reused by the visualization layer to distinguish moving from stationary vehicles
-- the task asks specifically for *moving* cars, and a parked car's track still
carries small jitter, so a net-displacement threshold separates the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyproj import Geod

from drone_vehicle_tracking.telemetry.models import GeoPoint, Track

_GEOD = Geod(ellps="WGS84")


def track_geo_points(track: Track) -> list[GeoPoint]:
    """Return only the track's points that carry a geo coordinate, in order."""
    return [point.geo for point in track.points if point.geo is not None]


def path_length_m(points: Sequence[GeoPoint]) -> float:
    """Total along-path geodesic distance (metres) over consecutive points."""
    total = 0.0
    for a, b in zip(points, points[1:], strict=False):
        total += float(_GEOD.inv(a.longitude, a.latitude, b.longitude, b.latitude)[2])
    return total


def net_displacement_m(points: Sequence[GeoPoint]) -> float:
    """Straight-line geodesic distance (metres) from first to last point."""
    if len(points) < 2:
        return 0.0
    first, last = points[0], points[-1]
    return float(_GEOD.inv(first.longitude, first.latitude, last.longitude, last.latitude)[2])
