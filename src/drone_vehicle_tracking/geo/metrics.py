"""Geometric metrics over geo-referenced tracks.

Pure and dependency-light (geodesic distance only), so it runs in CI and is
reused by the visualization layer to distinguish moving from stationary vehicles
-- the task asks specifically for *moving* cars, and a parked car's track still
carries small jitter, so a net-displacement threshold separates the two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class ScatterStats:
    """Spread of a set of geo points about their centroid, in metres."""

    rms_m: float
    max_m: float
    count: int


def geo_centroid(points: Sequence[GeoPoint]) -> GeoPoint:
    """Arithmetic mean position of a *local* cluster (small enough to ignore
    meridian convergence and antimeridian wrap)."""
    n = len(points)
    return GeoPoint(
        latitude=sum(p.latitude for p in points) / n,
        longitude=sum(p.longitude for p in points) / n,
    )


def reprojection_scatter_m(points: Sequence[GeoPoint]) -> ScatterStats:
    """Spread of repeated projections of a (nominally fixed) ground feature.

    A stationary feature seen across frames is projected independently each
    frame; with perfect geometry/telemetry those projections coincide. The
    geodesic distance of each projection from their centroid therefore measures
    the pipeline's *relative* (repeatability) accuracy -- no ground-control point
    required. Reported as RMS and maximum deviation. Note this is an upper bound
    on the projection error alone, since it also absorbs detection pixel jitter.
    """
    n = len(points)
    if n == 0:
        return ScatterStats(rms_m=0.0, max_m=0.0, count=0)
    centroid = geo_centroid(points)
    deviations = [
        float(_GEOD.inv(centroid.longitude, centroid.latitude, p.longitude, p.latitude)[2])
        for p in points
    ]
    rms = (sum(d * d for d in deviations) / n) ** 0.5
    return ScatterStats(rms_m=rms, max_m=max(deviations), count=n)
