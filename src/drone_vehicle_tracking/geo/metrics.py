"""Geometric metrics over geo-referenced tracks.

Pure and dependency-light (geodesic distance only), so it runs in CI and is
reused by the visualization layer to distinguish moving from stationary vehicles
-- the task asks specifically for *moving* cars, and a parked car's track still
carries small jitter, so a net-displacement threshold separates the two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

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
class TrackSpeed:
    """Average speed of one track: along-path distance over elapsed SRT time."""

    duration_s: float
    distance_m: float
    mean_speed_mps: float

    @property
    def mean_speed_kmh(self) -> float:
        """Mean speed in kilometres per hour."""
        return self.mean_speed_mps * 3.6


def track_speed(track: Track) -> TrackSpeed | None:
    """Mean speed over a track's geo-located, time-stamped points.

    Along-path geodesic distance divided by the elapsed time between the first and
    last usable point (timestamps come from the per-frame DJI SRT). Returns
    ``None`` when fewer than two points carry both a geo coordinate and a
    timestamp, or when the elapsed time is not positive. At very low displacement
    the path distance -- hence the speed -- is inflated by GNSS/pixel jitter, so a
    near-stationary track reads a small nonzero speed rather than exactly zero.
    """
    timed: list[tuple[datetime, GeoPoint]] = []
    for point in track.points:
        if point.geo is not None and point.timestamp is not None:
            timed.append((point.timestamp, point.geo))
    if len(timed) < 2:
        return None
    duration = (timed[-1][0] - timed[0][0]).total_seconds()
    if duration <= 0:
        return None
    distance = path_length_m([geo for _, geo in timed])
    return TrackSpeed(duration_s=duration, distance_m=distance, mean_speed_mps=distance / duration)


def track_position_error_m(track: Track) -> float | None:
    """Worst-case (maximum) self-reported accuracy across a track's points.

    Each point's ``position_error_m`` is the geometry-aware ground accuracy
    computed at georeferencing time; the conservative track-level figure is the
    maximum, so a track is never reported as more accurate than its loosest fix.
    Returns ``None`` when no point carries an error (e.g. before georeferencing).
    """
    errors: list[float] = []
    for point in track.points:
        if point.position_error_m is not None:
            errors.append(point.position_error_m)
    return max(errors) if errors else None


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
