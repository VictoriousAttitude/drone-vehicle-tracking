"""Geo-space track smoothing to suppress GNSS/pixel jitter.

A vehicle track carries two independent noise sources — the drone's GNSS (metres
without RTK) and per-frame detection-box jitter — which make plotted paths
zig-zag and inflate path length and speed. A centred moving average over each
track's WGS84 coordinates low-passes that jitter.

Smoothing is done in geo space (after projection), never in pixel space: a
parked car's pixels still travel as the drone flies and yaws, so only the
projected ground positions isolate the true jitter. Latitude and longitude are
filtered independently; the window is symmetric and shrinks at the ends, so the
first and last fixes stay anchored (net displacement is preserved) while
interior jitter is averaged out. A constant-velocity Kalman/RTS smoother would
be the next step if a motion model were wanted, but the moving average needs no
tuning, no extra dependency and invents no positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from drone_vehicle_tracking.telemetry.models import GeoPoint, Track


def _moving_average(values: list[float], half_width: int) -> list[float]:
    """Centred moving average with symmetric windows that shrink at both ends."""
    n = len(values)
    out: list[float] = []
    for i in range(n):
        half = min(half_width, i, n - 1 - i)
        window = values[i - half : i + half + 1]
        out.append(sum(window) / len(window))
    return out


def smooth_track(track: Track, window: int) -> Track:
    """Return ``track`` with its geo-located points moving-averaged.

    ``window`` is the full window size; ``window <= 1`` disables smoothing and
    returns the track unchanged. Points without a geo fix are left untouched and
    excluded from the averaged sequence.
    """
    if window <= 1:
        return track
    half = window // 2
    indexed: list[tuple[int, GeoPoint]] = []
    for i, point in enumerate(track.points):
        if point.geo is not None:
            indexed.append((i, point.geo))

    lats = _moving_average([geo.latitude for _, geo in indexed], half)
    lons = _moving_average([geo.longitude for _, geo in indexed], half)

    points = list(track.points)
    for (i, _), lat, lon in zip(indexed, lats, lons, strict=True):
        points[i] = replace(track.points[i], geo=GeoPoint(lat, lon))
    return Track(track_id=track.track_id, class_name=track.class_name, points=points)


def smooth_tracks(tracks: Sequence[Track], window: int) -> list[Track]:
    """Apply :func:`smooth_track` to every track."""
    return [smooth_track(track, window) for track in tracks]
