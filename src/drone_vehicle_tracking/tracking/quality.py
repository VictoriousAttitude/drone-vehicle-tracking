"""Track-quality filtering by detector confidence.

``min_track_length`` already drops tracks that are too *short* to trust, but a
track can clear that gate while being built mostly from borderline detections
just above the per-frame ``conf_threshold``. Averaging each track's carried
detection confidence gives an aggregate quality signal: tracks whose mean falls
below ``min_track_confidence`` are dropped, and the mean is surfaced on the
GeoJSON/map so weak tracks are visible rather than silently kept.

Tracks whose points carry no confidence (e.g. injected or hand-built tracks)
cannot be judged and are kept, so the filter degrades to a no-op rather than
discarding data it cannot assess.
"""

from __future__ import annotations

from collections.abc import Sequence

from drone_vehicle_tracking.telemetry.models import Track


def track_mean_confidence(track: Track) -> float | None:
    """Mean detector confidence over a track's points, or ``None`` if unknown."""
    confidences: list[float] = []
    for point in track.points:
        if point.confidence is not None:
            confidences.append(point.confidence)
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def filter_tracks_by_confidence(tracks: Sequence[Track], min_confidence: float) -> list[Track]:
    """Drop tracks whose mean confidence is below ``min_confidence``.

    ``min_confidence <= 0`` disables the filter. Tracks with no confidence data
    are kept (they cannot be assessed).
    """
    if min_confidence <= 0.0:
        return list(tracks)
    kept: list[Track] = []
    for track in tracks:
        mean = track_mean_confidence(track)
        if mean is None or mean >= min_confidence:
            kept.append(track)
    return kept
