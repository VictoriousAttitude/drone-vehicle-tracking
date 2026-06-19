"""Optional annotated-video output (boxes + track IDs) for visual QA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from drone_vehicle_tracking.telemetry.models import Track


def render_overlay(
    video_path: str | Path, tracks: Sequence[Track], output_path: str | Path
) -> None:
    """Burn track boxes and IDs onto the source video for sanity checking."""
    raise NotImplementedError
