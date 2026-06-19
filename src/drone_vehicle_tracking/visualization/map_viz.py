"""Render geo-referenced vehicle tracks onto an interactive folium/Leaflet map."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from drone_vehicle_tracking.telemetry.models import Track


def render_map(tracks: Sequence[Track], output_html: str | Path) -> None:
    """Write an interactive HTML map with one coloured polyline per track."""
    raise NotImplementedError
