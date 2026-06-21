"""Render a demo map from synthetic vehicle tracks.

This exists so the project has a reproducible visual that needs *none* of the
gitignored inputs (drone video, DJI telemetry, detector weights) and leaks no
real location: the tracks below are fabricated WGS84 paths, not a real flight.

Run it and open the generated HTML in a browser:

    python examples/make_demo.py

It exercises every map feature: moving tracks (solid coloured paths with
start/end markers and speed/confidence/accuracy popups) and a parked one (faint
grey dashed), over the OpenStreetMap and Esri satellite base layers.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

from drone_vehicle_tracking.telemetry.models import GeoPoint, Track, TrackPoint
from drone_vehicle_tracking.visualization.map_viz import render_map

# Synthetic anchor — an arbitrary placeholder, NOT a real flight location.
# Change these two numbers to relocate the demo; everything else is relative.
_BASE_LAT = 50.0
_BASE_LON = 10.0
_M_PER_DEG_LAT = 111_320.0
_T0 = datetime(2024, 1, 1, 12, 0, 0)


def _to_geo(east_m: float, north_m: float) -> GeoPoint:
    """Offset metres east/north of the anchor into a WGS84 coordinate."""
    lat = _BASE_LAT + north_m / _M_PER_DEG_LAT
    lon = _BASE_LON + east_m / (_M_PER_DEG_LAT * math.cos(math.radians(_BASE_LAT)))
    return GeoPoint(latitude=lat, longitude=lon)


def _arc(
    east_m: float, north_m: float, bearing_deg: float, length_m: float, turn_deg: float, n: int
) -> list[tuple[float, float]]:
    """Walk ``n`` points forward from a start offset, turning ``turn_deg`` per step."""
    step = length_m / (n - 1)
    offsets: list[tuple[float, float]] = []
    bearing = bearing_deg
    for _ in range(n):
        offsets.append((east_m, north_m))
        east_m += step * math.sin(math.radians(bearing))
        north_m += step * math.cos(math.radians(bearing))
        bearing += turn_deg
    return offsets


def _track(
    track_id: int,
    class_name: str,
    offsets: list[tuple[float, float]],
    confidence: float,
    position_error_m: float,
    step_s: float = 1.0,
) -> Track:
    points = [
        TrackPoint(
            frame_index=i,
            pixel_xy=(0.0, 0.0),
            geo=_to_geo(east_m, north_m),
            timestamp=_T0 + timedelta(seconds=i * step_s),
            confidence=confidence,
            position_error_m=position_error_m,
        )
        for i, (east_m, north_m) in enumerate(offsets)
    ]
    return Track(track_id=track_id, class_name=class_name, points=points)


def _demo_tracks() -> list[Track]:
    return [
        _track(
            1, "car", _arc(0.0, 0.0, 30.0, 80.0, 2.0, 12), confidence=0.86, position_error_m=3.4
        ),
        _track(
            2,
            "car",
            _arc(45.0, -25.0, 110.0, 60.0, -1.5, 10),
            confidence=0.78,
            position_error_m=3.7,
        ),
        _track(
            3,
            "truck",
            _arc(-30.0, 35.0, 200.0, 70.0, 1.0, 11),
            confidence=0.71,
            position_error_m=4.1,
        ),
        # Parked: < 3 m of jitter -> classified stationary (faint grey dashed).
        _track(
            4,
            "car",
            [(12.0, -45.0), (12.4, -44.8), (11.7, -45.3), (12.1, -44.9), (11.9, -45.1)],
            confidence=0.64,
            position_error_m=3.2,
        ),
    ]


def main() -> None:
    output = Path(__file__).parent / "demo_map.html"
    render_map(_demo_tracks(), output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
