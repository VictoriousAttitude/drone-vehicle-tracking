"""Render geo-referenced vehicle tracks onto an interactive folium/Leaflet map.

The task asks specifically for *moving* cars, so each track is classified by its
net displacement: tracks that travel at least ``moving_min_displacement_m`` are
drawn as solid coloured paths with start/end markers, while near-stationary ones
(parked cars whose track is GNSS/pixel jitter) are drawn faint and dashed so the
moving traffic stands out without discarding any detected vehicle.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from drone_vehicle_tracking.geo.metrics import (
    net_displacement_m,
    path_length_m,
    track_geo_points,
    track_speed,
)
from drone_vehicle_tracking.telemetry.models import GeoPoint, Track
from drone_vehicle_tracking.tracking.quality import track_mean_confidence

# Distinct, colour-blind-friendly hues cycled across moving tracks.
_PALETTE = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9a6324",
)
_STATIONARY_COLOR = "#808080"
_START_COLOR = "#1a9850"
_END_COLOR = "#d73027"

_SATELLITE_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_SATELLITE_ATTR = "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics"


def _mean_center(points: Sequence[GeoPoint]) -> tuple[float, float]:
    lat = sum(p.latitude for p in points) / len(points)
    lon = sum(p.longitude for p in points) / len(points)
    return lat, lon


def render_map(
    tracks: Sequence[Track],
    output_html: str | Path,
    moving_min_displacement_m: float = 3.0,
) -> None:
    """Write an interactive HTML map with one polyline per geo-located track.

    Raises ``ValueError`` if no track carries at least two geo-located points,
    since there is then nothing to plot.
    """
    import folium

    geo_by_track = [(track, track_geo_points(track)) for track in tracks]
    geo_by_track = [(track, pts) for track, pts in geo_by_track if len(pts) >= 2]
    if not geo_by_track:
        raise ValueError("No track has two or more geo-located points to plot.")

    all_points = [point for _, pts in geo_by_track for point in pts]
    fmap = folium.Map(location=list(_mean_center(all_points)), zoom_start=17, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(fmap)
    folium.TileLayer(_SATELLITE_TILES, attr=_SATELLITE_ATTR, name="Satellite").add_to(fmap)

    color_index = 0
    for track, points in geo_by_track:
        displacement = net_displacement_m(points)
        length = path_length_m(points)
        moving = displacement >= moving_min_displacement_m
        if moving:
            color = _PALETTE[color_index % len(_PALETTE)]
            color_index += 1
        else:
            color = _STATIONARY_COLOR

        speed = track_speed(track)
        speed_html = f"mean speed: {speed.mean_speed_kmh:.1f} km/h<br>" if speed is not None else ""
        confidence = track_mean_confidence(track)
        conf_html = f"mean confidence: {confidence:.2f}<br>" if confidence is not None else ""
        latlon = [(p.latitude, p.longitude) for p in points]
        popup = folium.Popup(
            f"<b>track {track.track_id}</b> ({track.class_name})<br>"
            f"net displacement: {displacement:.1f} m<br>"
            f"path length: {length:.1f} m<br>"
            f"{speed_html}"
            f"{conf_html}"
            f"points: {len(points)}",
            max_width=250,
        )
        folium.PolyLine(
            latlon,
            color=color,
            weight=4 if moving else 2,
            opacity=0.9 if moving else 0.4,
            dash_array=None if moving else "4",
            popup=popup,
        ).add_to(fmap)

        if moving:
            folium.CircleMarker(
                latlon[0], radius=4, color=_START_COLOR, fill=True, fill_opacity=1.0
            ).add_to(fmap)
            folium.CircleMarker(
                latlon[-1], radius=4, color=_END_COLOR, fill=True, fill_opacity=1.0
            ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.fit_bounds(
        [
            [min(p.latitude for p in all_points), min(p.longitude for p in all_points)],
            [max(p.latitude for p in all_points), max(p.longitude for p in all_points)],
        ]
    )
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))
