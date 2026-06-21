"""Cursor-on-Target (CoT) export for geo-referenced vehicle tracks.

CoT is the XML event format ingested by TAK clients (ATAK/WinTAK) and servers,
so emitting it lets the tracker feed a common operating picture directly. Each
track becomes one CoT ``<event>`` at its last known position; the vehicle's
heading and speed ride in a ``<track>`` sub-element, and the *self-reported*
horizontal accuracy is carried in the standard ``ce`` (circular error) field.

This is the practical, immediately-ingestible track format. The MISB analogue
for moving-target tracks is ST 0903 (VMTI), which is binary KLV multiplexed into
an ST 0601 video stream -- a heavier, video-coupled artefact noted as future work.

Pure and dependency-light (stdlib XML plus pyproj for the heading azimuth), so it
runs in CI without the ``[cv]`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from pyproj import Geod

from drone_vehicle_tracking.geo.metrics import (
    track_geo_points,
    track_position_error_m,
    track_speed,
)
from drone_vehicle_tracking.telemetry.models import GeoPoint, Track
from drone_vehicle_tracking.tracking.quality import track_mean_confidence

_GEOD = Geod(ellps="WGS84")
_UNKNOWN = "9999999.0"  # CoT sentinel for an unmodelled numeric field
_XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def _iso_utc(moment: datetime) -> str:
    """Format a CoT timestamp: ISO-8601 UTC, millisecond precision, trailing 'Z'.

    A naive datetime (the DJI SRT timestamps are naive) is taken to be UTC.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def final_bearing_deg(points: Sequence[GeoPoint]) -> float | None:
    """Heading of the last path segment: forward azimuth in degrees clockwise
    from true north (0..360). ``None`` when fewer than two points are given."""
    if len(points) < 2:
        return None
    before, last = points[-2], points[-1]
    azimuth = _GEOD.inv(before.longitude, before.latitude, last.longitude, last.latitude)[0]
    return float(azimuth) % 360.0


def track_to_cot_event(
    track: Track,
    *,
    cot_type: str,
    stale_seconds: float,
    position_error_m: float,
    generated_at: datetime,
) -> Element | None:
    """Build a CoT ``<event>`` for a track's last known position.

    Returns ``None`` for a track with no geo-located point. The event time is the
    last geo-located point's telemetry timestamp, falling back to ``generated_at``
    when the track carries no timestamps. The CoT ``ce`` (self-reported circular
    error, metres) is the track's worst-case geometry-aware accuracy when its
    points carry one, else the ``position_error_m`` fallback.
    """
    points = track_geo_points(track)
    if not points:
        return None
    last = points[-1]

    event_time = generated_at
    for point in reversed(track.points):
        if point.geo is not None and point.timestamp is not None:
            event_time = point.timestamp
            break

    event = Element("event")
    event.set("version", "2.0")
    event.set("uid", f"dvt-vehicle-{track.track_id}")
    event.set("type", cot_type)
    event.set("how", "m-g")  # machine-generated, geo-derived
    event.set("time", _iso_utc(event_time))
    event.set("start", _iso_utc(event_time))
    event.set("stale", _iso_utc(event_time + timedelta(seconds=stale_seconds)))

    point_el = SubElement(event, "point")
    point_el.set("lat", f"{last.latitude:.8f}")
    point_el.set("lon", f"{last.longitude:.8f}")
    point_el.set("hae", _UNKNOWN)  # height above ellipsoid unmodelled (flat-ground)
    track_error = track_position_error_m(track)
    ce = track_error if track_error is not None else position_error_m
    point_el.set("ce", f"{ce:.1f}")  # self-reported horizontal accuracy
    point_el.set("le", _UNKNOWN)  # vertical error unmodelled

    detail = SubElement(event, "detail")
    SubElement(detail, "contact").set("callsign", f"VEH-{track.track_id}")

    speed = track_speed(track)
    if speed is not None:
        bearing = final_bearing_deg(points)
        assert bearing is not None  # a speed implies >=2 timed, hence geo-located, points
        track_el = SubElement(detail, "track")
        track_el.set("course", f"{bearing:.1f}")
        track_el.set("speed", f"{speed.mean_speed_mps:.2f}")

    remarks = [f"class={track.class_name}"]
    confidence = track_mean_confidence(track)
    if confidence is not None:
        remarks.append(f"confidence={confidence:.2f}")
    SubElement(detail, "remarks").text = " ".join(remarks)

    return event


def tracks_to_cot(
    tracks: Sequence[Track],
    *,
    cot_type: str = "a-u-G",
    stale_seconds: float = 60.0,
    position_error_m: float = 3.0,
    generated_at: datetime | None = None,
) -> str:
    """Serialise tracks to a Cursor-on-Target XML document.

    The root ``<events>`` holds one standalone CoT ``<event>`` per geo-located
    track (TAK consumers ingest events individually; the wrapper just makes the
    file a single well-formed document). ``cot_type`` defaults to ``a-u-G`` --
    MIL-STD-2525 atom / unknown affiliation / Ground -- since a detector cannot
    establish friend-or-foe. Tracks with no geo-located point are skipped.
    """
    moment = generated_at or datetime.now(timezone.utc)
    root = Element("events")
    for track in tracks:
        event = track_to_cot_event(
            track,
            cot_type=cot_type,
            stale_seconds=stale_seconds,
            position_error_m=position_error_m,
            generated_at=moment,
        )
        if event is not None:
            root.append(event)
    indent(root)
    return f"{_XML_DECLARATION}\n{tostring(root, encoding='unicode')}\n"


def write_cot(
    tracks: Sequence[Track],
    output_path: str | Path,
    *,
    cot_type: str = "a-u-G",
    stale_seconds: float = 60.0,
    position_error_m: float = 3.0,
) -> None:
    """Write the CoT document for ``tracks`` to ``output_path``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        tracks_to_cot(
            tracks,
            cot_type=cot_type,
            stale_seconds=stale_seconds,
            position_error_m=position_error_m,
        )
    )
