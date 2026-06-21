from datetime import datetime, timedelta, timezone
from xml.etree.ElementTree import fromstring

import pytest

from drone_vehicle_tracking.export.cot import (
    _iso_utc,
    final_bearing_deg,
    track_to_cot_event,
    tracks_to_cot,
    write_cot,
)
from drone_vehicle_tracking.telemetry.models import GeoPoint, Track, TrackPoint


def _track(
    track_id: int = 1,
    coords: list[tuple[float, float]] | None = None,
    times: list[datetime] | None = None,
    confidences: list[float] | None = None,
    class_name: str = "car",
) -> Track:
    if coords is None:
        coords = [(48.0, 25.0), (48.001, 25.0)]
    points = [
        TrackPoint(
            frame_index=i,
            pixel_xy=(0.0, 0.0),
            geo=GeoPoint(lat, lon),
            timestamp=None if times is None else times[i],
            confidence=None if confidences is None else confidences[i],
        )
        for i, (lat, lon) in enumerate(coords)
    ]
    return Track(track_id=track_id, class_name=class_name, points=points)


def test_iso_utc_formats_naive_as_utc_with_millis() -> None:
    assert _iso_utc(datetime(2024, 1, 1, 12, 0, 0, 123000)) == "2024-01-01T12:00:00.123Z"


def test_iso_utc_converts_aware_time_to_utc() -> None:
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert _iso_utc(aware) == "2024-01-01T10:00:00.000Z"


def test_final_bearing_none_for_single_point() -> None:
    assert final_bearing_deg([GeoPoint(48.0, 25.0)]) is None


def test_final_bearing_due_north_is_zero() -> None:
    bearing = final_bearing_deg([GeoPoint(48.0, 25.0), GeoPoint(48.001, 25.0)])
    assert bearing == pytest.approx(0.0, abs=1.0)


def test_final_bearing_due_east_is_ninety() -> None:
    bearing = final_bearing_deg([GeoPoint(48.0, 25.0), GeoPoint(48.0, 25.001)])
    assert bearing == pytest.approx(90.0, abs=1.0)


def test_event_for_timed_moving_track_carries_track_and_accuracy() -> None:
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    track = _track(
        track_id=7,
        times=[t0, t0 + timedelta(seconds=10)],
        confidences=[0.8, 0.6],
    )
    event = track_to_cot_event(
        track, cot_type="a-u-G", stale_seconds=60, position_error_m=3.0, generated_at=t0
    )
    assert event is not None
    assert event.get("version") == "2.0"
    assert event.get("uid") == "dvt-vehicle-7"
    assert event.get("type") == "a-u-G"
    assert event.get("how") == "m-g"
    # Event time is the last point's timestamp; stale is that + stale_seconds.
    assert event.get("time") == "2024-01-01T12:00:10.000Z"
    assert event.get("stale") == "2024-01-01T12:01:10.000Z"

    point = event.find("point")
    assert point is not None
    assert point.get("ce") == "3.0"  # self-reported accuracy
    assert point.get("lat").startswith("48.001")  # the last position

    track_el = event.find("detail/track")
    assert track_el is not None
    assert float(track_el.get("course")) == pytest.approx(0.0, abs=1.0)  # heading north
    assert float(track_el.get("speed")) > 0.0
    assert event.find("detail/contact").get("callsign") == "VEH-7"
    remarks = event.find("detail/remarks").text
    assert "class=car" in remarks
    assert "confidence=0.70" in remarks  # mean of 0.8 and 0.6


def test_event_without_timestamps_uses_generated_at_and_omits_track() -> None:
    track = _track()  # geo but no timestamps -> speed not computable
    generated = datetime(2024, 1, 1, 9, 30, 0)
    event = track_to_cot_event(
        track, cot_type="a-u-G", stale_seconds=30, position_error_m=2.0, generated_at=generated
    )
    assert event is not None
    assert event.get("time") == "2024-01-01T09:30:00.000Z"  # fell back to generated_at
    assert event.find("detail/track") is None  # no speed -> no <track>
    assert event.find("detail/remarks").text == "class=car"  # no confidence appended


def test_event_none_for_track_without_geo() -> None:
    track = Track(
        track_id=1,
        class_name="car",
        points=[TrackPoint(frame_index=0, pixel_xy=(0.0, 0.0), geo=None)],
    )
    assert (
        track_to_cot_event(
            track,
            cot_type="a-u-G",
            stale_seconds=60,
            position_error_m=3.0,
            generated_at=datetime(2024, 1, 1),
        )
        is None
    )


def test_tracks_to_cot_skips_geoless_track_and_is_well_formed() -> None:
    geo = _track(track_id=1)
    geoless = Track(
        track_id=2,
        class_name="car",
        points=[TrackPoint(frame_index=0, pixel_xy=(0.0, 0.0), geo=None)],
    )
    xml = tracks_to_cot([geo, geoless], generated_at=datetime(2024, 1, 1, 12, 0, 0))
    assert xml.startswith('<?xml version="1.0"')
    root = fromstring(xml)
    assert root.tag == "events"
    events = root.findall("event")
    assert len(events) == 1  # the geoless track produced no event
    assert events[0].get("uid") == "dvt-vehicle-1"


def test_tracks_to_cot_defaults_generated_at_to_now() -> None:
    root = fromstring(tracks_to_cot([_track(track_id=1)]))  # no generated_at -> now()
    assert len(root.findall("event")) == 1


def test_write_cot_writes_parseable_file(tmp_path) -> None:
    out = tmp_path / "nested" / "tracks.cot"
    write_cot([_track(track_id=3)], out, position_error_m=5.0)
    root = fromstring(out.read_text())
    assert root.findall("event")[0].find("point").get("ce") == "5.0"
