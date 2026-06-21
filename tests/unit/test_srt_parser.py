from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from drone_vehicle_tracking.telemetry.models import TelemetryFrame
from drone_vehicle_tracking.telemetry.srt_parser import parse_srt

# DJI splits these fields across brackets inconsistently across firmware; the
# parser extracts each by key, so a synthetic block can list them in any order,
# in their own brackets or grouped, with arbitrary unknown fields interleaved.
_DEFAULT_FIELDS = {
    "latitude": "50.0",
    "longitude": "30.0",
    "rel_alt": "100.0",
    "abs_alt": "200.0",
    "gb_yaw": "10.0",
    "gb_pitch": "-89.9",
    "gb_roll": "0.0",
    "focal_len": "24.0",
}
_DEFAULT_TS = "2024-11-24 17:38:03.576"


def _block(
    fields: dict[str, str] | None = None,
    *,
    frame: int = 1,
    ts: str = _DEFAULT_TS,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Build one synthetic SRT block; each field gets its own bracket."""
    fields = _DEFAULT_FIELDS if fields is None else fields
    tokens = " ".join(f"[{name}: {value}]" for name, value in fields.items())
    return (
        f"{frame}\n00:00:00,000 --> 00:00:00,033\n"
        f"FrameCnt: {frame}, DiffTime: 33ms\n{ts}\n"
        f"{prefix}{tokens} {suffix}\n"
    )


def _parse_text(tmp_path: Path, text: str) -> list[TelemetryFrame]:
    srt = tmp_path / "synthetic.srt"
    srt.write_text(text)
    return parse_srt(srt)


def test_parses_all_frames(fixtures_dir: Path) -> None:
    frames = parse_srt(fixtures_dir / "sample.srt")
    assert [f.frame_index for f in frames] == [1, 2]


def test_first_frame_fields(fixtures_dir: Path) -> None:
    frame = parse_srt(fixtures_dir / "sample.srt")[0]
    assert frame.latitude == 50.0
    assert frame.longitude == 30.0
    assert frame.rel_alt == 100.0
    assert frame.abs_alt == 200.0
    assert frame.gimbal_yaw == 10.0
    assert frame.gimbal_pitch == -89.9
    assert frame.focal_len == 24.0
    assert frame.timestamp == datetime(2024, 11, 24, 17, 38, 3, 576000)


def test_blocks_without_framecnt_or_timestamp_are_skipped(tmp_path: Path) -> None:
    srt = tmp_path / "noise.srt"
    srt.write_text("just some header text\n\nnot a telemetry block at all\n")
    assert parse_srt(srt) == []


def test_missing_numeric_field_raises(tmp_path: Path) -> None:
    srt = tmp_path / "broken.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,033\n"
        "FrameCnt: 1, DiffTime: 33ms\n2024-11-24 17:38:03.576\n"
        "[latitude: 50.0] [longitude: 30.0]\n"  # missing rel_alt and the rest
    )
    with pytest.raises(ValueError, match="Missing field"):
        parse_srt(srt)


def test_field_order_is_irrelevant(tmp_path: Path) -> None:
    """Fields extracted by key -> a reordered block parses identically."""
    reordered = dict(reversed(list(_DEFAULT_FIELDS.items())))
    frame = _parse_text(tmp_path, _block(reordered))[0]
    assert (frame.latitude, frame.longitude) == (50.0, 30.0)
    assert (frame.rel_alt, frame.abs_alt) == (100.0, 200.0)
    assert (frame.gimbal_yaw, frame.gimbal_pitch, frame.gimbal_roll) == (10.0, -89.9, 0.0)
    assert frame.focal_len == 24.0


def test_unknown_fields_and_no_font_wrapper_are_tolerated(tmp_path: Path) -> None:
    """Camera/exposure fields (and any junk) are ignored; no <font> needed."""
    text = _block(
        {"latitude": "12.5", "longitude": "-7.25", **dict(list(_DEFAULT_FIELDS.items())[2:])},
        prefix="[iso: 100] [shutter: 1/800] [fnum: 2.8] ",
        suffix="[dzoom_ratio: 1.00] [unknown_field: 999]",
    )
    frame = _parse_text(tmp_path, text)[0]
    assert frame.latitude == 12.5
    assert frame.longitude == -7.25


def test_negative_values_keep_their_sign(tmp_path: Path) -> None:
    text = _block(
        {**_DEFAULT_FIELDS, "latitude": "-33.86", "longitude": "-70.66", "gb_yaw": "-179.5"}
    )
    frame = _parse_text(tmp_path, text)[0]
    assert frame.latitude == -33.86
    assert frame.longitude == -70.66
    assert frame.gimbal_yaw == -179.5


def test_nonnumeric_field_value_is_rejected_not_silently_zeroed(tmp_path: Path) -> None:
    """A garbage value must raise, never yield a frame with a silent-zero coord."""
    text = _block({**_DEFAULT_FIELDS, "latitude": "N/A"})
    with pytest.raises(ValueError, match="Missing field 'latitude'"):
        _parse_text(tmp_path, text)


@pytest.mark.parametrize("dropped", list(_DEFAULT_FIELDS))
def test_dropping_any_required_field_raises(tmp_path: Path, dropped: str) -> None:
    fields = {name: value for name, value in _DEFAULT_FIELDS.items() if name != dropped}
    with pytest.raises(ValueError, match=f"Missing field '{dropped}'"):
        _parse_text(tmp_path, _block(fields))


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    frame=st.integers(min_value=0, max_value=10**6),
    latitude=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
    longitude=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
    rel_alt=st.floats(min_value=-100, max_value=1000, allow_nan=False, allow_infinity=False),
    abs_alt=st.floats(min_value=-100, max_value=10000, allow_nan=False, allow_infinity=False),
    gb_yaw=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
    gb_pitch=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
    gb_roll=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
    focal_len=st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
)
def test_roundtrip_preserves_every_field_across_value_space(
    tmp_path: Path,
    frame: int,
    latitude: float,
    longitude: float,
    rel_alt: float,
    abs_alt: float,
    gb_yaw: float,
    gb_pitch: float,
    gb_roll: float,
    focal_len: float,
) -> None:
    """No cross-field leakage: each field round-trips to its own printed value."""
    raw = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "rel_alt": f"{rel_alt:.6f}",
        "abs_alt": f"{abs_alt:.6f}",
        "gb_yaw": f"{gb_yaw:.6f}",
        "gb_pitch": f"{gb_pitch:.6f}",
        "gb_roll": f"{gb_roll:.6f}",
        "focal_len": f"{focal_len:.6f}",
    }
    frames = _parse_text(tmp_path, _block(raw, frame=frame))
    assert len(frames) == 1
    parsed = frames[0]
    assert parsed.frame_index == frame
    assert parsed.latitude == float(raw["latitude"])
    assert parsed.longitude == float(raw["longitude"])
    assert parsed.rel_alt == float(raw["rel_alt"])
    assert parsed.abs_alt == float(raw["abs_alt"])
    assert parsed.gimbal_yaw == float(raw["gb_yaw"])
    assert parsed.gimbal_pitch == float(raw["gb_pitch"])
    assert parsed.gimbal_roll == float(raw["gb_roll"])
    assert parsed.focal_len == float(raw["focal_len"])
