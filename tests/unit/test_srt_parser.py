from datetime import datetime
from pathlib import Path

import pytest

from drone_vehicle_tracking.telemetry.srt_parser import parse_srt


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
