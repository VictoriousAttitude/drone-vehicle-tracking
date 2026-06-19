from datetime import datetime
from pathlib import Path

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
