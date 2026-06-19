"""Parser for DJI SRT subtitle files (per-frame flight telemetry).

DJI groups fields into brackets inconsistently, e.g.
``[rel_alt: 102.229 abs_alt: 426.185]`` and
``[gb_yaw: -65.8 gb_pitch: -89.9 gb_roll: 0.0]``. We therefore extract each
numeric field by key, independent of bracket grouping, which is robust across
firmware variants.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from drone_vehicle_tracking.telemetry.models import TelemetryFrame

_FRAME_CNT_RE = re.compile(r"FrameCnt:\s*(\d+)")
_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
_NUMERIC_FIELDS = (
    "latitude",
    "longitude",
    "rel_alt",
    "abs_alt",
    "gb_yaw",
    "gb_pitch",
    "gb_roll",
    "focal_len",
)
_FIELD_RES = {name: re.compile(rf"{name}:\s*(-?\d+(?:\.\d+)?)") for name in _NUMERIC_FIELDS}


def parse_srt(path: str | Path) -> list[TelemetryFrame]:
    """Parse a DJI SRT file into an ordered list of per-frame telemetry.

    Args:
        path: Path to the ``.SRT`` file.

    Returns:
        Telemetry frames in file order (FrameCnt ascending).

    Raises:
        ValueError: If a telemetry block is missing an expected numeric field.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    frames: list[TelemetryFrame] = []
    for block in text.split("\n\n"):
        m_frame = _FRAME_CNT_RE.search(block)
        m_ts = _TIMESTAMP_RE.search(block)
        if m_frame is None or m_ts is None:
            continue
        values: dict[str, float] = {}
        for name, regex in _FIELD_RES.items():
            match = regex.search(block)
            if match is None:
                raise ValueError(f"Missing field '{name}' in SRT block:\n{block.strip()}")
            values[name] = float(match.group(1))
        frames.append(
            TelemetryFrame(
                frame_index=int(m_frame.group(1)),
                timestamp=datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S.%f"),
                latitude=values["latitude"],
                longitude=values["longitude"],
                rel_alt=values["rel_alt"],
                abs_alt=values["abs_alt"],
                gimbal_yaw=values["gb_yaw"],
                gimbal_pitch=values["gb_pitch"],
                gimbal_roll=values["gb_roll"],
                focal_len=values["focal_len"],
            )
        )
    return frames
