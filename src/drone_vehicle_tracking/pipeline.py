"""End-to-end orchestration: telemetry + video -> geo-referenced tracks -> map.

This module wires the stages together but contains no algorithm logic itself,
so each stage stays independently testable and swappable.
"""

from __future__ import annotations

from pathlib import Path


def run(video_path: str | Path, srt_path: str | Path, config_path: str | Path) -> None:
    """Run detect -> track -> geo-reference -> visualize for one flight."""
    raise NotImplementedError
