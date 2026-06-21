"""Typed pipeline configuration loaded from YAML.

A single dataclass is the contract between the YAML file and the orchestration,
so the rest of the code never reaches into raw dictionaries and every tunable
has one documented home.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Resolved, validated settings consumed by :func:`pipeline.run`."""

    model: str
    conf_threshold: float
    imgsz: int
    classes: tuple[str, ...]
    min_track_length: int
    camera_model: str
    altitude_source: str
    frame_stride: int
    output_dir: str
    map_html: str
    moving_min_displacement_m: float
    smoothing_window: int
    min_track_confidence: float


def load_config(path: str | Path) -> PipelineConfig:
    """Parse a pipeline YAML config into a :class:`PipelineConfig`."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    detection = data["detection"]
    tracking = data["tracking"]
    projection = data["projection"]
    io = data["io"]
    visualization = data["visualization"]
    processing = data.get("processing", {})
    return PipelineConfig(
        model=str(detection["model"]),
        conf_threshold=float(detection["conf_threshold"]),
        imgsz=int(detection["imgsz"]),
        classes=tuple(str(c) for c in detection["classes"]),
        min_track_length=int(tracking["min_track_length"]),
        camera_model=str(data["camera"]["model"]),
        altitude_source=str(projection["altitude_source"]),
        frame_stride=int(io["frame_stride"]),
        output_dir=str(io["output_dir"]),
        map_html=str(visualization["map_html"]),
        moving_min_displacement_m=float(visualization["moving_min_displacement_m"]),
        smoothing_window=int(processing.get("smoothing_window", 1)),
        min_track_confidence=float(processing.get("min_track_confidence", 0.0)),
    )
