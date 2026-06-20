"""Real-data integration proof.

This is the one claim that *cannot* be simulated honestly: that the detector
finds real vehicles in genuine nadir imagery and the pipeline geo-references them.
It runs only against real data supplied via environment variables, so it is
skipped in CI (also carries the ``integration`` marker) and never depends on any
committed footage.

Run locally with::

    DVT_VIDEO=/path/flight.MP4 DVT_SRT=/path/flight.SRT \
        pytest -m integration tests/integration
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_real_flight_yields_geo_referenced_tracks(tmp_path: Path) -> None:
    video = os.environ.get("DVT_VIDEO")
    srt = os.environ.get("DVT_SRT")
    if not video or not srt:
        pytest.skip("set DVT_VIDEO and DVT_SRT to run the real-data proof")
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    pytest.importorskip("trackers")

    from drone_vehicle_tracking.pipeline import run

    model = os.environ.get("DVT_MODEL", "models/yolov8s-visdrone.pt")
    out_dir = tmp_path / "out"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "detection:\n"
        f"  model: {model}\n"
        "  conf_threshold: 0.25\n"
        "  imgsz: 1280\n"
        "  classes: [car, van, truck, bus]\n"
        "tracking:\n"
        "  min_track_length: 10\n"
        "camera:\n"
        "  model: mavic_3t_wide\n"
        "projection:\n"
        "  altitude_source: rel_alt\n"
        "io:\n"
        "  frame_stride: 1\n"
        f"  output_dir: {out_dir}\n"
        "visualization:\n"
        "  map_html: map.html\n"
        "  moving_min_displacement_m: 3.0\n"
    )

    tracks = run(video, srt, cfg)

    assert tracks, "expected at least one vehicle track on real footage"
    geo_points = [p for t in tracks for p in t.points if p.geo is not None]
    assert geo_points, "expected geo-referenced points"
    assert (out_dir / "tracks.geojson").exists()
    assert (out_dir / "map.html").exists()
