"""Tests that exercise the real CV stack (OpenCV + Ultralytics + trackers).

These import-or-skip the heavy ``[cv]`` extra, so they run locally and in the
dedicated ``[cv]`` CI job while being skipped on the light core+dev matrix.
They drive the full ``run()`` path on a *synthetic* video — no drone, no real
imagery — purely to cover the orchestration, video decode and model wiring.
"""

from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")
pytest.importorskip("supervision")
pytest.importorskip("trackers")

from drone_vehicle_tracking.pipeline import iter_video_frames, run  # noqa: E402


def _write_cv_config(tmp_path: Path, output_dir: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "detection:\n"
        "  model: yolov8n.pt\n"  # COCO weights; Ultralytics auto-downloads
        "  conf_threshold: 0.10\n"
        "  imgsz: 320\n"
        "  classes: [car, truck, bus]\n"
        "tracking:\n"
        "  min_track_length: 1\n"
        "camera:\n"
        "  model: mavic_3t_wide\n"
        "projection:\n"
        "  altitude_source: rel_alt\n"
        "io:\n"
        "  frame_stride: 1\n"
        f"  output_dir: {output_dir}\n"
        "visualization:\n"
        "  map_html: map.html\n"
        "  moving_min_displacement_m: 3.0\n"
    )
    return cfg


def test_iter_video_frames_reads_synthetic_mp4(make_video) -> None:
    video = make_video(num_frames=6)
    frames = list(iter_video_frames(video, frame_stride=2))
    assert [idx for idx, _ in frames] == [1, 3, 5]  # 1-based, every 2nd frame
    assert frames[0][1].shape == (240, 320, 3)


def test_iter_video_frames_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_video_frames(tmp_path / "does-not-exist.mp4"))


def test_run_full_stack_on_synthetic_video(tmp_path, make_video, make_srt) -> None:
    video = make_video(num_frames=8)
    srt = make_srt([(48.0 + i * 0.0005, 25.0) for i in range(8)])
    out_dir = tmp_path / "out"
    cfg = _write_cv_config(tmp_path, out_dir)

    # No injected components: this drives the real detector, tracker and decode.
    tracks = run(video, srt, cfg)

    assert isinstance(tracks, list)
    assert (out_dir / "tracks.geojson").exists()
