from pathlib import Path

from drone_vehicle_tracking.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_default_config() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config.classes == ("car", "van", "truck", "bus")
    assert config.camera_model == "mavic_3t_wide"
    assert config.altitude_source == "rel_alt"
    assert config.frame_stride >= 1
    assert config.imgsz >= 640
    assert 0.0 < config.conf_threshold < 1.0
    assert config.min_track_length >= 1
    assert config.map_html.endswith(".html")
    assert config.moving_min_displacement_m > 0.0
    assert config.smoothing_window >= 1
    assert config.min_track_confidence >= 0.0
    assert config.cot_type == "a-u-G"
    assert config.cot_stale_seconds > 0
    assert config.position_error_m > 0.0
    assert config.tilt_error_deg > 0.0
    assert 0.0 < config.altitude_relative_error < 1.0
    assert 0.0 < config.focal_relative_error < 1.0
    assert config.yaw_error_deg > 0.0
    assert config.stabilize is True
    assert config.stabilization_window > 1


def test_stabilization_defaults_off_without_processing_section(tmp_path: Path) -> None:
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text(
        "detection:\n"
        "  model: unused.pt\n"
        "  conf_threshold: 0.25\n"
        "  imgsz: 1280\n"
        "  classes: [car]\n"
        "tracking:\n"
        "  min_track_length: 1\n"
        "camera:\n"
        "  model: mavic_3t_wide\n"
        "projection:\n"
        "  altitude_source: rel_alt\n"
        "io:\n"
        "  frame_stride: 1\n"
        "  output_dir: outputs\n"
        "visualization:\n"
        "  map_html: map.html\n"
        "  moving_min_displacement_m: 3.0\n"
    )
    config = load_config(minimal)
    assert config.stabilize is False
    assert config.stabilization_window == 61
