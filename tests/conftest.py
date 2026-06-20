from collections.abc import Callable, Sequence
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _srt_block(index: int, lat: float, lon: float) -> str:
    ms = (index - 1) * 33
    start = f"00:00:{ms // 1000:02d},{ms % 1000:03d}"
    return (
        f"{index}\n"
        f"{start} --> {start}\n"
        f'<font size="28">FrameCnt: {index}, DiffTime: 33ms\n'
        f"2024-11-24 17:38:{3 + index // 30:02d}.{(index * 33) % 1000:03d}\n"
        f"[iso: 100] [shutter: 1/800] [fnum: 2.8] [ev: 0] [focal_len: 24.00] "
        f"[dzoom_ratio: 1.00], [latitude: {lat:.6f}] [longitude: {lon:.6f}] "
        f"[rel_alt: 80.000 abs_alt: 180.000] "
        f"[gb_yaw: 0.0 gb_pitch: -90.0 gb_roll: 0.0] </font>\n"
    )


@pytest.fixture
def make_srt(tmp_path: Path) -> Callable[[Sequence[tuple[float, float]]], Path]:
    """Return a factory that writes a synthetic DJI SRT for the given per-frame coords."""

    def _make(coords: Sequence[tuple[float, float]]) -> Path:
        blocks = [_srt_block(i, lat, lon) for i, (lat, lon) in enumerate(coords, start=1)]
        path = tmp_path / "synthetic.srt"
        path.write_text("\n".join(blocks))
        return path

    return _make


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[[int, int, int], Path]:
    """Return a factory that writes a synthetic MP4 with a moving white square.

    Skips the test if OpenCV is unavailable (the light CI environment).
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    def _make(num_frames: int = 10, width: int = 320, height: int = 240) -> Path:
        path = tmp_path / "synthetic.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        assert writer.isOpened()
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 20 + i * 5
            frame[100:140, x : x + 40] = 255
            writer.write(frame)
        writer.release()
        return path

    return _make
