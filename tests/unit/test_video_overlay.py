import pytest

from drone_vehicle_tracking.telemetry.models import Track, TrackPoint
from drone_vehicle_tracking.visualization.video_overlay import (
    BoxAnnotation,
    overlay_index,
    render_overlay,
)


def _track(track_id: int, points: list[TrackPoint], cls: str = "car") -> Track:
    return Track(track_id=track_id, class_name=cls, points=points)


def test_overlay_index_groups_boxes_by_frame() -> None:
    a = _track(
        1,
        [
            TrackPoint(frame_index=1, pixel_xy=(0.0, 0.0), bbox_xyxy=(0.0, 0.0, 10.0, 10.0)),
            TrackPoint(frame_index=2, pixel_xy=(0.0, 0.0), bbox_xyxy=(1.0, 1.0, 11.0, 11.0)),
        ],
    )
    b = _track(
        2,
        [TrackPoint(frame_index=1, pixel_xy=(0.0, 0.0), bbox_xyxy=(5.0, 5.0, 15.0, 15.0))],
    )
    index = overlay_index([a, b])
    assert set(index) == {1, 2}
    assert len(index[1]) == 2  # both tracks present on frame 1
    assert len(index[2]) == 1
    ann = index[2][0]
    assert isinstance(ann, BoxAnnotation)
    assert ann.track_id == 1
    assert ann.bbox_xyxy == (1.0, 1.0, 11.0, 11.0)


def test_overlay_index_skips_points_without_bbox() -> None:
    track = _track(
        1,
        [
            TrackPoint(frame_index=1, pixel_xy=(0.0, 0.0), bbox_xyxy=None),
            TrackPoint(frame_index=2, pixel_xy=(0.0, 0.0), bbox_xyxy=(0.0, 0.0, 4.0, 4.0)),
        ],
    )
    index = overlay_index([track])
    assert set(index) == {2}  # the bbox-less point produced no annotation


def test_distinct_tracks_get_distinct_colors() -> None:
    points = [TrackPoint(frame_index=1, pixel_xy=(0.0, 0.0), bbox_xyxy=(0.0, 0.0, 1.0, 1.0))]
    index = overlay_index([_track(0, points), _track(1, points)])
    assert index[1][0].color != index[1][1].color


def test_render_overlay_writes_annotated_video(make_video, tmp_path) -> None:
    video = make_video(8)
    track = _track(
        3,
        [
            TrackPoint(frame_index=2, pixel_xy=(60.0, 140.0), bbox_xyxy=(40.0, 100.0, 80.0, 140.0)),
            # frame 100 does not exist in the clip -> exercises the empty .get default
            TrackPoint(frame_index=100, pixel_xy=(0.0, 0.0), bbox_xyxy=(0.0, 0.0, 4.0, 4.0)),
        ],
    )
    out = tmp_path / "overlay.mp4"
    render_overlay(video, [track], out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_overlay_missing_file_raises(tmp_path) -> None:
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError):
        render_overlay(tmp_path / "nope.mp4", [], tmp_path / "out.mp4")
