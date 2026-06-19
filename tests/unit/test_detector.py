import numpy as np

from drone_vehicle_tracking.detection.detector import build_detections
from drone_vehicle_tracking.interfaces import Detector
from drone_vehicle_tracking.telemetry.models import Detection

NAMES = {0: "person", 2: "car", 5: "bus", 7: "truck"}


def test_build_detections_maps_fields() -> None:
    det = build_detections(
        7,
        np.array([[10.0, 20.0, 30.0, 40.0]]),
        np.array([0.91]),
        np.array([2]),
        NAMES,
    )[0]
    assert det.frame_index == 7
    assert det.bbox_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert det.confidence == 0.91
    assert det.class_name == "car"


def test_build_detections_filters_by_allowed() -> None:
    out = build_detections(
        1,
        np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
        np.array([0.5, 0.5]),
        np.array([2, 0]),  # car, person
        NAMES,
        allowed=["car", "truck", "bus"],
    )
    assert [d.class_name for d in out] == ["car"]


def test_build_detections_empty() -> None:
    out = build_detections(1, np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int), NAMES)
    assert out == []


def test_fake_detector_satisfies_protocol() -> None:
    class FakeDetector:
        def detect(self, frame_index: int, image: np.ndarray) -> list[Detection]:
            return []

    assert isinstance(FakeDetector(), Detector)
