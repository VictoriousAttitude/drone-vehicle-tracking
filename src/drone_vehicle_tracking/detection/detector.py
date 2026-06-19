"""Per-frame vehicle detection via Ultralytics YOLO.

The Ultralytics/torch import is deferred to construction so the pure parsing
logic (``build_detections``) stays importable and unit-testable without the
heavy CV/DL stack.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.telemetry.models import Detection


def build_detections(
    frame_index: int,
    xyxy: npt.NDArray[np.float64],
    confidences: npt.NDArray[np.float64],
    class_ids: npt.NDArray[np.int_],
    names: Mapping[int, str],
    allowed: Iterable[str] | None = None,
) -> list[Detection]:
    """Convert raw YOLO outputs into ``Detection`` contracts.

    Args:
        frame_index: Index of the frame these detections belong to.
        xyxy: ``(N, 4)`` array of pixel bounding boxes.
        confidences: ``(N,)`` array of confidences.
        class_ids: ``(N,)`` array of integer class ids.
        names: Mapping from class id to class name (e.g. ``model.names``).
        allowed: Optional whitelist of class names to keep.
    """
    allow = set(allowed) if allowed is not None else None
    detections: list[Detection] = []
    for box, conf, cid in zip(xyxy, confidences, class_ids, strict=True):
        name = names[int(cid)]
        if allow is not None and name not in allow:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append(
            Detection(
                frame_index=frame_index,
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=float(conf),
                class_name=name,
            )
        )
    return detections


class YoloVehicleDetector:
    """Ultralytics YOLO wrapped to emit ``Detection`` contracts for vehicle classes."""

    def __init__(self, model: str, conf: float, classes: Sequence[str], imgsz: int = 1280) -> None:
        from ultralytics import YOLO

        self._model: Any = YOLO(model)
        self._conf = conf
        self._classes = list(classes)
        self._imgsz = imgsz

    def detect(self, frame_index: int, image: npt.NDArray[np.uint8]) -> list[Detection]:
        result = self._model.predict(image, conf=self._conf, imgsz=self._imgsz, verbose=False)[0]
        boxes = result.boxes
        return build_detections(
            frame_index,
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            result.names,
            allowed=self._classes,
        )
