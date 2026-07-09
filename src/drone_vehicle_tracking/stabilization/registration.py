"""Frame-to-frame registration on the ground plane (OpenCV, ``[cv]`` extra).

Consecutive nadir frames are related by a near-similarity transform of the
ground plane. It is estimated from sparse features: Shi-Tomasi corners tracked
with pyramidal Lucas-Kanade optical flow, then a RANSAC similarity fit.
Moving vehicles violate the ground-plane motion and are rejected as RANSAC
outliers -- the static background dominates the correspondences. Any failure
(too few corners, lost tracking, degenerate fit) returns ``None`` so the
caller can split the flight into independent fusion segments instead of
propagating a bad measurement.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from drone_vehicle_tracking.geo.fusion import FrameTransform

_MAX_CORNERS = 400
_QUALITY_LEVEL = 0.01
_MIN_DISTANCE_PX = 20
_MIN_POINTS = 12
_RANSAC_THRESHOLD_PX = 3.0


def to_gray(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a single-channel view of ``image`` (BGR frames are converted)."""
    if image.ndim == 2:
        return image
    import cv2

    return cast("npt.NDArray[np.uint8]", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def estimate_transform(
    prev_gray: npt.NDArray[np.uint8], curr_gray: npt.NDArray[np.uint8]
) -> FrameTransform | None:
    """Estimate the pixel similarity transform from ``prev_gray`` to ``curr_gray``.

    Returns ``None`` when the registration is unreliable (textureless imagery,
    lost feature tracks or a degenerate RANSAC fit).
    """
    import cv2

    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=_MAX_CORNERS,
        qualityLevel=_QUALITY_LEVEL,
        minDistance=_MIN_DISTANCE_PX,
    )
    if corners is None or len(corners) < _MIN_POINTS:
        return None
    moved, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
    tracked = status.reshape(-1) == 1
    src = corners.reshape(-1, 2)[tracked]
    dst = moved.reshape(-1, 2)[tracked]
    if len(src) < _MIN_POINTS:
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_THRESHOLD_PX
    )
    if matrix is None:
        return None
    flat = [float(value) for value in np.asarray(matrix, dtype=np.float64).reshape(-1)]
    return FrameTransform(matrix=(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5]))
