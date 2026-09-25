"""Contour extraction: binary mask -> structured :class:`Detection` objects."""

from __future__ import annotations

import cv2
import numpy as np

from ..detection import BBox, Detection, Point
from .geometry import bbox_centroid


def contour_centroid(contour: np.ndarray, bbox: BBox, mode: str = "bbox_center") -> Point:
    """Centroid of a contour.

    ``bbox_center`` is the default because it is stable for tracking (it does not wobble when
    the silhouette changes). ``moments`` uses the contour's area centroid and falls back to the
    box centre for degenerate (zero-area) contours.
    """
    if mode == "moments":
        m = cv2.moments(contour)
        if m["m00"] > 0:
            return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    return bbox_centroid(bbox)


def extract_detections(
    mask: np.ndarray,
    *,
    min_area: float,
    max_area: float | None = None,
    timestamp: float = 0.0,
    source: str = "motion",
    centroid_mode: str = "bbox_center",
) -> list[Detection]:
    """Find external contours in ``mask`` and convert them to detections.

    Contours whose area (``cv2.contourArea``) is below ``min_area`` or above ``max_area`` are
    discarded. Detections carry ``confidence=None`` because classical CV produces none. Results
    are sorted by bounding box so output order never depends on OpenCV's traversal order.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or (max_area is not None and area > max_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bbox: BBox = (int(x), int(y), int(w), int(h))
        detections.append(
            Detection(
                bbox=bbox,
                centroid=contour_centroid(contour, bbox, centroid_mode),
                source=source,
                timestamp=timestamp,
            )
        )
    detections.sort(key=lambda d: d.bbox)
    return detections
