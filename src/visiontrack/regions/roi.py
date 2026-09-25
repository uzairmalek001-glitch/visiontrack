"""Static region-of-interest geometry: polygons and rectangles."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import ZoneConfig
from ..detection import BBox, Point
from ..vision.geometry import (
    bbox_intersects,
    bbox_intersects_polygon,
    point_in_polygon,
    polygon_array,
)


@dataclass(slots=True)
class Region:
    """A named region built from a validated :class:`~visiontrack.config.ZoneConfig`.

    Rectangles are stored as an explicit 4-point polygon too, so membership/intersection tests
    have one code path regardless of how the zone was declared.
    """

    name: str
    points: list[Point]
    mode: str  # "centroid" or "bbox"
    _polygon: np.ndarray

    @classmethod
    def from_config(cls, name: str, config: ZoneConfig) -> Region:
        if config.rect is not None:
            x, y, w, h = config.rect
            points: list[Point] = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        else:
            assert config.polygon is not None
            points = [(p[0], p[1]) for p in config.polygon]
        return cls(name=name, points=points, mode=config.mode, _polygon=polygon_array(points))

    def contains_point(self, point: Point) -> bool:
        """True if ``point`` is inside or on the boundary of the region."""
        return point_in_polygon(point, self._polygon)

    def contains_bbox(self, bbox: BBox) -> bool:
        """True if ``bbox`` overlaps the region (used when ``mode == 'bbox'``)."""
        return bbox_intersects_polygon(bbox, self.points, self._polygon)

    def contains(self, centroid: Point, bbox: BBox) -> bool:
        """Membership test using this region's configured ``mode``."""
        if self.mode == "bbox":
            return self.contains_bbox(bbox)
        return self.contains_point(centroid)

    def draw(self, image: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
        """Draw the region outline onto a BGR image (used by the overlay only)."""
        pts = self._polygon.astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    def mask(self, size: tuple[int, int]) -> np.ndarray:
        """Render a 0/255 mask of this region at frame ``(width, height)`` — for ROI-limited motion."""
        w, h = size
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [self._polygon.astype(np.int32)], 255)
        return m


def build_regions(zones: dict[str, ZoneConfig]) -> dict[str, Region]:
    """Build all :class:`Region` objects from validated zone configs, preserving order."""
    return {name: Region.from_config(name, cfg) for name, cfg in zones.items()}


def combined_mask(regions: list[Region], size: tuple[int, int]) -> np.ndarray:
    """Union mask of several regions (used for ``motion.include_zones``)."""
    w, h = size
    out = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        out = cv2.bitwise_or(out, region.mask(size))
    return out
