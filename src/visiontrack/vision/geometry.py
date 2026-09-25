"""Pure geometry helpers (points, boxes, segments, polygons).

Conventions
-----------
* Coordinates are pixels, origin top-left, ``y`` grows downwards.
* Boxes are ``(x, y, w, h)``. Boxes and polygons are treated as *closed* sets: touching edges
  count as intersecting.
* ``side_of_line`` returns the sign of the 2-D cross product ``(b - a) x (p - a)`` in image
  coordinates. A positive value means ``p`` lies on the *right-hand side* of the directed line
  ``a -> b`` as seen on screen (e.g. for a line pointing right, positive is *below* it).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from ..detection import BBox, Point

PointLike = Sequence[float]


def bbox_centroid(bbox: BBox) -> Point:
    """Geometric centre of a bounding box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def bbox_area(bbox: BBox) -> int:
    """Area of a bounding box in pixels."""
    return max(0, bbox[2]) * max(0, bbox[3])


def bbox_corners(bbox: BBox) -> list[Point]:
    """Corners in clockwise order starting at the top-left."""
    x, y, w, h = bbox
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def bbox_intersects(a: BBox, b: BBox) -> bool:
    """True if two boxes overlap or touch."""
    return (
        a[0] <= b[0] + b[2]
        and b[0] <= a[0] + a[2]
        and a[1] <= b[1] + b[3]
        and b[1] <= a[1] + a[3]
    )


def bbox_intersection_area(a: BBox, b: BBox) -> int:
    """Area of the overlap of two boxes (0 if disjoint or merely touching)."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return ix * iy if ix > 0 and iy > 0 else 0


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    inter = bbox_intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_union(boxes: Sequence[BBox]) -> BBox | None:
    """Smallest box enclosing all ``boxes`` (``None`` for an empty sequence)."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def distance(p: PointLike, q: PointLike) -> float:
    """Euclidean distance."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def polygon_array(points: Sequence[PointLike]) -> np.ndarray:
    """Convert points to the ``float32`` ``(N, 1, 2)`` layout OpenCV geometry functions expect."""
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def point_in_polygon(point: PointLike, polygon: np.ndarray) -> bool:
    """True if ``point`` is inside or on the boundary of ``polygon`` (via ``cv2.pointPolygonTest``)."""
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def polygon_area(points: Sequence[PointLike]) -> float:
    """Absolute polygon area (shoelace formula)."""
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def cross(o: PointLike, a: PointLike, b: PointLike) -> float:
    """2-D cross product of ``(a - o)`` and ``(b - o)``."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def side_of_line(p: PointLike, a: PointLike, b: PointLike) -> float:
    """Signed side of ``p`` relative to the directed line ``a -> b`` (see module docstring)."""
    return cross(a, b, p)


def signed_distance_to_line(p: PointLike, a: PointLike, b: PointLike) -> float:
    """Signed perpendicular distance from ``p`` to the infinite line through ``a`` and ``b``."""
    length = distance(a, b)
    if length == 0:
        raise ValueError("line endpoints must differ")
    return side_of_line(p, a, b) / length


def _on_segment(a: PointLike, b: PointLike, p: PointLike) -> bool:
    """Given collinear points, is ``p`` within the bounding box of segment ``ab``?"""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(p1: PointLike, p2: PointLike, q1: PointLike, q2: PointLike) -> bool:
    """True if closed segments ``p1p2`` and ``q1q2`` share at least one point."""
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        (d1 == 0 and _on_segment(q1, q2, p1))
        or (d2 == 0 and _on_segment(q1, q2, p2))
        or (d3 == 0 and _on_segment(p1, p2, q1))
        or (d4 == 0 and _on_segment(p1, p2, q2))
    )


def is_simple_polygon(points: Sequence[PointLike]) -> bool:
    """True if no two non-adjacent edges of the polygon intersect (no self-intersections)."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # adjacent edges share a vertex by definition
            b1, b2 = points[j], points[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True


def bbox_intersects_polygon(bbox: BBox, points: Sequence[PointLike], polygon: np.ndarray) -> bool:
    """Exact box/polygon overlap test for convex *and* concave polygons.

    They intersect if a box corner is inside the polygon, a polygon vertex is inside the box,
    or any box edge crosses any polygon edge.
    """
    corners = bbox_corners(bbox)
    if any(point_in_polygon(c, polygon) for c in corners):
        return True
    x, y, w, h = bbox
    if any(x <= px <= x + w and y <= py <= y + h for px, py in points):
        return True
    n = len(points)
    for i in range(4):
        c1, c2 = corners[i], corners[(i + 1) % 4]
        for j in range(n):
            if segments_intersect(c1, c2, points[j], points[(j + 1) % n]):
                return True
    return False


def heading_degrees(vx: float, vy: float) -> float | None:
    """Direction of a velocity vector in degrees in ``[0, 360)``.

    0 = right (+x), 90 = down (+y, clockwise on screen). ``None`` for a zero vector.
    """
    if vx == 0 and vy == 0:
        return None
    return math.degrees(math.atan2(vy, vx)) % 360.0


def cardinal_direction(vx: float, vy: float, min_speed: float = 0.0) -> str:
    """Coarse direction label: ``up``, ``down``, ``left``, ``right`` or ``stationary``."""
    if math.hypot(vx, vy) <= min_speed or (vx == 0 and vy == 0):
        return "stationary"
    if abs(vx) >= abs(vy):
        return "right" if vx > 0 else "left"
    return "down" if vy > 0 else "up"
