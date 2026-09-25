"""Unified detection model shared by every detector (classical or ML)."""

from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[int, int, int, int]
"""Bounding box as ``(x, y, width, height)`` in processed-frame pixels."""

Point = tuple[float, float]
"""A point as ``(x, y)`` in processed-frame pixels (origin top-left, y grows downwards)."""

DETERMINISTIC_SOURCES = frozenset({"motion", "contour"})
"""Sources that are classical CV and therefore must never carry a confidence value."""


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object or region in one frame.

    The tracker and event engine only consume this type, so they do not care whether a
    detection came from motion analysis, contour analysis or an ML model.

    Attributes:
        bbox: ``(x, y, w, h)`` in pixels.
        centroid: ``(cx, cy)`` in pixels.
        source: Origin of the detection, e.g. ``"motion"`` or ``"ml"``.
        timestamp: Stream time of the frame in seconds.
        detection_id: Optional identifier; deterministic detectors leave this ``None``.
        label: Optional class label (ML detectors).
        confidence: Model confidence in ``[0, 1]``. Always ``None`` for deterministic sources:
            classical detectors do not produce calibrated confidences and we never invent them.
    """

    bbox: BBox
    centroid: Point
    source: str
    timestamp: float = 0.0
    detection_id: int | None = None
    label: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _, _, w, h = self.bbox
        if w < 0 or h < 0:
            raise ValueError(f"bbox width/height must be non-negative, got {self.bbox}")
        if self.confidence is not None:
            if self.source in DETERMINISTIC_SOURCES:
                raise ValueError(
                    f"deterministic source {self.source!r} must not carry a confidence value"
                )
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence must be within [0, 1], got {self.confidence}")

    @property
    def area(self) -> int:
        """Bounding-box area in pixels."""
        return self.bbox[2] * self.bbox[3]
