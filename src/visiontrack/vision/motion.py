"""Classical motion detection (frame differencing / running average / MOG2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import MotionConfig
from ..detection import Detection
from .contours import extract_detections
from .preprocessing import MorphologyFilter, threshold_mask

_MOG2_FOREGROUND = 127
"""MOG2 marks shadows as 127 and foreground as 255; pixels > 127 are foreground."""


@dataclass(slots=True)
class MotionResult:
    """Structured output of one motion-detection step (nothing is printed or logged here).

    Attributes:
        mask: Final binary foreground mask (0/255) after ROI masking and morphology.
        detections: One :class:`Detection` per accepted contour (``source='motion'``).
        foreground_ratio: Fraction of mask pixels that are foreground.
        ready: ``False`` while the background model is still warming up; detections are
            suppressed during warm-up so an unsettled model cannot create false events.
    """

    mask: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    foreground_ratio: float = 0.0
    ready: bool = True

    @property
    def present(self) -> bool:
        """True if at least one motion region was detected."""
        return bool(self.detections)


class MotionDetector:
    """Stateful motion detector operating on preprocessed grayscale frames.

    The ``gray`` array passed to :meth:`detect` may be retained by reference for the next call
    (frame differencing), so callers must not modify it in place afterwards. The
    :class:`~visiontrack.vision.preprocessing.Preprocessor` never does.
    """

    def __init__(self, config: MotionConfig) -> None:
        self._cfg = config
        self._morph = MorphologyFilter(config.morphology)
        self.reset()

    def reset(self) -> None:
        """Forget the background model (e.g. after a resolution change)."""
        self._frames_seen = 0
        self._prev: np.ndarray | None = None
        self._background: np.ndarray | None = None
        self._subtractor = None
        if self._cfg.method == "mog2":
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self._cfg.history,
                varThreshold=self._cfg.var_threshold,
                detectShadows=self._cfg.detect_shadows,
            )

    def detect(
        self, gray: np.ndarray, timestamp: float = 0.0, roi_mask: np.ndarray | None = None
    ) -> MotionResult:
        """Run motion detection on one frame.

        Args:
            gray: Single-channel ``uint8`` frame.
            timestamp: Stream time stamped onto the detections.
            roi_mask: Optional 0/255 mask; motion outside of it is ignored.
        """
        cfg = self._cfg
        raw_mask, model_ready = self._foreground(gray)
        self._frames_seen += 1
        ready = model_ready and self._frames_seen > cfg.warmup_frames

        if roi_mask is not None:
            raw_mask = cv2.bitwise_and(raw_mask, roi_mask)
        mask = self._morph(raw_mask)
        ratio = cv2.countNonZero(mask) / mask.size

        detections: list[Detection] = []
        if ready:
            detections = extract_detections(
                mask,
                min_area=cfg.min_area,
                max_area=cfg.max_area,
                timestamp=timestamp,
                source="motion",
                centroid_mode=cfg.centroid_mode,
            )
        return MotionResult(mask=mask, detections=detections, foreground_ratio=ratio, ready=ready)

    def _foreground(self, gray: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return ``(binary_foreground_mask, model_has_reference)`` for the configured method."""
        cfg = self._cfg
        if cfg.method == "frame_diff":
            if self._prev is None:
                self._prev = gray
                return np.zeros_like(gray), False
            diff = cv2.absdiff(gray, self._prev)
            self._prev = gray
            return threshold_mask(diff, cfg.threshold, cfg.threshold_type), True

        if cfg.method == "running_average":
            if self._background is None:
                self._background = gray.astype(np.float32)
                return np.zeros_like(gray), False
            diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._background))
            cv2.accumulateWeighted(gray, self._background, cfg.background_alpha)
            return threshold_mask(diff, cfg.threshold, cfg.threshold_type), True

        assert self._subtractor is not None  # method == "mog2"
        rate = -1.0 if cfg.learning_rate is None else cfg.learning_rate
        fg = self._subtractor.apply(gray, learningRate=rate)
        return threshold_mask(fg, _MOG2_FOREGROUND, "binary"), True
