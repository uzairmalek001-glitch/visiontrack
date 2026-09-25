"""Wires every stage together: capture -> preprocessing -> motion -> ROI -> tracking -> state ->
events -> dispatch. :meth:`Pipeline.process_frame` is a pure-ish step (side effects limited to
event sinks) so it is easy to unit test without a real video source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .capture.base import Frame, FrameSource
from .config import VisionTrackConfig
from .detection import Detection
from .detectors.base import BaseDetector
from .detectors.optional_ml import load_detector
from .errors import FrameError
from .events.engine import EventEngine
from .events.models import Event
from .output.overlay import Overlay
from .regions.roi import Region, build_regions, combined_mask
from .tracking.centroid import CentroidTracker
from .tracking.track import Track
from .utils.timing import MetricsCollector, MonotonicClock
from .vision.motion import MotionDetector, MotionResult
from .vision.preprocessing import Preprocessor

logger = logging.getLogger("visiontrack")


@dataclass(slots=True)
class StepResult:
    """Everything produced by processing one frame — used by the CLI, tests, and replay."""

    frame_number: int
    timestamp: float
    motion: MotionResult
    tracks: list[Track]
    removed_track_ids: list[int]
    events: list[Event]
    color: np.ndarray | None = None


@dataclass
class Pipeline:
    """Owns every stage's state for one video source and processes frames one at a time."""

    config: VisionTrackConfig
    source_name: str
    keep_color: bool = False

    preprocessor: Preprocessor = field(init=False)
    motion_detector: MotionDetector = field(init=False)
    tracker: CentroidTracker = field(init=False)
    regions: dict[str, Region] = field(init=False)
    events: EventEngine = field(init=False)
    ml_detector: BaseDetector | None = field(init=False)
    metrics: MetricsCollector = field(default_factory=MetricsCollector)
    overlay: Overlay | None = field(init=False, default=None)
    _include_mask: np.ndarray | None = field(init=False, default=None)
    _exclude_mask: np.ndarray | None = field(init=False, default=None)
    _mask_size: tuple[int, int] | None = field(init=False, default=None)
    _consecutive_bad_frames: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        cfg = self.config
        self.preprocessor = Preprocessor(cfg.preprocessing, keep_color=self.keep_color)
        self.motion_detector = MotionDetector(cfg.motion)
        self.tracker = CentroidTracker(cfg.tracking)
        self.regions = build_regions(cfg.zones)
        self.events = EventEngine.from_config(cfg, self.regions, self.source_name)
        self.ml_detector = load_detector(cfg.ml)
        if cfg.display.enabled:
            self.overlay = Overlay(cfg.display)

    def add_event_sink(self, sink) -> None:
        """Register an event sink (see :meth:`EventEngine.add_sink`)."""
        self.events.add_sink(sink)

    def _roi_masks(self, size: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
        if size != self._mask_size:
            cfg = self.config.motion
            include = [self.regions[n] for n in cfg.include_zones]
            exclude = [self.regions[n] for n in cfg.exclude_zones]
            self._include_mask = combined_mask(include, size) if include else None
            self._exclude_mask = combined_mask(exclude, size) if exclude else None
            self._mask_size = size
        return self._include_mask, self._exclude_mask

    def _motion_roi_mask(self, size: tuple[int, int]) -> np.ndarray | None:
        include, exclude = self._roi_masks(size)
        if include is None and exclude is None:
            return None
        w, h = size
        mask = include if include is not None else np.full((h, w), 255, dtype=np.uint8)
        if exclude is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))
        return mask

    def process_frame(self, frame: Frame) -> StepResult:
        """Run one frame through every pipeline stage and return the structured result.

        Raises:
            FrameError: propagated from preprocessing for a malformed or inconsistently-sized
                frame; the caller decides whether to count it as a dropped frame and continue.
        """
        pre = self.preprocessor.process(frame.image)
        roi_mask = self._motion_roi_mask(pre.size)
        motion = self.motion_detector.detect(pre.gray, frame.timestamp, roi_mask)

        detections: list[Detection] = []
        if "motion" in self.config.tracking.input_sources:
            detections += motion.detections
        if "ml" in self.config.tracking.input_sources and self.ml_detector is not None:
            ml_image = pre.color if pre.color is not None else cv2.cvtColor(pre.gray, cv2.COLOR_GRAY2BGR)
            detections += self.ml_detector.detect(ml_image, frame.timestamp)

        tracks = self.tracker.update(detections, frame.timestamp, frame.frame_number)
        removed = self.tracker.removed_since_last_update()

        events = self.events.step(frame.frame_number, frame.timestamp, tracks, removed, motion)

        return StepResult(
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            motion=motion,
            tracks=tracks,
            removed_track_ids=removed,
            events=events,
            color=pre.color,
        )

    def run(self, source: FrameSource, on_step=None) -> MetricsCollector:
        """Drive ``source`` to completion, processing every frame.

        Args:
            source: An already-open (or about-to-be-opened) frame source.
            on_step: Optional callback invoked with each :class:`StepResult`.
        """
        clock = MonotonicClock()
        interval = 1.0 / self.config.source.processing_fps if self.config.source.processing_fps else None
        source_interval = 1.0 / source.fps if source.fps else None
        with source:
            last_processed_stream_t: float | None = None
            for frame in source:
                self.metrics.record_read(clock.now(), frame.capture_latency)
                if interval is not None and last_processed_stream_t is not None:
                    if frame.timestamp - last_processed_stream_t < interval:
                        self.metrics.frames_skipped += 1
                        continue
                last_processed_stream_t = frame.timestamp

                t0 = clock.now()
                try:
                    result = self.process_frame(frame)
                except FrameError as exc:
                    self.metrics.frames_dropped += 1
                    self._consecutive_bad_frames += 1
                    logger.warning("dropped frame %d: %s", frame.frame_number, exc)
                    if self._consecutive_bad_frames >= self.config.pipeline.max_consecutive_bad_frames:
                        raise
                    continue
                self._consecutive_bad_frames = 0
                latency = clock.now() - t0
                self.metrics.tracker_overflows = self.tracker.overflow_count
                self.metrics.record_processed(
                    clock.now(), latency, frame.timestamp, len(result.tracks), len(result.events), source_interval
                )
                if on_step is not None:
                    on_step(result)
                if self.overlay is not None and result.color is not None:
                    self.overlay.note_events(result.events, result.timestamp)
                    rendered = self.overlay.render(
                        result.color,
                        mask=result.motion.mask if self.config.display.show_mask else None,
                        regions=self.regions,
                        lines={name: (tuple(l.start), tuple(l.end)) for name, l in self.config.lines.items()},
                        tracks=result.tracks,
                        frame_number=result.frame_number,
                        timestamp=result.timestamp,
                        processing_fps=self.metrics.processing_fps,
                        event_count=self.metrics.events_total,
                    )
                    if not self.overlay.show(rendered):
                        break
        if self.overlay is not None:
            self.overlay.close()
        return self.metrics
