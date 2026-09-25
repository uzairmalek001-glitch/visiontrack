"""Live camera and generic network/device frame source.

Uses the same ``cv2.VideoCapture`` backend as :class:`~visiontrack.capture.video.VideoFileSource`,
so an RTSP/HTTP stream URL works today via ``CameraSource`` without redesigning the pipeline —
only the frame-index-based timestamp of file playback is replaced with wall-clock elapsed time,
since a live source has no fixed frame rate to derive timestamps from.
"""

from __future__ import annotations

import time

import cv2

from ..config import SourceConfig
from ..errors import CaptureError
from ..utils.timing import Clock, MonotonicClock
from .base import Frame, FrameSource


class CameraSource(FrameSource):
    """Reads frames from a webcam index or a stream URI (RTSP/HTTP/etc.)."""

    def __init__(
        self, device: int | str, config: SourceConfig | None = None, clock: Clock | None = None
    ) -> None:
        self._device = device
        self._cfg = config or SourceConfig()
        self._clock = clock or MonotonicClock()
        self._cap: cv2.VideoCapture | None = None
        self._size: tuple[int, int] | None = None
        self._frame_number = 0
        self._start_time: float | None = None
        self._consecutive_failures = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"could not open camera/stream: {self._device}")
        if self._cfg.capture_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.capture_width)
        if self._cfg.capture_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.capture_height)
        if self._cfg.capture_fps:
            cap.set(cv2.CAP_PROP_FPS, self._cfg.capture_fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self._size = (w, h)
        self._cap = cap
        self._frame_number = 0
        self._start_time = None
        self._consecutive_failures = 0

    def read(self) -> Frame | None:
        if self._cap is None:
            raise CaptureError("read() called before open()")
        if self._cfg.max_frames is not None and self._frame_number >= self._cfg.max_frames:
            return None
        t0 = time.perf_counter()
        ok, image = self._cap.read()
        latency = time.perf_counter() - t0
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._cfg.max_read_failures:
                raise CaptureError(
                    f"camera/stream stopped delivering frames after "
                    f"{self._consecutive_failures} consecutive failures: {self._device}"
                )
            return self.read()
        self._consecutive_failures = 0
        now = self._clock.now()
        if self._start_time is None:
            self._start_time = now
        frame = Frame(
            image=image,
            frame_number=self._frame_number,
            timestamp=now - self._start_time,
            capture_latency=latency,
        )
        self._frame_number += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is not None:
            reported = self._cap.get(cv2.CAP_PROP_FPS)
            if reported and reported > 0:
                return reported
        return self._cfg.capture_fps or self._cfg.assumed_fps

    @property
    def frame_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def name(self) -> str:
        return f"camera:{self._device}"
