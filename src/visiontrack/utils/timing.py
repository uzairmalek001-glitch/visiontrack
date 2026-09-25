"""Clocks and performance measurement helpers."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


class Clock(Protocol):
    """Anything with a monotonic ``now()`` in seconds."""

    def now(self) -> float: ...


class MonotonicClock:
    """Wall-clock based monotonic timer (``time.perf_counter``)."""

    def now(self) -> float:
        return time.perf_counter()


class ManualClock:
    """A clock advanced by hand; used by tests to make live sources deterministic."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class RollingRate:
    """Events-per-second over a sliding time window."""

    def __init__(self, window_s: float = 2.0) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._window = window_s
        self._stamps: deque[float] = deque()

    def tick(self, t: float) -> None:
        """Record one event at time ``t`` (seconds, monotonic)."""
        self._stamps.append(t)
        cutoff = t - self._window
        while len(self._stamps) > 2 and self._stamps[0] < cutoff:
            self._stamps.popleft()

    @property
    def rate(self) -> float:
        """Current rate in events/second (0.0 until two events were seen)."""
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


class LatencyStats:
    """Running latency statistics in milliseconds (mean/max exact, p95 over a recent window)."""

    def __init__(self, window: int = 512) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.last_ms = 0.0
        self._recent: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        self.max_ms = max(self.max_ms, ms)
        self._recent.append(ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    @property
    def p95_ms(self) -> float:
        if not self._recent:
            return 0.0
        ordered = sorted(self._recent)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[index]

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class MetricsCollector:
    """Aggregates pipeline performance counters.

    Definitions (see README "Performance considerations"):

    * ``frames_skipped``  - frames intentionally not processed because of ``processing_fps``.
    * ``frames_dropped``  - frames discarded because they were unusable (bad type/shape).
    * ``late_frames``     - processed frames whose processing time exceeded the source frame
      interval, i.e. the pipeline could not keep up in real time. A synchronous loop cannot
      observe frames a camera driver dropped internally, so this is the honest proxy.
    """

    frames_read: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0
    frames_dropped: int = 0
    late_frames: int = 0
    events_total: int = 0
    active_tracks: int = 0
    max_active_tracks: int = 0
    tracker_overflows: int = 0
    first_stream_time: float | None = None
    last_stream_time: float | None = None
    processing_latency: LatencyStats = field(default_factory=LatencyStats)
    capture_latency: LatencyStats = field(default_factory=LatencyStats)
    _processing_rate: RollingRate = field(default_factory=RollingRate)
    _capture_rate: RollingRate = field(default_factory=RollingRate)

    def record_read(self, wall_now: float, capture_latency_s: float | None) -> None:
        self.frames_read += 1
        self._capture_rate.tick(wall_now)
        if capture_latency_s is not None:
            self.capture_latency.add(capture_latency_s * 1000.0)

    def record_processed(
        self,
        wall_now: float,
        latency_s: float,
        stream_time: float,
        active_tracks: int,
        events: int,
        frame_interval_s: float | None,
    ) -> None:
        self.frames_processed += 1
        self._processing_rate.tick(wall_now)
        self.processing_latency.add(latency_s * 1000.0)
        if self.first_stream_time is None:
            self.first_stream_time = stream_time
        self.last_stream_time = stream_time
        self.active_tracks = active_tracks
        self.max_active_tracks = max(self.max_active_tracks, active_tracks)
        self.events_total += events
        if frame_interval_s is not None and latency_s > frame_interval_s:
            self.late_frames += 1

    @property
    def processing_fps(self) -> float:
        return self._processing_rate.rate

    @property
    def capture_fps(self) -> float:
        return self._capture_rate.rate

    @property
    def stream_seconds(self) -> float:
        if self.first_stream_time is None or self.last_stream_time is None:
            return 0.0
        return self.last_stream_time - self.first_stream_time

    @property
    def events_per_minute(self) -> float:
        seconds = self.stream_seconds
        return self.events_total / seconds * 60.0 if seconds > 0 else 0.0

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of all counters."""
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "frames_dropped": self.frames_dropped,
            "late_frames": self.late_frames,
            "capture_fps": round(self.capture_fps, 2),
            "processing_fps": round(self.processing_fps, 2),
            "processing_latency": self.processing_latency.as_dict(),
            "capture_latency": self.capture_latency.as_dict(),
            "active_tracks": self.active_tracks,
            "max_active_tracks": self.max_active_tracks,
            "tracker_overflows": self.tracker_overflows,
            "events_total": self.events_total,
            "events_per_minute": round(self.events_per_minute, 2),
            "stream_seconds": round(self.stream_seconds, 3),
        }
