"""Tracker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..detection import Detection
from .track import Track


class BaseTracker(ABC):
    """Associates per-frame detections with persistent tracks."""

    @abstractmethod
    def update(self, detections: list[Detection], timestamp: float, frame_number: int) -> list[Track]:
        """Advance the tracker by one frame and return all currently active tracks."""

    @abstractmethod
    def removed_since_last_update(self) -> list[int]:
        """Track IDs removed (went stale) during the most recent :meth:`update` call."""
