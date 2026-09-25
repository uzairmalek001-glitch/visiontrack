"""Deterministic centroid tracker: greedy nearest-centroid association.

Algorithm (per frame):
  1. Compute the Euclidean distance between every existing track and every new detection.
  2. Repeatedly pick the globally smallest remaining distance and match that pair, provided the
     distance is within ``max_distance`` and both ends are still unmatched — this greedy choice
     is deterministic because ties are broken by ``(track_id, detection_index)``.
  3. Unmatched existing tracks get ``disappeared_frames += 1``; tracks over ``max_disappeared``
     are removed.
  4. Unmatched detections become new tracks, unless ``max_tracks`` would be exceeded.
"""

from __future__ import annotations

from ..config import TrackingConfig
from ..detection import Detection
from ..vision.geometry import distance
from .base import BaseTracker
from .track import Track


class CentroidTracker(BaseTracker):
    """See module docstring for the association algorithm."""

    def __init__(self, config: TrackingConfig) -> None:
        self._cfg = config
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._removed: list[int] = []
        self.overflow_count = 0

    @property
    def tracks(self) -> dict[int, Track]:
        """All currently active tracks, keyed by ``track_id``."""
        return self._tracks

    def removed_since_last_update(self) -> list[int]:
        return list(self._removed)

    def update(self, detections: list[Detection], timestamp: float, frame_number: int) -> list[Track]:
        cfg = self._cfg
        self._removed = []

        detections = [d for d in detections if d.area >= cfg.minimum_track_area]

        track_ids = sorted(self._tracks)
        candidates: list[tuple[float, int, int]] = []
        for tid in track_ids:
            track = self._tracks[tid]
            for di, det in enumerate(detections):
                d = distance(track.centroid, det.centroid)
                if d <= cfg.max_distance:
                    candidates.append((d, tid, di))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for d, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            self._tracks[tid].apply_match(detections[di], timestamp, frame_number, cfg.velocity_smoothing)
            matched_tracks.add(tid)
            matched_dets.add(di)

        for tid in track_ids:
            if tid not in matched_tracks:
                track = self._tracks[tid]
                track.mark_disappeared()
                if track.disappeared_frames > cfg.max_disappeared:
                    del self._tracks[tid]
                    self._removed.append(tid)

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            if len(self._tracks) >= cfg.max_tracks:
                self.overflow_count += 1
                continue
            track = Track(
                track_id=self._next_id,
                centroid=det.centroid,
                bbox=det.bbox,
                first_seen=timestamp,
                last_seen=timestamp,
                last_seen_frame=frame_number,
                label=det.label,
                source=det.source,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1

        return list(self._tracks.values())
