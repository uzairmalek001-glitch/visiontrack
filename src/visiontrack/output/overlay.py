"""Optional OpenCV visualization overlay. Never required for the pipeline to run."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import DisplayConfig
from ..events.models import Event
from ..regions.roi import Region
from ..tracking.track import Track

_TRACK_COLOR = (0, 255, 0)
_ZONE_COLOR = (255, 200, 0)
_TEXT_COLOR = (255, 255, 255)
_EVENT_COLOR = (0, 220, 255)
_LINE_COLOR = (0, 120, 255)


@dataclass(slots=True)
class _ShownEvent:
    text: str
    expires_at: float


class Overlay:
    """Draws tracks, zones, lines, and recent events onto a BGR frame for display."""

    def __init__(self, config: DisplayConfig) -> None:
        self._cfg = config
        self._recent: deque[_ShownEvent] = deque(maxlen=max(config.max_events_shown, 1))

    def note_events(self, events: list[Event], timestamp: float) -> None:
        for event in events:
            label = str(event.event_type)
            if event.track_id is not None:
                label += f" track={event.track_id}"
            if event.zone is not None:
                label += f" zone={event.zone}"
            self._recent.append(_ShownEvent(label, timestamp + self._cfg.event_ttl_s))

    def render(
        self,
        color: np.ndarray,
        *,
        mask: np.ndarray | None,
        regions: dict[str, Region],
        lines: dict[str, tuple[tuple[float, float], tuple[float, float]]],
        tracks: list[Track],
        frame_number: int,
        timestamp: float,
        processing_fps: float,
        event_count: int,
    ) -> np.ndarray:
        image = color.copy()

        if self._cfg.show_zones:
            for region in regions.values():
                region.draw(image, _ZONE_COLOR)
                x, y = (int(v) for v in region.points[0])
                cv2.putText(image, region.name, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _ZONE_COLOR, 1)
            for name, (start, end) in lines.items():
                p1 = (int(start[0]), int(start[1]))
                p2 = (int(end[0]), int(end[1]))
                cv2.line(image, p1, p2, _LINE_COLOR, 2)
                cv2.putText(image, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, _LINE_COLOR, 1)

        if self._cfg.show_tracks:
            for track in tracks:
                x, y, w, h = (int(v) for v in track.bbox)
                cv2.rectangle(image, (x, y), (x + w, y + h), _TRACK_COLOR, 2)
                cx, cy = (int(v) for v in track.centroid)
                cv2.circle(image, (cx, cy), 3, _TRACK_COLOR, -1)
                label = f"#{track.track_id}"
                if self._cfg.show_velocity:
                    label += f" {track.speed:.0f}px/s {track.direction}"
                cv2.putText(image, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TRACK_COLOR, 1)

        hud = [
            f"frame={frame_number} t={timestamp:.2f}s fps={processing_fps:.1f}",
            f"tracks={len(tracks)} events={event_count}",
        ]
        if self._cfg.show_mask:
            hud.append("mask=on")
        for i, line in enumerate(hud):
            cv2.putText(image, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1)

        y0 = 20 + 18 * len(hud) + 10
        active = [e for e in self._recent if e.expires_at >= timestamp]
        for i, shown in enumerate(active):
            cv2.putText(image, shown.text, (8, y0 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _EVENT_COLOR, 1)

        if self._cfg.show_mask and mask is not None:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            h = image.shape[0] // 4
            w = int(mask.shape[1] * h / mask.shape[0])
            thumb = cv2.resize(mask_bgr, (w, h))
            image[0:h, image.shape[1] - w : image.shape[1]] = thumb

        return image

    def show(self, image: np.ndarray) -> bool:
        """Display one frame; returns ``False`` if the user requested to quit (pressed 'q')."""
        cv2.imshow(self._cfg.window_name, image)
        key = cv2.waitKey(self._cfg.wait_ms) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        """Destroy the display window, if one was ever created."""
        try:
            cv2.destroyWindow(self._cfg.window_name)
        except cv2.error:
            pass
