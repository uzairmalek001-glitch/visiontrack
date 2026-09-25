"""Event rules: state machines that turn per-frame state into :class:`EventDraft` objects.

Each rule is independent and stateful (keyed internally by track id / zone / line name), so
noisy frame-level signals become debounced, semantically meaningful events. Rules never emit
duplicate events for unchanged state — that is what makes the cooldown layer in
:mod:`visiontrack.events.engine` a *second*, coarser safety net rather than the only one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import (
    DwellRuleConfig,
    LineConfig,
    LineRuleConfig,
    MotionRuleConfig,
    VelocityRuleConfig,
    ZoneActivityRuleConfig,
    ZoneTransitionRuleConfig,
)
from ..regions.zones import ZoneOccupancyTracker
from ..tracking.track import Track
from ..vision.geometry import signed_distance_to_line
from ..vision.motion import MotionResult
from .models import EventDraft, EventType


@dataclass
class RuleContext:
    """Everything a rule needs for one frame."""

    frame_number: int
    timestamp: float
    tracks: list[Track]
    removed_track_ids: list[int]
    motion: MotionResult | None
    zone_membership: dict[str, dict[int, bool]]  # zone_name -> {track_id: raw_inside}
    all_zone_names: list[str]


class MotionActivityRule:
    """MOTION_STARTED / MOTION_ACTIVE / MOTION_STOPPED from consecutive-frame debouncing."""

    def __init__(self, config: MotionRuleConfig) -> None:
        self._cfg = config
        self._consecutive_motion = 0
        self._consecutive_still = 0
        self._active = False
        self._last_active_emit: float | None = None

    def update(self, ctx: RuleContext) -> list[EventDraft]:
        if not self._cfg.enabled or ctx.motion is None:
            return []
        drafts: list[EventDraft] = []
        present = ctx.motion.present
        if present:
            self._consecutive_motion += 1
            self._consecutive_still = 0
        else:
            self._consecutive_still += 1
            self._consecutive_motion = 0

        if not self._active and self._consecutive_motion >= self._cfg.start_frames:
            self._active = True
            self._last_active_emit = ctx.timestamp
            drafts.append(EventDraft(EventType.MOTION_STARTED))
        elif self._active and present and self._cfg.active_interval_s > 0:
            # Heartbeat only while motion is still actually present this frame - not merely
            # while `_active` is still true because MOTION_STOPPED has not yet debounced.
            if self._last_active_emit is None or (
                ctx.timestamp - self._last_active_emit >= self._cfg.active_interval_s
            ):
                self._last_active_emit = ctx.timestamp
                drafts.append(EventDraft(EventType.MOTION_ACTIVE))

        if self._active and self._consecutive_still >= self._cfg.stop_frames:
            self._active = False
            drafts.append(EventDraft(EventType.MOTION_STOPPED))
        return drafts


class ZoneTransitionRule:
    """OBJECT_ENTERED_ZONE / OBJECT_LEFT_ZONE from debounced zone occupancy."""

    def __init__(self, config: ZoneTransitionRuleConfig, occupancy: ZoneOccupancyTracker) -> None:
        self._cfg = config
        self._occupancy = occupancy

    def update(self, ctx: RuleContext) -> list[EventDraft]:
        if not self._cfg.enabled:
            return []
        drafts: list[EventDraft] = []
        tracks_by_id = {t.track_id: t for t in ctx.tracks}
        for zone_name, memberships in ctx.zone_membership.items():
            for track_id, raw_inside in memberships.items():
                transition = self._occupancy.update(track_id, zone_name, raw_inside, ctx.timestamp)
                if transition is None:
                    continue
                track = tracks_by_id.get(track_id)
                bbox = track.bbox if track else None
                if transition.entered:
                    if track is not None:
                        track.zones.add(zone_name)
                    drafts.append(
                        EventDraft(EventType.OBJECT_ENTERED_ZONE, track_id=track_id, zone=zone_name, bbox=bbox)
                    )
                else:
                    if track is not None:
                        track.zones.discard(zone_name)
                    meta = {}
                    if transition.dwell_seconds is not None:
                        meta["dwell_seconds"] = transition.dwell_seconds
                    drafts.append(
                        EventDraft(
                            EventType.OBJECT_LEFT_ZONE, track_id=track_id, zone=zone_name, bbox=bbox, metadata=meta
                        )
                    )
        return drafts


class DwellRule:
    """OBJECT_DWELL_STARTED / OBJECT_DWELL_ENDED once a track spends >= threshold in a zone."""

    def __init__(self, config: DwellRuleConfig, occupancy: ZoneOccupancyTracker) -> None:
        self._cfg = config
        self._occupancy = occupancy
        self._dwelling: set[tuple[int, str]] = set()

    def _threshold(self, zone_threshold: float | None) -> float:
        return zone_threshold if zone_threshold is not None else self._cfg.threshold_s

    def update(self, ctx: RuleContext, zone_thresholds: dict[str, float | None]) -> list[EventDraft]:
        if not self._cfg.enabled:
            return []
        drafts: list[EventDraft] = []
        tracks_by_id = {t.track_id: t for t in ctx.tracks}

        for track in ctx.tracks:
            for zone_name in list(track.zones):
                key = (track.track_id, zone_name)
                dwell = self._occupancy.dwell_seconds(track.track_id, zone_name, ctx.timestamp)
                threshold = self._threshold(zone_thresholds.get(zone_name))
                if dwell is not None and dwell >= threshold and key not in self._dwelling:
                    self._dwelling.add(key)
                    drafts.append(
                        EventDraft(
                            EventType.OBJECT_DWELL_STARTED,
                            track_id=track.track_id,
                            zone=zone_name,
                            bbox=track.bbox,
                            metadata={"dwell_seconds": dwell},
                        )
                    )

        for key in list(self._dwelling):
            track_id, zone_name = key
            if track_id not in tracks_by_id or zone_name not in tracks_by_id[track_id].zones:
                self._dwelling.discard(key)
                track = tracks_by_id.get(track_id)
                drafts.append(
                    EventDraft(
                        EventType.OBJECT_DWELL_ENDED,
                        track_id=track_id,
                        zone=zone_name,
                        bbox=track.bbox if track else None,
                    )
                )
        for track_id in ctx.removed_track_ids:
            for key in [k for k in self._dwelling if k[0] == track_id]:
                self._dwelling.discard(key)
                drafts.append(EventDraft(EventType.OBJECT_DWELL_ENDED, track_id=key[0], zone=key[1]))
        return drafts


class LineCrossingRule:
    """LINE_CROSSED when a track's centroid changes side of a configured line.

    A ``hysteresis_px`` dead band around the line prevents a centroid that is merely jittering
    on the line from registering repeated crossings.
    """

    def __init__(self, config: LineRuleConfig, lines: dict[str, LineConfig]) -> None:
        self._cfg = config
        self._lines = lines
        self._last_side: dict[tuple[int, str], int] = {}  # -1, 0 (dead band), or +1

    @staticmethod
    def _sign(distance: float, dead_band: float) -> int:
        if distance > dead_band:
            return 1
        if distance < -dead_band:
            return -1
        return 0

    def update(self, ctx: RuleContext) -> list[EventDraft]:
        if not self._cfg.enabled or not self._lines:
            return []
        drafts: list[EventDraft] = []
        for track in ctx.tracks:
            for name, line in self._lines.items():
                key = (track.track_id, name)
                # Perpendicular distance in pixels, so hysteresis_px is a real dead-band width -
                # not the raw (line-length-scaled) cross product.
                d = signed_distance_to_line(track.centroid, tuple(line.start), tuple(line.end))
                side = self._sign(d, self._cfg.hysteresis_px)
                prev = self._last_side.get(key)
                if side != 0:
                    if prev is not None and prev != 0 and prev != side:
                        crossed_positive = side > 0  # entering the positive (right-hand) side
                        wanted = (
                            line.direction == "any"
                            or (line.direction == "positive" and crossed_positive)
                            or (line.direction == "negative" and not crossed_positive)
                        )
                        if wanted:
                            drafts.append(
                                EventDraft(
                                    EventType.LINE_CROSSED,
                                    track_id=track.track_id,
                                    bbox=track.bbox,
                                    metadata={"line": name, "direction": "positive" if crossed_positive else "negative"},
                                )
                            )
                    self._last_side[key] = side
        for tid in ctx.removed_track_ids:
            for key in [k for k in self._last_side if k[0] == tid]:
                del self._last_side[key]
        return drafts


class VelocityRule:
    """VELOCITY_EXCEEDED when a track's smoothed speed passes a pixel/second threshold.

    Pixel units only — VisionTrack performs no camera calibration, so this is never a physical
    (real-world) speed.
    """

    def __init__(self, config: VelocityRuleConfig) -> None:
        self._cfg = config

    def update(self, ctx: RuleContext) -> list[EventDraft]:
        if not self._cfg.enabled:
            return []
        drafts: list[EventDraft] = []
        for track in ctx.tracks:
            if track.matched_observations < self._cfg.min_observations:
                continue
            if track.speed >= self._cfg.threshold_px_s:
                drafts.append(
                    EventDraft(
                        EventType.VELOCITY_EXCEEDED,
                        track_id=track.track_id,
                        bbox=track.bbox,
                        metadata={"speed_px_s": track.speed, "direction": track.direction},
                    )
                )
        return drafts


class ZoneActivityRule:
    """ZONE_ACTIVATED / ZONE_DEACTIVATED when a zone gains/loses all occupants."""

    def __init__(self, config: ZoneActivityRuleConfig) -> None:
        self._cfg = config
        self._active: set[str] = set()
        self._empty_since: dict[str, float] = {}

    def update(self, ctx: RuleContext) -> list[EventDraft]:
        if not self._cfg.enabled:
            return []
        drafts: list[EventDraft] = []
        occupied: dict[str, int] = {name: 0 for name in ctx.all_zone_names}
        for track in ctx.tracks:
            for zone_name in track.zones:
                occupied[zone_name] = occupied.get(zone_name, 0) + 1

        for zone_name in ctx.all_zone_names:
            count = occupied.get(zone_name, 0)
            if count > 0:
                self._empty_since.pop(zone_name, None)
                if zone_name not in self._active:
                    self._active.add(zone_name)
                    drafts.append(EventDraft(EventType.ZONE_ACTIVATED, zone=zone_name))
            elif zone_name in self._active:
                start = self._empty_since.setdefault(zone_name, ctx.timestamp)
                if ctx.timestamp - start >= self._cfg.inactive_after_s:
                    self._active.discard(zone_name)
                    self._empty_since.pop(zone_name, None)
                    drafts.append(EventDraft(EventType.ZONE_DEACTIVATED, zone=zone_name))
        return drafts
