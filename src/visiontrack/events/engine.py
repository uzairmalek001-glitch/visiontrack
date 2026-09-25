"""Event engine: turns rule output into stamped, cooldown-filtered :class:`Event` objects and
dispatches them to sinks (console logger, JSONL writer, overlay, ...).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from ..config import EventsConfig, LineConfig, VisionTrackConfig, ZoneConfig
from ..regions.roi import Region
from ..regions.zones import ZoneOccupancyTracker
from ..tracking.track import Track
from ..utils.ids import IdGenerator, format_event_id
from ..vision.motion import MotionResult
from .models import Event, EventDraft
from .rules import (
    DwellRule,
    LineCrossingRule,
    MotionActivityRule,
    RuleContext,
    VelocityRule,
    ZoneActivityRule,
    ZoneTransitionRule,
)

EventSink = Callable[[Event], None]


def _cooldown_key(draft: EventDraft) -> tuple[str, int | None, str | None]:
    return (str(draft.event_type), draft.track_id, draft.zone or draft.metadata.get("line"))


class EventEngine:
    """Owns every rule, applies cooldown debouncing, and stamps drafts into full events.

    Args:
        events_config: Event-rule configuration (thresholds, cooldowns).
        regions: Built zone geometry, used to compute per-frame zone membership.
        zone_configs: Raw zone configs, used for their per-zone ``dwell_seconds`` override.
        lines: Line-crossing configuration.
        source_name: Value stamped onto every emitted event's ``source`` field.
        wall_clock_timestamps: If true, also stamp ISO-8601 wall-clock time (non-reproducible).
    """

    def __init__(
        self,
        events_config: EventsConfig,
        regions: dict[str, Region],
        zone_configs: dict[str, ZoneConfig],
        lines: dict[str, LineConfig],
        source_name: str,
        wall_clock_timestamps: bool = False,
    ) -> None:
        self._cfg = events_config
        self._regions = regions
        self._zone_configs = zone_configs
        self._lines = lines
        self._source_name = source_name
        self._wall_clock = wall_clock_timestamps
        self._ids = IdGenerator()
        self._last_fired: dict[tuple[str, int | None, str | None], float] = {}
        self._occupancy = ZoneOccupancyTracker(
            events_config.zone_transitions.enter_frames, events_config.zone_transitions.exit_frames
        )
        self._motion_rule = MotionActivityRule(events_config.motion)
        self._zone_rule = ZoneTransitionRule(events_config.zone_transitions, self._occupancy)
        self._dwell_rule = DwellRule(events_config.dwell, self._occupancy)
        self._line_rule = LineCrossingRule(events_config.line_crossing, lines)
        self._velocity_rule = VelocityRule(events_config.velocity)
        self._activity_rule = ZoneActivityRule(events_config.zone_activity)
        self._sinks: list[EventSink] = []

    @classmethod
    def from_config(
        cls, config: VisionTrackConfig, regions: dict[str, Region], source_name: str
    ) -> EventEngine:
        return cls(
            config.events,
            regions,
            config.zones,
            config.lines,
            source_name,
            config.logging.wall_clock_timestamps,
        )

    def add_sink(self, sink: EventSink) -> None:
        """Register a callable invoked once per emitted event, in emission order."""
        self._sinks.append(sink)

    def _zone_membership(self, tracks: list[Track]) -> dict[str, dict[int, bool]]:
        membership: dict[str, dict[int, bool]] = {name: {} for name in self._regions}
        for track in tracks:
            for name, region in self._regions.items():
                membership[name][track.track_id] = region.contains(track.centroid, track.bbox)
        return membership

    def step(
        self,
        frame_number: int,
        timestamp: float,
        tracks: list[Track],
        removed_track_ids: list[int],
        motion: MotionResult | None,
    ) -> list[Event]:
        """Run every rule for one frame, apply cooldowns, and dispatch the resulting events."""
        ctx = RuleContext(
            frame_number=frame_number,
            timestamp=timestamp,
            tracks=tracks,
            removed_track_ids=removed_track_ids,
            motion=motion,
            zone_membership=self._zone_membership(tracks),
            all_zone_names=list(self._regions),
        )
        zone_thresholds = {name: cfg.dwell_seconds for name, cfg in self._zone_configs.items()}

        drafts: list[EventDraft] = []
        drafts += self._motion_rule.update(ctx)
        drafts += self._zone_rule.update(ctx)
        drafts += self._dwell_rule.update(ctx, zone_thresholds)
        drafts += self._line_rule.update(ctx)
        drafts += self._velocity_rule.update(ctx)
        drafts += self._activity_rule.update(ctx)

        events: list[Event] = []
        for draft in drafts:
            cooldown = self._cfg.cooldowns.get(str(draft.event_type), 0.0)
            key = _cooldown_key(draft)
            if cooldown > 0:
                last = self._last_fired.get(key)
                if last is not None and timestamp - last < cooldown:
                    continue
                self._last_fired[key] = timestamp
            event = Event(
                event_id=format_event_id(self._ids.next()),
                event_type=draft.event_type,
                timestamp=timestamp,
                frame_number=frame_number,
                source=self._source_name,
                track_id=draft.track_id,
                zone=draft.zone,
                bbox=draft.bbox,
                metadata=draft.metadata,
                wall_time=datetime.now(UTC).isoformat() if self._wall_clock else None,
            )
            events.append(event)
            for sink in self._sinks:
                sink(event)

        for tid in removed_track_ids:
            self._occupancy.forget_track(tid)  # bound memory: stop tracking zone state for dead tracks

        return events

    def occupied_zones(self, track_id: int) -> list[str]:
        """Zones the given track is currently confirmed inside."""
        return self._occupancy.zones_for_track(track_id)
