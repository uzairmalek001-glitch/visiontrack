import unittest

from visiontrack.config import ZoneConfig, config_from_dict
from visiontrack.detection import Detection
from visiontrack.events.engine import EventEngine
from visiontrack.events.models import EventType
from visiontrack.regions.roi import build_regions
from visiontrack.tracking.track import Track
from visiontrack.vision.motion import MotionResult
import numpy as np


def make_motion(present: bool, ready: bool = True) -> MotionResult:
    mask = np.zeros((10, 10), dtype=np.uint8)
    detections = [Detection(bbox=(0, 0, 5, 5), centroid=(2.5, 2.5), source="motion")] if present else []
    return MotionResult(mask=mask, detections=detections, ready=ready)


def make_track(track_id, centroid, matched_observations=5, velocity=(0.0, 0.0)):
    return Track(
        track_id=track_id,
        centroid=centroid,
        bbox=(int(centroid[0] - 5), int(centroid[1] - 5), 10, 10),
        first_seen=0.0,
        last_seen=0.0,
        last_seen_frame=0,
        matched_observations=matched_observations,
        velocity=velocity,
    )


class MotionEventTests(unittest.TestCase):
    def _engine(self, **overrides):
        cfg = config_from_dict({"events": {"motion": {"start_frames": 2, "stop_frames": 2, **overrides}}})
        return EventEngine.from_config(cfg, {}, "test")

    def test_motion_started_after_start_frames(self):
        engine = self._engine()
        e1 = engine.step(0, 0.0, [], [], make_motion(True))
        self.assertEqual(e1, [])
        e2 = engine.step(1, 1.0, [], [], make_motion(True))
        self.assertEqual([ev.event_type for ev in e2], [EventType.MOTION_STARTED])

    def test_motion_stopped_after_stop_frames(self):
        engine = self._engine()
        engine.step(0, 0.0, [], [], make_motion(True))
        engine.step(1, 1.0, [], [], make_motion(True))  # started
        engine.step(2, 2.0, [], [], make_motion(False))
        e = engine.step(3, 3.0, [], [], make_motion(False))
        self.assertEqual([ev.event_type for ev in e], [EventType.MOTION_STOPPED])

    def test_motion_active_repeats_on_interval(self):
        cfg = config_from_dict(
            {"events": {"motion": {"start_frames": 1, "stop_frames": 100, "active_interval_s": 1.0}}}
        )
        engine = EventEngine.from_config(cfg, {}, "test")
        e0 = engine.step(0, 0.0, [], [], make_motion(True))
        self.assertEqual([ev.event_type for ev in e0], [EventType.MOTION_STARTED])
        e1 = engine.step(1, 0.5, [], [], make_motion(True))  # too soon for MOTION_ACTIVE
        self.assertEqual(e1, [])
        e2 = engine.step(2, 1.2, [], [], make_motion(True))  # interval elapsed
        self.assertEqual([ev.event_type for ev in e2], [EventType.MOTION_ACTIVE])

    def test_no_events_when_motion_result_not_ready(self):
        engine = self._engine()
        e = engine.step(0, 0.0, [], [], make_motion(True, ready=False))
        self.assertEqual(e, [])


class ZoneEventTests(unittest.TestCase):
    def _engine(self):
        cfg = config_from_dict(
            {
                "zones": {"a": {"rect": [0, 0, 100, 100]}},
                "events": {"zone_transitions": {"enter_frames": 2, "exit_frames": 2}},
            }
        )
        regions = build_regions(cfg.zones)
        return EventEngine.from_config(cfg, regions, "test")

    def test_entered_zone_event_after_debounce(self):
        engine = self._engine()
        track = make_track(1, (50, 50))
        engine.step(0, 0.0, [track], [], None)
        events = engine.step(1, 1.0, [track], [], None)
        # ZONE_ACTIVATED also fires here (the zone gained its first occupant this frame).
        types = [e.event_type for e in events]
        self.assertIn(EventType.OBJECT_ENTERED_ZONE, types)
        entered = next(e for e in events if e.event_type == EventType.OBJECT_ENTERED_ZONE)
        self.assertEqual(entered.zone, "a")
        self.assertEqual(entered.track_id, 1)

    def test_left_zone_event_after_debounce(self):
        engine = self._engine()
        inside = make_track(1, (50, 50))
        engine.step(0, 0.0, [inside], [], None)
        engine.step(1, 1.0, [inside], [], None)  # entered
        outside = make_track(1, (500, 500))
        engine.step(2, 2.0, [outside], [], None)
        events = engine.step(3, 3.0, [outside], [], None)
        self.assertEqual([e.event_type for e in events], [EventType.OBJECT_LEFT_ZONE])

    def test_zone_activated_and_deactivated(self):
        cfg = config_from_dict(
            {
                "zones": {"a": {"rect": [0, 0, 100, 100]}},
                "events": {
                    "zone_transitions": {"enter_frames": 1, "exit_frames": 1},
                    "zone_activity": {"inactive_after_s": 1.0},
                },
            }
        )
        regions = build_regions(cfg.zones)
        engine = EventEngine.from_config(cfg, regions, "test")
        inside = make_track(1, (50, 50))
        events = engine.step(0, 0.0, [inside], [], None)
        self.assertIn(EventType.ZONE_ACTIVATED, [e.event_type for e in events])

        outside = make_track(1, (500, 500))
        engine.step(1, 1.0, [outside], [], None)  # leaves, zone now empty
        events2 = engine.step(2, 3.0, [outside], [], None)  # 2s later, past inactive_after_s
        self.assertIn(EventType.ZONE_DEACTIVATED, [e.event_type for e in events2])


class DwellEventTests(unittest.TestCase):
    def test_dwell_started_after_threshold_and_ended_on_exit(self):
        cfg = config_from_dict(
            {
                "zones": {"a": {"rect": [0, 0, 100, 100]}},
                "events": {
                    "zone_transitions": {"enter_frames": 1, "exit_frames": 1},
                    "dwell": {"threshold_s": 2.0},
                },
            }
        )
        regions = build_regions(cfg.zones)
        engine = EventEngine.from_config(cfg, regions, "test")
        inside = make_track(1, (50, 50))
        engine.step(0, 0.0, [inside], [], None)  # entered at t=0
        e1 = engine.step(1, 1.0, [inside], [], None)  # dwell 1s, below threshold
        self.assertNotIn(EventType.OBJECT_DWELL_STARTED, [e.event_type for e in e1])
        e2 = engine.step(2, 2.5, [inside], [], None)  # dwell 2.5s, crosses threshold
        self.assertIn(EventType.OBJECT_DWELL_STARTED, [e.event_type for e in e2])

        outside = make_track(1, (500, 500))
        # exit_frames=1: the zone-exit (and therefore dwell-end) confirms on the first outside frame.
        e3 = engine.step(3, 3.0, [outside], [], None)
        self.assertIn(EventType.OBJECT_DWELL_ENDED, [e.event_type for e in e3])

    def test_zone_specific_dwell_override(self):
        cfg = config_from_dict(
            {
                "zones": {"a": {"rect": [0, 0, 100, 100], "dwell_seconds": 0.5}},
                "events": {
                    "zone_transitions": {"enter_frames": 1, "exit_frames": 1},
                    "dwell": {"threshold_s": 100.0},  # global default would never fire
                },
            }
        )
        regions = build_regions(cfg.zones)
        engine = EventEngine.from_config(cfg, regions, "test")
        inside = make_track(1, (50, 50))
        engine.step(0, 0.0, [inside], [], None)
        events = engine.step(1, 0.6, [inside], [], None)
        self.assertIn(EventType.OBJECT_DWELL_STARTED, [e.event_type for e in events])


class LineCrossingEventTests(unittest.TestCase):
    def _engine(self, direction="any"):
        cfg = config_from_dict({"lines": {"l": {"start": [100, 0], "end": [100, 200], "direction": direction}}})
        return EventEngine.from_config(cfg, {}, "test")

    def test_crossing_emits_line_crossed(self):
        engine = self._engine()
        left = make_track(1, (50, 50))
        engine.step(0, 0.0, [left], [], None)
        right = make_track(1, (150, 50))
        events = engine.step(1, 1.0, [right], [], None)
        self.assertEqual([e.event_type for e in events], [EventType.LINE_CROSSED])

    def test_no_crossing_when_staying_on_one_side(self):
        engine = self._engine()
        a = make_track(1, (50, 50))
        engine.step(0, 0.0, [a], [], None)
        b = make_track(1, (60, 50))
        events = engine.step(1, 1.0, [b], [], None)
        self.assertEqual(events, [])

    def test_directed_line_ignores_wrong_direction_crossing(self):
        # For this line, moving left (x=50) -> right (x=150) lands on the line's negative side;
        # a filter for the "positive" direction must therefore ignore it.
        engine = self._engine(direction="positive")
        left = make_track(1, (50, 50))
        engine.step(0, 0.0, [left], [], None)
        right = make_track(1, (150, 50))
        events = engine.step(1, 1.0, [right], [], None)
        self.assertEqual(events, [])

    def test_directed_line_reports_matching_direction_crossing(self):
        engine = self._engine(direction="negative")
        left = make_track(1, (50, 50))
        engine.step(0, 0.0, [left], [], None)
        right = make_track(1, (150, 50))  # lands on the negative side -> matches the filter
        events = engine.step(1, 1.0, [right], [], None)
        self.assertEqual([e.event_type for e in events], [EventType.LINE_CROSSED])

    def test_jitter_within_hysteresis_band_does_not_count_as_crossing(self):
        cfg = config_from_dict(
            {
                "lines": {"l": {"start": [100, 0], "end": [100, 200]}},
                "events": {"line_crossing": {"hysteresis_px": 10.0}},
            }
        )
        engine = EventEngine.from_config(cfg, {}, "test")
        a = make_track(1, (95, 50))  # within dead band (distance 5 < 10)
        engine.step(0, 0.0, [a], [], None)
        b = make_track(1, (105, 50))  # also within dead band
        events = engine.step(1, 1.0, [b], [], None)
        self.assertEqual(events, [])


class VelocityEventTests(unittest.TestCase):
    def test_velocity_exceeded_fires_above_threshold(self):
        cfg = config_from_dict({"events": {"velocity": {"enabled": True, "threshold_px_s": 50.0, "min_observations": 2}}})
        engine = EventEngine.from_config(cfg, {}, "test")
        fast = make_track(1, (0, 0), matched_observations=5, velocity=(100.0, 0.0))
        events = engine.step(0, 0.0, [fast], [], None)
        self.assertEqual([e.event_type for e in events], [EventType.VELOCITY_EXCEEDED])

    def test_velocity_below_threshold_does_not_fire(self):
        cfg = config_from_dict({"events": {"velocity": {"enabled": True, "threshold_px_s": 500.0}}})
        engine = EventEngine.from_config(cfg, {}, "test")
        slow = make_track(1, (0, 0), matched_observations=5, velocity=(10.0, 0.0))
        events = engine.step(0, 0.0, [slow], [], None)
        self.assertEqual(events, [])

    def test_velocity_requires_minimum_observations(self):
        cfg = config_from_dict({"events": {"velocity": {"enabled": True, "threshold_px_s": 10.0, "min_observations": 10}}})
        engine = EventEngine.from_config(cfg, {}, "test")
        fast_but_new = make_track(1, (0, 0), matched_observations=2, velocity=(1000.0, 0.0))
        events = engine.step(0, 0.0, [fast_but_new], [], None)
        self.assertEqual(events, [])


class CooldownTests(unittest.TestCase):
    def test_repeated_line_crossings_within_cooldown_are_suppressed(self):
        cfg = config_from_dict(
            {
                "lines": {"l": {"start": [100, 0], "end": [100, 200]}},
                "events": {"cooldowns": {"LINE_CROSSED": 5.0}},
            }
        )
        engine = EventEngine.from_config(cfg, {}, "test")
        left = make_track(1, (50, 50))
        engine.step(0, 0.0, [left], [], None)
        right = make_track(1, (150, 50))
        e1 = engine.step(1, 1.0, [right], [], None)
        self.assertEqual(len(e1), 1)

        back_left = make_track(1, (50, 50))
        engine.step(2, 1.2, [back_left], [], None)
        back_right = make_track(1, (150, 50))
        e2 = engine.step(3, 1.4, [back_right], [], None)  # within 5s cooldown of the first crossing
        self.assertEqual(e2, [])

    def test_event_allowed_again_after_cooldown_elapses(self):
        cfg = config_from_dict(
            {
                "lines": {"l": {"start": [100, 0], "end": [100, 200]}},
                "events": {"cooldowns": {"LINE_CROSSED": 1.0}},
            }
        )
        engine = EventEngine.from_config(cfg, {}, "test")
        left = make_track(1, (50, 50))
        engine.step(0, 0.0, [left], [], None)
        right = make_track(1, (150, 50))
        e1 = engine.step(1, 1.0, [right], [], None)
        self.assertEqual(len(e1), 1)

        # Stay on the right side for a frame (no crossing), then cross back after the cooldown.
        still_right = make_track(1, (160, 50))
        engine.step(2, 2.0, [still_right], [], None)
        back_left = make_track(1, (50, 50))
        e2 = engine.step(3, 3.0, [back_left], [], None)  # 2s after first crossing, cooldown elapsed
        self.assertEqual(len(e2), 1)


class SinkAndSequenceTests(unittest.TestCase):
    def test_sink_receives_every_emitted_event_in_order(self):
        cfg = config_from_dict({"lines": {"l": {"start": [100, 0], "end": [100, 200]}}})
        engine = EventEngine.from_config(cfg, {}, "test")
        received = []
        engine.add_sink(received.append)
        left = make_track(1, (50, 50))
        engine.step(0, 0.0, [left], [], None)
        right = make_track(1, (150, 50))
        engine.step(1, 1.0, [right], [], None)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event_type, EventType.LINE_CROSSED)

    def test_event_ids_are_sequential(self):
        cfg = config_from_dict(
            {"events": {"motion": {"start_frames": 1, "stop_frames": 1, "active_interval_s": 0}}}
        )
        engine = EventEngine.from_config(cfg, {}, "test")
        e1 = engine.step(0, 0.0, [], [], make_motion(True))
        e2 = engine.step(1, 1.0, [], [], make_motion(False))
        self.assertEqual(e1[0].event_id, "evt-000001")
        self.assertEqual(e2[0].event_id, "evt-000002")


if __name__ == "__main__":
    unittest.main()
