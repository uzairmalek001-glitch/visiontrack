import unittest

from visiontrack.config import ZoneConfig
from visiontrack.regions.roi import Region, build_regions, combined_mask
from visiontrack.regions.zones import ZoneOccupancyTracker


class RegionMembershipTests(unittest.TestCase):
    def test_rect_zone_contains_centroid_inside(self):
        region = Region.from_config("a", ZoneConfig(rect=[0, 0, 100, 100]))
        self.assertTrue(region.contains_point((50, 50)))

    def test_rect_zone_excludes_centroid_outside(self):
        region = Region.from_config("a", ZoneConfig(rect=[0, 0, 100, 100]))
        self.assertFalse(region.contains_point((500, 500)))

    def test_polygon_zone_membership(self):
        region = Region.from_config(
            "tri", ZoneConfig(polygon=[[0, 0], [100, 0], [50, 100]])
        )
        self.assertTrue(region.contains_point((50, 10)))
        self.assertFalse(region.contains_point((5, 90)))

    def test_bbox_mode_intersection_counts_partial_overlap(self):
        region = Region.from_config("a", ZoneConfig(rect=[0, 0, 50, 50], mode="bbox"))
        # centroid outside, but bbox overlaps the zone
        self.assertFalse(region.contains_point((60, 60)))
        self.assertTrue(region.contains_bbox((40, 40, 30, 30)))

    def test_build_regions_from_multiple_zone_configs(self):
        regions = build_regions(
            {"a": ZoneConfig(rect=[0, 0, 10, 10]), "b": ZoneConfig(rect=[20, 20, 10, 10])}
        )
        self.assertEqual(set(regions), {"a", "b"})

    def test_region_mask_marks_only_the_region(self):
        region = Region.from_config("a", ZoneConfig(rect=[10, 10, 20, 20]))
        mask = region.mask((100, 100))
        self.assertEqual(mask[15, 15], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_combined_mask_unions_regions(self):
        r1 = Region.from_config("a", ZoneConfig(rect=[0, 0, 10, 10]))
        r2 = Region.from_config("b", ZoneConfig(rect=[50, 50, 10, 10]))
        mask = combined_mask([r1, r2], (100, 100))
        self.assertEqual(mask[5, 5], 255)
        self.assertEqual(mask[55, 55], 255)
        self.assertEqual(mask[90, 90], 0)


class ZoneOccupancyDebounceTests(unittest.TestCase):
    def test_no_transition_reported_below_enter_frames(self):
        occ = ZoneOccupancyTracker(enter_frames=3, exit_frames=3)
        self.assertIsNone(occ.update(1, "a", True, 0.0))
        self.assertIsNone(occ.update(1, "a", True, 1.0))

    def test_enter_transition_after_enough_consecutive_frames(self):
        occ = ZoneOccupancyTracker(enter_frames=3, exit_frames=3)
        occ.update(1, "a", True, 0.0)
        occ.update(1, "a", True, 1.0)
        t = occ.update(1, "a", True, 2.0)
        self.assertIsNotNone(t)
        self.assertTrue(t.entered)
        self.assertTrue(occ.is_inside(1, "a"))

    def test_single_frame_flicker_does_not_confirm_exit(self):
        occ = ZoneOccupancyTracker(enter_frames=2, exit_frames=3)
        occ.update(1, "a", True, 0.0)
        occ.update(1, "a", True, 1.0)  # confirmed entered
        occ.update(1, "a", False, 2.0)  # 1 miss
        occ.update(1, "a", True, 3.0)  # back inside, resets miss streak
        self.assertTrue(occ.is_inside(1, "a"))

    def test_exit_confirmed_after_enough_consecutive_frames_outside(self):
        occ = ZoneOccupancyTracker(enter_frames=2, exit_frames=2)
        occ.update(1, "a", True, 0.0)
        occ.update(1, "a", True, 1.0)  # entered
        occ.update(1, "a", False, 2.0)
        t = occ.update(1, "a", False, 3.0)
        self.assertIsNotNone(t)
        self.assertFalse(t.entered)
        self.assertFalse(occ.is_inside(1, "a"))

    def test_dwell_seconds_measures_time_since_confirmed_entry(self):
        occ = ZoneOccupancyTracker(enter_frames=1, exit_frames=1)
        occ.update(1, "a", True, 10.0)
        self.assertAlmostEqual(occ.dwell_seconds(1, "a", 13.0), 3.0)

    def test_exit_event_reports_dwell_duration(self):
        occ = ZoneOccupancyTracker(enter_frames=1, exit_frames=1)
        occ.update(1, "a", True, 10.0)
        t = occ.update(1, "a", False, 14.0)
        self.assertAlmostEqual(t.dwell_seconds, 4.0)

    def test_forget_track_clears_state_and_reports_open_exit(self):
        occ = ZoneOccupancyTracker(enter_frames=1, exit_frames=1)
        occ.update(1, "a", True, 0.0)
        transitions = occ.forget_track(1)
        self.assertEqual(len(transitions), 1)
        self.assertFalse(occ.is_inside(1, "a"))

    def test_zones_for_track_lists_only_confirmed_zones(self):
        occ = ZoneOccupancyTracker(enter_frames=1, exit_frames=1)
        occ.update(1, "a", True, 0.0)
        occ.update(1, "b", True, 0.0)
        self.assertEqual(set(occ.zones_for_track(1)), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
