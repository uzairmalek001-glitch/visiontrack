import unittest

from visiontrack.config import TrackingConfig
from visiontrack.detection import Detection
from visiontrack.tracking.centroid import CentroidTracker


def det(cx, cy, w=20, h=20, source="motion"):
    return Detection(bbox=(int(cx - w / 2), int(cy - h / 2), w, h), centroid=(cx, cy), source=source)


class BasicAssociationTests(unittest.TestCase):
    def setUp(self):
        self.tracker = CentroidTracker(TrackingConfig(max_disappeared=3, max_distance=50))

    def test_first_detection_creates_a_track(self):
        tracks = self.tracker.update([det(50, 50)], timestamp=0.0, frame_number=0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

    def test_ids_increment_and_never_repeat(self):
        # max_disappeared=3 tolerates 3 consecutive misses; track 1 is removed on the 4th miss.
        self.tracker.update([det(50, 50)], 0.0, 0)
        self.tracker.update([], 1.0, 1)
        self.tracker.update([], 2.0, 2)
        self.tracker.update([], 3.0, 3)
        self.tracker.update([], 4.0, 4)  # track 1 removed here
        tracks = self.tracker.update([det(50, 50)], 5.0, 5)
        self.assertEqual(tracks[0].track_id, 2)

    def test_same_object_keeps_same_id_across_frames(self):
        self.tracker.update([det(50, 50)], 0.0, 0)
        tracks = self.tracker.update([det(55, 50)], 1.0, 1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

    def test_far_detection_does_not_match_becomes_new_track(self):
        self.tracker.update([det(50, 50)], 0.0, 0)
        tracks = self.tracker.update([det(500, 500)], 1.0, 1)
        ids = {t.track_id for t in tracks}
        self.assertEqual(ids, {1, 2})

    def test_multiple_objects_tracked_independently(self):
        tracks = self.tracker.update([det(50, 50), det(300, 300)], 0.0, 0)
        self.assertEqual(len(tracks), 2)
        tracks2 = self.tracker.update([det(55, 52), det(305, 298)], 1.0, 1)
        ids = sorted(t.track_id for t in tracks2)
        self.assertEqual(ids, [1, 2])

    def test_crossing_paths_assign_by_nearest_centroid_not_swap(self):
        # Two objects approach; greedy nearest-centroid keeps ids stable while still separated.
        self.tracker.update([det(0, 50), det(200, 50)], 0.0, 0)
        tracks = self.tracker.update([det(20, 50), det(180, 50)], 1.0, 1)
        by_id = {t.track_id: t.centroid for t in tracks}
        self.assertEqual(by_id[1], (20, 50))
        self.assertEqual(by_id[2], (180, 50))


class DisappearanceTests(unittest.TestCase):
    def setUp(self):
        self.tracker = CentroidTracker(TrackingConfig(max_disappeared=2, max_distance=50))

    def test_track_survives_temporary_disappearance(self):
        self.tracker.update([det(50, 50)], 0.0, 0)
        self.tracker.update([], 1.0, 1)  # disappeared_frames=1
        tracks = self.tracker.update([det(52, 51)], 2.0, 2)  # reappears within tolerance
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].disappeared_frames, 0)

    def test_track_removed_after_max_disappeared_exceeded(self):
        self.tracker.update([det(50, 50)], 0.0, 0)
        self.tracker.update([], 1.0, 1)
        self.tracker.update([], 2.0, 2)
        tracks = self.tracker.update([], 3.0, 3)  # disappeared_frames now 3 > max_disappeared=2
        self.assertEqual(tracks, [])
        self.assertIn(1, self.tracker.removed_since_last_update())

    def test_removed_since_last_update_reports_only_latest_frame(self):
        self.tracker.update([det(50, 50)], 0.0, 0)
        self.tracker.update([], 1.0, 1)
        self.tracker.update([], 2.0, 2)
        self.tracker.update([], 3.0, 3)  # removed here
        self.assertEqual(self.tracker.removed_since_last_update(), [1])
        self.tracker.update([], 4.0, 4)
        self.assertEqual(self.tracker.removed_since_last_update(), [])


class VelocityTests(unittest.TestCase):
    def test_velocity_reflects_direction_and_speed(self):
        tracker = CentroidTracker(TrackingConfig(max_disappeared=3, max_distance=200, velocity_smoothing=1.0))
        tracker.update([det(0, 50)], 0.0, 0)
        tracks = tracker.update([det(100, 50)], 1.0, 1)  # 100 px in 1s => 100 px/s rightward
        self.assertAlmostEqual(tracks[0].velocity[0], 100.0, places=3)
        self.assertAlmostEqual(tracks[0].velocity[1], 0.0, places=3)
        self.assertEqual(tracks[0].direction, "right")
        self.assertAlmostEqual(tracks[0].speed, 100.0, places=3)

    def test_velocity_smoothing_dampens_a_single_outlier(self):
        tracker = CentroidTracker(TrackingConfig(max_disappeared=3, max_distance=500, velocity_smoothing=0.5))
        tracker.update([det(0, 50)], 0.0, 0)
        tracker.update([det(10, 50)], 1.0, 1)  # 10 px/s
        tracks = tracker.update([det(410, 50)], 2.0, 2)  # 400 px/s outlier
        self.assertLess(tracks[0].velocity[0], 400.0)
        self.assertGreater(tracks[0].velocity[0], 10.0)


class MinimumAreaTests(unittest.TestCase):
    def test_tiny_detections_are_ignored(self):
        tracker = CentroidTracker(TrackingConfig(max_disappeared=3, max_distance=50, minimum_track_area=1000))
        tracks = tracker.update([det(50, 50, w=5, h=5)], 0.0, 0)
        self.assertEqual(tracks, [])


class OverflowTests(unittest.TestCase):
    def test_new_tracks_beyond_max_tracks_are_dropped_and_counted(self):
        tracker = CentroidTracker(TrackingConfig(max_disappeared=3, max_distance=50, max_tracks=2))
        detections = [det(x, 50) for x in (0, 200, 400)]
        tracks = tracker.update(detections, 0.0, 0)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracker.overflow_count, 1)


if __name__ == "__main__":
    unittest.main()
