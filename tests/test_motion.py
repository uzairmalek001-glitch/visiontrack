import unittest

from tests.helpers import blank_frame, frame_with_rect, moving_rectangle_sequence, noisy_frame
from visiontrack.config import MotionConfig, MorphologyConfig
from visiontrack.vision.motion import MotionDetector
from visiontrack.vision.preprocessing import to_gray


def _gray(image):
    return to_gray(image)


class FrameDiffMotionTests(unittest.TestCase):
    def setUp(self):
        cfg = MotionConfig(method="frame_diff", threshold=25, min_area=50, warmup_frames=0)
        cfg.morphology = MorphologyConfig(open_iterations=0, close_iterations=0, dilate_iterations=0)
        self.detector = MotionDetector(cfg)

    def test_no_motion_on_first_frame_warmup(self):
        result = self.detector.detect(_gray(blank_frame()))
        self.assertFalse(result.present)

    def test_no_motion_between_identical_frames(self):
        self.detector.detect(_gray(blank_frame()))
        result = self.detector.detect(_gray(blank_frame()))
        self.assertFalse(result.present)

    def test_motion_detected_for_moved_rectangle(self):
        self.detector.detect(_gray(frame_with_rect(10, 10)))
        result = self.detector.detect(_gray(frame_with_rect(60, 10)))
        self.assertTrue(result.present)
        self.assertEqual(result.detections[0].source, "motion")
        self.assertIsNone(result.detections[0].confidence)

    def test_small_change_below_min_area_ignored(self):
        cfg = MotionConfig(method="frame_diff", threshold=25, min_area=100_000, warmup_frames=0)
        detector = MotionDetector(cfg)
        detector.detect(_gray(frame_with_rect(10, 10, w=5, h=5)))
        result = detector.detect(_gray(frame_with_rect(15, 10, w=5, h=5)))
        self.assertFalse(result.present)

    def test_reset_forgets_previous_frame(self):
        self.detector.detect(_gray(frame_with_rect(10, 10)))
        self.detector.reset()
        result = self.detector.detect(_gray(frame_with_rect(60, 10)))
        self.assertFalse(result.present)  # warm-up again after reset


class RunningAverageMotionTests(unittest.TestCase):
    def test_settles_on_static_background(self):
        cfg = MotionConfig(method="running_average", threshold=20, min_area=50, warmup_frames=3, background_alpha=0.3)
        detector = MotionDetector(cfg)
        frames = [blank_frame() for _ in range(8)]
        results = [detector.detect(_gray(f)) for f in frames]
        self.assertFalse(any(r.present for r in results))

    def test_detects_object_entering_scene(self):
        cfg = MotionConfig(method="running_average", threshold=20, min_area=50, warmup_frames=3, background_alpha=0.3)
        detector = MotionDetector(cfg)
        for _ in range(5):
            detector.detect(_gray(blank_frame()))
        result = detector.detect(_gray(frame_with_rect(80, 60, w=30, h=30)))
        self.assertTrue(result.present)


class Mog2MotionTests(unittest.TestCase):
    def test_mog2_detects_moving_object(self):
        cfg = MotionConfig(method="mog2", min_area=50, warmup_frames=0, history=20, var_threshold=8.0)
        detector = MotionDetector(cfg)
        result = None
        for frame in moving_rectangle_sequence(15, step=8):
            result = detector.detect(_gray(frame))
        self.assertTrue(result.present)


class RoiMaskingTests(unittest.TestCase):
    def test_motion_outside_roi_mask_is_ignored(self):
        import numpy as np

        cfg = MotionConfig(method="frame_diff", threshold=25, min_area=50, warmup_frames=0)
        detector = MotionDetector(cfg)
        w, h = 200, 150
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:50, 0:50] = 255  # only top-left allowed

        detector.detect(_gray(frame_with_rect(100, 100)), roi_mask=mask)  # object outside ROI
        result = detector.detect(_gray(frame_with_rect(110, 100)), roi_mask=mask)
        self.assertFalse(result.present)

    def test_motion_inside_roi_mask_is_detected(self):
        import numpy as np

        cfg = MotionConfig(method="frame_diff", threshold=25, min_area=50, warmup_frames=0)
        detector = MotionDetector(cfg)
        w, h = 200, 150
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:80, 0:80] = 255

        detector.detect(_gray(frame_with_rect(10, 10)), roi_mask=mask)
        result = detector.detect(_gray(frame_with_rect(30, 10)), roi_mask=mask)
        self.assertTrue(result.present)


class NoisyFrameTests(unittest.TestCase):
    def test_low_amplitude_sensor_noise_does_not_trigger_motion(self):
        cfg = MotionConfig(
            method="running_average", threshold=40, min_area=100, warmup_frames=3, background_alpha=0.3
        )
        cfg.morphology = MorphologyConfig(open_iterations=1, close_iterations=0, dilate_iterations=0)
        detector = MotionDetector(cfg)
        results = [detector.detect(_gray(noisy_frame(seed=i, sigma=5.0))) for i in range(10)]
        self.assertFalse(any(r.present for r in results[3:]))


if __name__ == "__main__":
    unittest.main()
