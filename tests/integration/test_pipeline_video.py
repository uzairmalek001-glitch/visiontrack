"""End-to-end integration tests: real video files through the full Pipeline, headless.

No physical camera is used anywhere - only files written by OpenCV's VideoWriter to a temp
directory, exercising the same cv2.VideoCapture decode path a real video file would.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from visiontrack.capture.video import VideoFileSource
from visiontrack.config import SourceConfig, config_from_dict
from visiontrack.pipeline import Pipeline


def _write_video(path: str, frames: list[np.ndarray], fps: float = 30.0) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _moving_object_frames(n: int, width=160, height=120, step=6, size=16) -> list[np.ndarray]:
    bg = np.full((height, width, 3), 30, np.uint8)
    frames = []
    for i in range(n):
        frame = bg.copy()
        x = 5 + i * step
        if x + size < width:
            cv2.rectangle(frame, (x, 50), (x + size, 50 + size), (230, 230, 230), -1)
        frames.append(frame)
    return frames


def _static_frames(n: int, width=160, height=120) -> list[np.ndarray]:
    bg = np.full((height, width, 3), 30, np.uint8)
    return [bg.copy() for _ in range(n)]


class MovingObjectPipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "moving.mp4")
        _write_video(self.path, _moving_object_frames(60))

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, **overrides):
        base = {
            "preprocessing": {"gaussian_blur_kernel": 3},
            "motion": {"method": "frame_diff", "min_area": 30, "warmup_frames": 0},
            "tracking": {"max_distance": 100},
        }
        base.update(overrides)
        return config_from_dict(base)

    def test_motion_and_tracking_events_are_produced(self):
        config = self._config()
        pipeline = Pipeline(config=config, source_name="moving.mp4")
        events = []
        pipeline.add_event_sink(events.append)
        source = VideoFileSource(self.path, config.source)
        metrics = pipeline.run(source)

        self.assertEqual(metrics.frames_read, 60)
        self.assertEqual(metrics.frames_processed, 60)
        self.assertGreater(metrics.max_active_tracks, 0)
        types = {e.event_type for e in events}
        self.assertIn("MOTION_STARTED", {str(t) for t in types})

    def test_line_crossing_detected_for_object_traversing_frame(self):
        config = self._config(lines={"mid": {"start": [80, 0], "end": [80, 120]}})
        pipeline = Pipeline(config=config, source_name="moving.mp4")
        events = []
        pipeline.add_event_sink(events.append)
        source = VideoFileSource(self.path, config.source)
        pipeline.run(source)
        self.assertTrue(any(str(e.event_type) == "LINE_CROSSED" for e in events))

    def test_zone_entered_and_left_for_object_passing_through(self):
        config = self._config(zones={"zone": {"rect": [40, 30, 60, 60]}})
        pipeline = Pipeline(config=config, source_name="moving.mp4")
        events = []
        pipeline.add_event_sink(events.append)
        source = VideoFileSource(self.path, config.source)
        pipeline.run(source)
        types = [str(e.event_type) for e in events]
        self.assertIn("OBJECT_ENTERED_ZONE", types)
        self.assertIn("OBJECT_LEFT_ZONE", types)

    def test_deterministic_replay_produces_identical_event_stream(self):
        config = self._config()
        events_a: list = []
        p1 = Pipeline(config=config, source_name="moving.mp4")
        p1.add_event_sink(events_a.append)
        p1.run(VideoFileSource(self.path, config.source))

        events_b: list = []
        p2 = Pipeline(config=self._config(), source_name="moving.mp4")
        p2.add_event_sink(events_b.append)
        p2.run(VideoFileSource(self.path, config.source))

        self.assertEqual([e.to_dict() for e in events_a], [e.to_dict() for e in events_b])


class StaticVideoPipelineTests(unittest.TestCase):
    def test_no_motion_events_on_a_static_video(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "static.mp4")
            _write_video(path, _static_frames(30))
            config = config_from_dict({"motion": {"method": "frame_diff", "warmup_frames": 0}})
            pipeline = Pipeline(config=config, source_name="static.mp4")
            events = []
            pipeline.add_event_sink(events.append)
            pipeline.run(VideoFileSource(path, config.source))
            self.assertEqual(events, [])


class CorruptedAndMissingVideoTests(unittest.TestCase):
    def test_missing_file_raises_capture_error(self):
        from visiontrack.errors import CaptureError

        with self.assertRaises(CaptureError):
            VideoFileSource("/no/such/video.mp4").open()

    def test_corrupted_file_raises_capture_error_on_open(self):
        from visiontrack.errors import CaptureError

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "corrupt.mp4"
            path.write_bytes(b"this is not a real video file")
            with self.assertRaises(CaptureError):
                VideoFileSource(str(path)).open()


class ImageSequenceTests(unittest.TestCase):
    def test_image_sequence_source_reads_frames_in_order(self):
        from visiontrack.capture.video import ImageSequenceSource

        with tempfile.TemporaryDirectory() as d:
            for i, frame in enumerate(_static_frames(5)):
                cv2.imwrite(str(Path(d) / f"frame_{i:03d}.png"), frame)
            source = ImageSequenceSource(d, SourceConfig(assumed_fps=10.0))
            source.open()
            frames = list(source)
            source.close()
            self.assertEqual(len(frames), 5)
            self.assertAlmostEqual(frames[1].timestamp, 0.1, places=5)


if __name__ == "__main__":
    unittest.main()
