"""Tests for the optional ML detector interface and loader.

These confirm the deterministic engine's core promise: it must build and run identically
whether or not any ML backend is installed. No actual ML framework is imported anywhere here.
"""

from __future__ import annotations

import unittest

import numpy as np

from visiontrack.config import MLConfig
from visiontrack.detection import Detection
from visiontrack.detectors.base import BaseDetector
from visiontrack.detectors.optional_ml import load_detector
from visiontrack.errors import DetectorUnavailableError


class DisabledMlTests(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(load_detector(MLConfig(enabled=False)))

    def test_disabled_ignores_a_missing_backend(self):
        self.assertIsNone(load_detector(MLConfig(enabled=False, backend=None)))


class BackendResolutionTests(unittest.TestCase):
    def test_nonexistent_module_is_swallowed_when_not_required(self):
        cfg = MLConfig(enabled=True, backend="nonexistent.module:Detector", required=False)
        self.assertIsNone(load_detector(cfg))

    def test_nonexistent_module_raises_when_required(self):
        cfg = MLConfig(enabled=True, backend="nonexistent.module:Detector", required=True)
        with self.assertRaises(DetectorUnavailableError):
            load_detector(cfg)

    def test_malformed_backend_string_raises_when_required(self):
        cfg = MLConfig(enabled=True, backend="not-a-valid-spec", required=True)
        with self.assertRaises(DetectorUnavailableError):
            load_detector(cfg)

    def test_class_not_a_base_detector_subclass_raises_when_required(self):
        cfg = MLConfig(enabled=True, backend="tests.optional.test_ml_detector:NotADetector", required=True)
        with self.assertRaises(DetectorUnavailableError):
            load_detector(cfg)

    def test_valid_stub_backend_loads_and_is_usable(self):
        cfg = MLConfig(enabled=True, backend="tests.optional.test_ml_detector:StubDetector", required=True)
        detector = load_detector(cfg)
        self.assertIsInstance(detector, StubDetector)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        detections = detector.detect(frame, timestamp=1.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].source, "ml")
        self.assertIsNotNone(detections[0].confidence)


class DeterministicDetectionInvariantTests(unittest.TestCase):
    def test_deterministic_detection_cannot_carry_a_confidence(self):
        with self.assertRaises(ValueError):
            Detection(bbox=(0, 0, 10, 10), centroid=(5, 5), source="motion", confidence=0.9)

    def test_ml_detection_confidence_must_be_in_unit_range(self):
        with self.assertRaises(ValueError):
            Detection(bbox=(0, 0, 10, 10), centroid=(5, 5), source="ml", confidence=1.5)


class NotADetector:
    """A class that does not implement BaseDetector - used to test rejection."""


class StubDetector(BaseDetector):
    """A minimal, fully working ML detector adapter used only by these tests."""

    def __init__(self, model_path=None, confidence_threshold=0.5, labels=None, **options):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> list[Detection]:
        return [
            Detection(
                bbox=(0, 0, 10, 10),
                centroid=(5, 5),
                source="ml",
                timestamp=timestamp,
                label="stub",
                confidence=0.99,
            )
        ]


if __name__ == "__main__":
    unittest.main()
