"""Optional ML detector interface. The deterministic engine never imports this module's
implementations — only :class:`BaseDetector` as a type, when ``ml.enabled`` wires one in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..detection import Detection


class BaseDetector(ABC):
    """Interface every ML detector adapter (YOLO, ONNX Runtime, TensorFlow, ...) must implement."""

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> list[Detection]:
        """Run inference on one BGR frame and return normalized :class:`Detection` objects.

        Implementations must set ``source="ml"`` and a ``confidence`` in ``[0, 1]`` on every
        detection returned.
        """

    def close(self) -> None:
        """Release model resources. Default no-op; override if the backend holds a session/handle."""
