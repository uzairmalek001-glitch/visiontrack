"""Dynamic loading of an optional ML detector backend.

VisionTrack's core never imports an ML framework. When ``ml.enabled: true``, the pipeline calls
:func:`load_detector`, which imports *only* the module named in ``ml.backend`` — so a machine
without PyTorch/ONNX/TensorFlow installed can run the deterministic engine untouched.

``ml.backend`` is a ``"package.module:ClassName"`` string pointing at a concrete
:class:`~visiontrack.detectors.base.BaseDetector` subclass, e.g.::

    ml:
      enabled: true
      backend: "myproject.detectors.yolo:YoloDetector"
      model_path: "yolov8n.onnx"

No such adapter ships with VisionTrack V1: this module only provides the loading machinery, per
the "optional ML architecture" design goal (the deterministic engine must not depend on ML).
"""

from __future__ import annotations

import importlib

from ..config import MLConfig
from ..errors import DetectorUnavailableError
from .base import BaseDetector


def load_detector(config: MLConfig) -> BaseDetector | None:
    """Instantiate the detector named by ``config.backend``.

    Returns ``None`` (never raises) if ``config.enabled`` is ``False``. If loading fails and
    ``config.required`` is ``False``, returns ``None`` so the deterministic pipeline continues
    without ML. If ``config.required`` is ``True``, raises :class:`DetectorUnavailableError`.
    """
    if not config.enabled:
        return None
    assert config.backend is not None  # enforced by MLConfig.validate
    try:
        return _instantiate(config.backend, config)
    except Exception as exc:  # noqa: BLE001 - intentionally broad: any backend failure is caught
        if config.required:
            raise DetectorUnavailableError(
                f"required ML backend {config.backend!r} could not be loaded: {exc}"
            ) from exc
        return None


def _instantiate(backend: str, config: MLConfig) -> BaseDetector:
    if ":" not in backend:
        raise DetectorUnavailableError(
            f"ml.backend must be 'package.module:ClassName', got {backend!r}"
        )
    module_name, class_name = backend.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise DetectorUnavailableError(f"{module_name!r} has no attribute {class_name!r}")
    if not (isinstance(cls, type) and issubclass(cls, BaseDetector)):
        raise DetectorUnavailableError(f"{backend} is not a BaseDetector subclass")
    return cls(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        labels=config.labels,
        **config.options,
    )
