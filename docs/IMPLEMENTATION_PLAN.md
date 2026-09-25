# VisionTrack – architecture & implementation plan

## Layering (dependencies point downwards only)

```
cli ─► pipeline ─► capture / vision / tracking / regions / events / output / detectors
                        │         │         │           │          │
                        └─────────┴─────────┴───────────┴──────────┴──► detection.py (Detection), errors.py, utils/
config.py is a leaf: it only depends on events.models (for EventType names).
```

* `detection.py`  – the unified `Detection` model. Tracking/events never know where a detection came from.
* `capture/`      – `FrameSource` interface + camera / video file / image sequence / network stream.
* `vision/`       – stateless-ish OpenCV building blocks (preprocess, motion, contours, geometry).
* `tracking/`     – `BaseTracker` + deterministic `CentroidTracker`; `Track` state.
* `regions/`      – polygon / rectangle ROI, ROI masks, debounced zone occupancy.
* `events/`       – event models, rule classes (state machines), engine with cooldowns, dispatcher/sinks.
* `detectors/`    – `BaseDetector` + optional ML adapter machinery (never imported by the core path).
* `output/`       – console logger, JSONL writer, OpenCV overlay.
* `pipeline.py`   – wires the stages; `process_frame()` is a pure step that is easy to test.
* `replay.py`     – deterministic replay + event-stream comparison.

## Determinism rules

* Timestamps are *stream time* (`frame_index / fps` for files, monotonic clock for live sources).
* Event IDs are sequential; wall-clock time is opt-in and never used for decisions.
* Association is greedy with explicit tie-breaking (distance, track id, detection index).
* No randomness, no threads, no wall-clock sleeps.

## Phases

| Phase | Scope |
|-------|-------|
| 1 | structure, config + validation, capture, preprocessing, motion detection, CLI `run`/`inspect-config` |
| 2 | ROI, centroid tracker, track lifecycle, event engine (motion + zone events) |
| 3 | line crossing, dwell, cooldown/debounce, JSONL events |
| 4 | overlay/display, performance metrics, deterministic replay |
| 5 | broad test suite, README, optional ML adapter interface |

After each phase: run tests → inspect failures → fix root causes → rerun → update docs.
