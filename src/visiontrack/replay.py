"""Deterministic replay: run a file/image-sequence source through the pipeline and, optionally,
compare the resulting event stream to a previously recorded JSONL baseline for regression testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .capture import open_source
from .config import VisionTrackConfig
from .events.models import Event
from .output.jsonl import JsonlEventWriter, read_jsonl_events
from .pipeline import Pipeline
from .utils.timing import MetricsCollector


@dataclass(slots=True)
class ReplayDiff:
    """Difference between two event streams at the first point they diverge."""

    index: int
    expected: dict | None
    actual: dict | None

    @property
    def matches(self) -> bool:
        return self.expected == self.actual


def replay(
    source_uri: str, config: VisionTrackConfig, jsonl_out: str | None = None
) -> tuple[list[Event], MetricsCollector]:
    """Run ``source_uri`` through a fresh pipeline and return every emitted event, in order.

    With deterministic settings (no ``ml`` component, or a deterministic ML backend, and
    ``logging.wall_clock_timestamps: false``) the same input file always produces the same
    event sequence, which is what makes this useful for regression testing.
    """
    pipeline = Pipeline(config=config, source_name=source_uri)
    writer = None
    if jsonl_out:
        writer = JsonlEventWriter(jsonl_out)
        writer.open()
        pipeline.add_event_sink(writer)
    try:
        events: list[Event] = []
        pipeline.add_event_sink(events.append)
        source = open_source(source_uri, config.source)
        metrics = pipeline.run(source)
        return events, metrics
    finally:
        if writer is not None:
            writer.close()


def compare_events(expected_path: str, actual: list[Event]) -> list[ReplayDiff]:
    """Compare ``actual`` events to a recorded JSONL baseline.

    Only fields that matter for reproducibility are compared (``wall_time`` is excluded, since
    it is non-deterministic by nature). Returns every mismatching index; an empty list means
    the streams matched exactly.
    """
    expected = read_jsonl_events(expected_path)
    diffs: list[ReplayDiff] = []
    length = max(len(expected), len(actual))
    for i in range(length):
        e = _comparable(expected[i]) if i < len(expected) else None
        a = _comparable(actual[i]) if i < len(actual) else None
        if e != a:
            diffs.append(ReplayDiff(i, e, a))
    return diffs


def _comparable(event: Event) -> dict:
    data = event.to_dict()
    data.pop("wall_time", None)
    return data
