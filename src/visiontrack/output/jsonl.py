"""JSONL event writer: one compact JSON object per line, one line per event.

Frames are never written here — only structured event records — per the privacy design goal
of not persisting raw video by default.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from ..events.models import Event
from ..errors import ReplayError


class JsonlEventWriter:
    """Append-mode JSONL sink. Use as a context manager or call :meth:`close` explicitly."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._file = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        """Event-sink interface: write one line and flush so a crash never loses buffered events."""
        if self._file is None:
            self.open()
        self._file.write(event.to_json())
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlEventWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl_events(path: str) -> list[Event]:
    """Read a JSONL event log and report invalid baselines cleanly."""
    import json

    events: list[Event] = []

    try:
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReplayError(
                        f"{path}: invalid JSON at line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(data, dict):
                    raise ReplayError(
                        f"{path}: expected an event object at line {line_number}"
                    )

                try:
                    events.append(Event.from_dict(data))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ReplayError(
                        f"{path}: invalid event at line {line_number}: {exc}"
                    ) from exc

    except ReplayError:
        raise
    except OSError as exc:
        raise ReplayError(f"{path}: cannot read replay baseline: {exc}") from exc

    return events
