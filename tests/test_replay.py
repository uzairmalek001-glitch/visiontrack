import tempfile
import unittest
from pathlib import Path

from visiontrack.errors import ReplayError
from visiontrack.output.jsonl import read_jsonl_events


class ReplayJsonlTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_malformed_json(self):
        with self.assertRaisesRegex(ReplayError, r"invalid JSON at line 1"):
            read_jsonl_events(
                self.write("bad.jsonl", "{BROKEN JSON\n")
            )

    def test_missing_event_field(self):
        with self.assertRaisesRegex(ReplayError, r"invalid event at line 1"):
            read_jsonl_events(
                self.write("bad.jsonl", '{"event_id":"x"}\n')
            )

    def test_missing_baseline(self):
        with self.assertRaisesRegex(ReplayError, r"cannot read replay baseline"):
            read_jsonl_events(
                str(self.root / "does_not_exist.jsonl")
            )

    def test_non_object_json(self):
        with self.assertRaisesRegex(ReplayError, r"expected an event object at line 1"):
            read_jsonl_events(
                self.write("bad.jsonl", "[]\n")
            )


if __name__ == "__main__":
    unittest.main()
