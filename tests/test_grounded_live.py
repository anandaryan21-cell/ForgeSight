"""The fused dashboard queues advice against the exact saved record."""

from concurrent.futures import Future
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from forgesight.grounded_live import GroundedLive


class GroundedLiveTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.queue = Mock()
        self.bridge = GroundedLive(self.queue, self.root, self.root / "llama.sock", 30)
        self.record = {
            "status": "ANALYZED", "candidate": "ANOMALY_CANDIDATE",
            "decision": {"grade": "REJECT"}, "inspection_id": "test-inspection",
        }

    def test_skip_normal_or_failed_record(self):
        for updates in ({"candidate": "NORMAL_CANDIDATE"}, {"status": "ERROR"},
                        {"decision": {"grade": "PASS"}}):
            self.bridge.submit({**self.record, **updates}, Mock())
        self.queue.submit.assert_not_called()

    @patch("forgesight.grounded_live.advise")
    def test_async_advice_bound_to_its_inspection(self, advise):
        future = Future()
        self.queue.submit.return_value = future
        self.bridge.submit(self.record, lambda: False)
        self.assertEqual(self.bridge.state(self.record, False)["status"], "PENDING")
        other = {**self.record, "inspection_id": "other"}
        self.assertEqual(self.bridge.state(other, False)["status"], "NOT_ELIGIBLE")
        advise.return_value = {"status": "CHECKLIST_FALLBACK", "advice": "Review by hand."}
        job, = self.queue.submit.call_args.args
        future.set_result(job())
        advise.assert_called_once_with(self.record, self.bridge.rules, None, 8)
        state = self.bridge.state(self.record, False)
        self.assertEqual(state["status"], "CHECKLIST_FALLBACK")
        self.assertEqual(state["advisory"]["note"], "Review by hand.")
        saved = json.loads((self.root / "test-inspection.json").read_text())
        self.assertEqual(saved, advise.return_value)
