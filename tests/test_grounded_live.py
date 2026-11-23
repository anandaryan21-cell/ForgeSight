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
        advise.assert_called_once_with(self.record, self.bridge.rules, None, 8, llama_root=None)
        state = self.bridge.state(self.record, False)
        self.assertEqual(state["status"], "CHECKLIST_FALLBACK")
        self.assertEqual(state["advisory"]["note"], "Review by hand.")
        saved = json.loads((self.root / "test-inspection.json").read_text())
        self.assertEqual(saved, advise.return_value)

    @patch("forgesight.grounded_live.advise")
    def test_timeout_replaces_an_alive_but_unresponsive_worker(self, advise):
        self.bridge.worker_unhealthy = Mock()
        advise.return_value = {"status": "CHECKLIST_FALLBACK",
                               "persistent_worker_unhealthy": True}
        self.bridge._run(self.record, lambda: True)
        self.bridge.worker_unhealthy.assert_called_once_with()

    def test_saved_runtime_is_independent_of_current_worker_availability(self):
        for recorded, available in (("PERSISTENT_WORKER", False), ("NOT_REQUESTED", True)):
            with self.subTest(recorded=recorded):
                future = Future()
                future.set_result({"status": "LLAMA_ADVICE", "advice": "Check the surface.",
                                   "llama_runtime_mode": recorded})
                self.bridge.latest_job = (self.record["inspection_id"], future)
                state = self.bridge.state(self.record, available)
                self.assertEqual(state["llama_runtime"], recorded)
                self.assertEqual(state["advice_worker_available_now"], available)

    def test_restarting_worker_is_not_reported_available(self):
        from forgesight.jetson_live_ui import LiveService
        service = LiveService.__new__(LiveService)
        service.advice_worker = Mock()
        service.advice_worker.poll.return_value = None
        service.advice_worker_respawning = True
        self.assertFalse(service._grounded_worker_available())
