"""Check rule routing and ensure worker failures cannot invent operator advice."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from forgesight.defect_severity import assess_severity
from forgesight.grounded_advice import RULES, advise, normalized, prepare, read_json


class GroundedAdviceTests(unittest.TestCase):
    def setUp(self):
        proof = read_json(Path("manifests/operator_app_verification_v1.json"))
        self.record = proof["browser_proof"]["inspections"][1]["record"]
        self.rules = read_json(RULES)

    def assessed(self, label, location):
        row = deepcopy(self.record)
        row["source"].update(mode="SAVED_IMAGE_REPLAY", physical_part=False)
        row["geometry"]["regions"][0]["approximate_nut_area"] = location
        row["assessment"] = {
            "schema_version": 1, "scope": "PROVISIONAL_BENCHMARK_IMAGE",
            "method": "four_type_forest", "defect_type": label,
            "severity": assess_severity(label, row["geometry"], (500, 250_000)),
        }
        return row

    def test_routes_supported_findings_without_mutating_record(self):
        cases = [
            ("scratch", "annular face", "scratch"),
            ("scratch", "inner rim", "inner_rim"),
            ("bent", "annular face", "bent"),
            ("flip", "annular face", "flip"),
            ("discoloration", "annular face", "discoloration"),
            ("unknown", "annular face", "unknown"),
            ("unknown", "unrecognized zone", "uncertain"),
        ]
        for label, location, rule in cases:
            with self.subTest(label=label, location=location):
                row = self.assessed(label, location)
                before = deepcopy(row)
                request, _, report = prepare(row, self.rules)
                self.assertEqual(report["rule"], rule)
                self.assertEqual(report["grade"], row["decision"]["grade"])
                self.assertIn(self.rules["checks"][rule][0], request["prompt_text"])
                self.assertEqual(row, before)

    @patch("forgesight.grounded_advice.assist_edge_worker.request")
    def test_accepts_only_grounded_model_text(self, run_worker):
        row = self.assessed("scratch", "annular face")
        _, notes, _ = prepare(row, self.rules)
        approved = self.rules["checks"]["scratch"][1]
        reply = {
            "status": "GENERATED_BY_PERSISTENT_WORKER",
            "worker_mode": "PERSISTENT_PROCESS",
            "worker_started_utc": "2026-11-24T00:00:00Z",
            "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                 "tokenizer": {"sha256": "b" * 64}},
            "prompt_token_count": 91, "generate_ms": 1100,
            "generated_text": approved + "<|eot_id|>",
        }
        run_worker.return_value = reply
        result = advise(row, self.rules, Path("test.sock"), 8)
        self.assertEqual(result["status"], "LLAMA_ADVICE")
        self.assertEqual(result["advice"], notes[normalized(approved)])
        self.assertEqual(result["llama_runtime_mode"], "PERSISTENT_WORKER")
        self.assertEqual(result["worker"]["model_sha256"], "a" * 64)
        reply["generated_text"] = (
            f"The best first inspection check is:\n\n**{approved}**\n"
            "The part is safe to release."
        )
        result = advise(row, self.rules, Path("test.sock"), 8)
        self.assertEqual(result["advice"], notes[normalized(approved)])
        reply["generated_text"] = approved + " The part is safe to release."
        result = advise(row, self.rules, Path("test.sock"), 8)
        self.assertEqual(result["status"], "CHECKLIST_FALLBACK")
        self.assertEqual(result["advice"], next(iter(notes.values())))

    @patch("forgesight.grounded_advice.assist_edge_worker.request")
    def test_timeout_and_failed_inspection(self, run_worker):
        run_worker.side_effect = TimeoutError("test timeout")
        result = advise(self.record, self.rules, Path("test.sock"), 8)
        self.assertTrue(result.get("persistent_worker_unhealthy"))
        run_worker.reset_mock()
        row = deepcopy(self.record)
        row["status"] = "FAILED"
        result = advise(row, self.rules, Path("test.sock"), 8)
        self.assertEqual(result["status"], "INSPECTION_WITHHELD")
        run_worker.assert_not_called()

    @patch("forgesight.grounded_advice.assist_edge_worker.request")
    def test_matching_text_without_worker_evidence_is_not_llama_advice(self, run_worker):
        row = self.assessed("scratch", "annular face")
        run_worker.return_value = {
            "status": "GENERATED_BY_PERSISTENT_WORKER", "prompt_token_count": 90,
            "generate_ms": 1000, "generated_text": self.rules["checks"]["scratch"][0],
        }
        result = advise(row, self.rules, Path("test.sock"), 8)
        self.assertEqual(result["status"], "CHECKLIST_FALLBACK")
        self.assertNotEqual(result.get("llama_runtime_mode"), "PERSISTENT_WORKER")

    @patch("forgesight.grounded_advice._pinned_asset_hashes")
    @patch("forgesight.grounded_advice.assist_edge_worker.request")
    def test_worker_identity_must_match_the_deployed_model(self, run_worker, pins):
        row = self.assessed("scratch", "annular face")
        run_worker.return_value = {
            "status": "GENERATED_BY_PERSISTENT_WORKER", "worker_mode": "PERSISTENT_PROCESS",
            "worker_started_utc": "2026-11-24T00:00:00Z", "prompt_token_count": 90,
            "generate_ms": 1000, "generated_text": self.rules["checks"]["scratch"][0],
            "runtime_evidence": {"artifact": {"sha256": "a" * 64},
                                 "tokenizer": {"sha256": "b" * 64}},
        }
        pins.return_value = ({"c" * 64}, {"b" * 64})
        result = advise(row, self.rules, Path("test.sock"), 8, llama_root=Path("runtime"))
        self.assertEqual(result["status"], "CHECKLIST_FALLBACK")
        self.assertNotEqual(result["llama_runtime_mode"], "PERSISTENT_WORKER")
        pins.return_value = ({"a" * 64}, {"b" * 64})
        result = advise(row, self.rules, Path("test.sock"), 8, llama_root=Path("runtime"))
        self.assertEqual(result["status"], "LLAMA_ADVICE")
        pins.assert_called_with(Path("runtime"))

    @patch("forgesight.grounded_advice.assist_edge_worker.request")
    def test_malformed_worker_reports_keep_the_checklist_fallback(self, run_worker):
        for reply in (["bad report"], {"runtime_evidence": ["bad evidence"]},
                      {"runtime_evidence": {"artifact": ["bad artifact"]}}):
            with self.subTest(reply=reply):
                run_worker.return_value = reply
                result = advise(self.record, self.rules, Path("test.sock"), 8)
                self.assertEqual(result["status"], "CHECKLIST_FALLBACK")
