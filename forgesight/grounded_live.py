"""Optional bridge from saved Jetson inspections to the grounded advisory."""

import json
import os
from pathlib import Path
from uuid import uuid4

from .grounded_advice import RULES, advise, read_json


class GroundedLive:
    """One background advisory at a time, outside the vision result path."""

    def __init__(self, queue, report_root, worker_socket, timeout,
                 worker_unhealthy=None, llama_root=None):
        self.queue = queue
        self.report_root = Path(report_root)
        self.worker_socket = worker_socket
        self.timeout = min(timeout, 8)
        self.worker_unhealthy = worker_unhealthy
        self.llama_root = llama_root
        self.rules = read_json(RULES)
        self.latest_job = None

    def submit(self, record, worker_ready):
        if (record["status"] != "ANALYZED"
                or record["candidate"] != "ANOMALY_CANDIDATE"
                or record["decision"]["grade"] not in ("REVIEW", "REJECT")):
            return
        future = self.queue.submit(lambda: self._run(record, worker_ready))
        self.latest_job = (record["inspection_id"], future)

    def _run(self, record, worker_ready):
        result = advise(record, self.rules, self.worker_socket if worker_ready() else None,
                        self.timeout, llama_root=self.llama_root)
        if result.get("persistent_worker_unhealthy") and self.worker_unhealthy is not None:
            self.worker_unhealthy()
        self.report_root.mkdir(parents=True, exist_ok=True)
        target = self.report_root / (record["inspection_id"] + ".json")
        temporary = target.with_suffix("." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return result

    def state(self, latest, worker_available):
        mode = {"llama_runtime": "NOT_RECORDED",
                "advice_worker_available_now": worker_available}
        state = {"status": "NOT_ELIGIBLE", "advisory": None, **mode}
        job = self.latest_job
        if job is None or latest is None or latest.get("inspection_id") != job[0]:
            return state
        _, future = job
        if future is None:
            return {**state, "status": "ADVICE_BUSY"}
        if not future.done():
            return {**state, "status": "PENDING"}
        result = future.result()
        if result["status"] == "INSPECTION_WITHHELD":
            return state
        return {**state, "status": result["status"],
                "llama_runtime": result.get("llama_runtime_mode", "NOT_RECORDED"),
                "worker": result.get("worker"),
                "advisory": {"note": result["advice"]} if "advice" in result else None,
                "elapsed_ms": result.get("advice_elapsed_ms")}
