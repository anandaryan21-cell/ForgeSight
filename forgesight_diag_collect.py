#!/usr/bin/env python3
"""Read-only ForgeSight Jetson diagnostic bundle collector.

Collects deployment evidence for the composed-advice prompt-budget failure
(207 prompt tokens vs the exported prefill bound) into a new timestamped
directory. Nothing is installed, modified, generated or sent anywhere; the
script only reads and copies files into the bundle. No model is loaded and
no generation runs; token counts are produced by re-running this script with
the pinned runtime venv python, which loads ONLY the tokenizer.

One command from the ForgeSight folder, no flags, no path editing:

  python3 forgesight_diag_collect.py

The checkout defaults to the current working directory, so run it from the
ForgeSight folder (the script itself may live in the repo root or in /tmp).

What it does automatically:
- finds the deployed Llama runtime under ~/.local/share/forgesight: exactly
  one Stage 2.9-marked install carrying the live profile is selected; zero
  or several matches are reported in the bundle, never guessed;
- inventories device, git state and deployed asset hashes, and copies the
  small profile/export JSONs that carry the real prefill/context budgets;
- discovers saved inspection record.json files under the runtime's sibling
  runs/ directories and the checkout's runs/ directory (bounded, newest
  first) and copies the small ones into the bundle;
- when the pinned runtime venv exists, re-runs itself per discovered record
  with that interpreter to count the composed prompt with the pinned
  tokenizer against the deployed budget; refusals are recorded as evidence;
- writes missing_items.json listing everything it could not collect and why
  (files known to live on the PC are marked unavailable, not fetched).

Optional manual overrides: --llama-root, --record (single-record token mode
with the venv python), --report (copy a specific small JSON in).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import zipfile

sys.dont_write_bytecode = True  # never write __pycache__ into the checkout
os.environ["GIT_OPTIONAL_LOCKS"] = "0"  # git reads must not take lock files

MAX_COPY_BYTES = 2 * 1024 * 1024  # report/asset JSONs are small; refuse bulk
MAX_RECORDS = 8  # bound on discovered records counted/copied per run
MAX_EXTRA_JSON = 8  # bound on extra small run JSONs copied per run
CHILD_TIMEOUT_SECONDS = 180
STAGE_2_9_PURPOSE = "FORGESIGHT_STAGE_2_9_CPU_RUNTIME"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "present": False}
    stat = path.stat()
    return {"path": str(path), "present": True, "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256(path)}


def write_json(directory, name, payload):
    target = directory / name
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def copy_small_json(source, directory, target_name=None):
    source = Path(source)
    if not source.is_file():
        return {"source": str(source), "copied": False, "reason": "not a file"}
    if source.stat().st_size > MAX_COPY_BYTES:
        return {"source": str(source), "copied": False, "reason": "over 2 MiB cap"}
    try:
        json.loads(source.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        return {"source": str(source), "copied": False,
                "reason": "not parseable JSON: {}".format(exc)}
    target = directory / (target_name or source.name)
    if target.exists():
        return {"source": str(source), "copied": False, "reason": "name already in bundle"}
    shutil.copyfile(str(source), str(target))
    return {"source": str(source), "copied": True, "bundle_file": target.name,
            "sha256": sha256(target)}


def git_value(checkout, argv):
    try:
        result = subprocess.run(["git", "-C", str(checkout)] + argv,
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": "{}: {}".format(type(exc).__name__, exc)}
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()[:2000]}


def git_checkout_payload(checkout):
    # No remote listing: remote URLs can carry embedded credentials.
    return {
        "checkout": str(checkout),
        "head": git_value(checkout, ["rev-parse", "HEAD"]),
        "tree": git_value(checkout, ["rev-parse", "HEAD^{tree}"]),
        "status_porcelain": git_value(checkout, ["status", "--porcelain"])}


def assess_llama_root(candidate):
    """What validates or disqualifies one possible runtime root (read-only)."""
    candidate = Path(candidate)
    reasons = {"candidate": str(candidate), "is_dir": candidate.is_dir()}
    marker = candidate / ".forgesight-runtime.json"
    reasons["marker_present"] = marker.is_file()
    if marker.is_file():
        try:
            marker_doc = json.loads(marker.read_text())
            purpose = marker_doc.get("purpose")
        except (ValueError, OSError) as exc:
            marker_doc = {}
            purpose = None
            reasons["marker_parse_error"] = str(exc)
        reasons["marker_purpose"] = purpose
        reasons["marker_matches_stage_2_9"] = (purpose == STAGE_2_9_PURPOSE)
        reasons["marker_prefix_matches"] = (marker_doc.get("prefix") in (None, str(candidate)))
    # Readiness predicate (same shape as the app's own check, plus the files
    # token mode needs): live profile, PTE, tokenizer and the venv python.
    reasons["profile_present"] = (candidate / "model" / "edge_llama.json").is_file()
    reasons["pte_present"] = (candidate / "model" / "llama3_2_1b_xnnpack.pte").is_file()
    reasons["tokenizer_present"] = (candidate / "model" / "tokenizer.model").is_file()
    reasons["venv_python_present"] = (candidate / "venv" / "bin" / "python").is_file()
    reasons["validated"] = bool(reasons.get("marker_matches_stage_2_9")
                                and reasons.get("marker_prefix_matches", True)
                                and reasons["profile_present"] and reasons["pte_present"]
                                and reasons["tokenizer_present"] and reasons["venv_python_present"])
    return reasons


def resolve_llama_root(args):
    """Pick the deployed Llama runtime root without inventing paths.

    An explicit --llama-root always wins. Otherwise look only at the
    documented data directory's direct children and accept exactly one
    Stage 2.9-marked installation that carries the live profile. Zero or
    several validated installs are reported, never guessed.
    """
    if args.llama_root:
        explicit = Path(args.llama_root)
        return explicit, {"mode": "explicit", "selected": str(explicit),
                          "assessment": assess_llama_root(explicit)}
    data_dir = Path.home() / ".local" / "share" / "forgesight"
    discovery = {"mode": "auto", "search_dir": str(data_dir),
                 "search_dir_present": data_dir.is_dir(), "candidates": []}
    validated = []
    if data_dir.is_dir():
        for child in sorted(data_dir.iterdir()):
            if not child.is_dir():
                continue
            assessment = assess_llama_root(child)
            discovery["candidates"].append(assessment)
            if assessment["validated"]:
                validated.append(child)
    if len(validated) == 1:
        discovery["status"] = "SELECTED_UNIQUE_VALIDATED"
        discovery["selected"] = str(validated[0])
        return validated[0], discovery
    discovery["status"] = "NONE_VALIDATED" if not validated else "AMBIGUOUS_NOT_SELECTED"
    discovery["selected"] = None
    return None, discovery


def discover_saved_runs(args):
    """Find saved inspection records and small run JSONs in known run roots.

    Bounded: direct runs/ children of each directory under
    ~/.local/share/forgesight, the checkout's own runs/ directory, and the
    feature-audit space ~/forgesight-feature-audits/*/*/records plus
    */advice (the layout observed on the actual device); newest records
    first, capped, JSON only, never images or bulk files.
    """
    run_roots = []
    data_dir = Path.home() / ".local" / "share" / "forgesight"
    if data_dir.is_dir():
        for child in sorted(data_dir.iterdir()):
            if (child / "runs").is_dir():
                run_roots.append(child / "runs")
    checkout_runs = Path(args.checkout) / "runs"
    if checkout_runs.is_dir():
        run_roots.append(checkout_runs)
    records = []
    for runs in run_roots:
        for record in runs.glob("*/record.json"):
            if record.is_file():
                records.append(record)
    audit_space = Path.home() / "forgesight-feature-audits"
    audit_records = []
    audit_reports = []
    if audit_space.is_dir():
        audit_records = [path for path in
                         audit_space.glob("*/*/records/*/result/record.json")
                         if path.is_file()]
        audit_reports = [path for path in
                         audit_space.glob("*/*/advice/*/advice_report.json")
                         if path.is_file()]
    records = sorted(set(records) | set(audit_records),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    extra_jsons = list(audit_reports)
    for record in records[:MAX_RECORDS]:
        for sibling in sorted(record.parent.glob("*.json")):
            if sibling != record and sibling.is_file():
                extra_jsons.append(sibling)
    return {"run_roots": [str(root) for root in run_roots],
            "audit_space": str(audit_space),
            "audit_space_present": audit_space.is_dir(),
            "records": [str(record) for record in records[:MAX_RECORDS]],
            "records_found_total": len(records),
            "extra_jsons": [str(path) for path in extra_jsons[:MAX_EXTRA_JSON]]}


def collect_inventory(args, bundle, git_payload, llama_root, discovery, runs, missing):
    device = {"utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version,
              "platform": platform.platform(),
              "machine": platform.machine()}
    model_file = Path("/proc/device-tree/model")
    device["device_tree_model"] = (
        model_file.read_bytes().decode(errors="replace").rstrip("\x00")
        if model_file.is_file() else None)
    write_json(bundle, "device.json", device)

    write_json(bundle, "git_checkout.json", git_payload)

    write_json(bundle, "llama_root_discovery.json", discovery)
    if discovery.get("mode") == "auto" and discovery.get("status") != "SELECTED_UNIQUE_VALIDATED":
        missing.append({"item": "deployed Llama runtime root",
                        "status": discovery.get("status"),
                        "detail": "see llama_root_discovery.json candidates"})
    if llama_root is None:
        write_json(bundle, "deployed_llama_assets.json", {
            "error": "no unique validated Llama runtime root; see llama_root_discovery.json"})
        write_json(bundle, "copied_profiles.json", {
            "error": "no unique validated Llama runtime root; see llama_root_discovery.json"})
    else:
        root = Path(llama_root) / "model"
        names = ["edge_llama.json", "llama3_2_1b_xnnpack.pte", "tokenizer.model",
                 "llama3_2_1b_export.json", "llama3_2_1b_assets.json"]
        assets = {name: file_entry(root / name) for name in names}
        write_json(bundle, "deployed_llama_assets.json", assets)
        for name in names:
            if not assets[name]["present"]:
                missing.append({"item": "deployed asset {}".format(name),
                                "status": "MISSING", "detail": str(root / name)})
        # The small profile/export JSONs define the actual prefill and context
        # budgets; copy them verbatim so the fix is sized on the real values.
        copied = {name: copy_small_json(root / name, bundle)
                  for name in ("edge_llama.json", "llama3_2_1b_export.json",
                               "llama3_2_1b_assets.json")}
        write_json(bundle, "copied_profiles.json", copied)
        missing.append({
            "item": "PTE internal metadata (get_max_seq_len etc.)",
            "status": "NOT_COLLECTED",
            "detail": "reading it would require loading the program; the "
                      "recorded export manifest values are copied instead"})

    write_json(bundle, "saved_runs_discovery.json", runs)
    if not runs["records"]:
        missing.append({"item": "saved inspection record.json files",
                        "status": "NONE_FOUND",
                        "detail": "searched: {}".format(runs["run_roots"])})
    # Unique bundle names: every run's record is record.json, and sibling run
    # JSONs share names across runs; the position prefix keeps every copy and
    # ties it to the token_budget_NN.json of the same position. The
    # source/bundle_file pairs are the provenance map.
    copied_records = [copy_small_json(Path(record), bundle,
                                      "record_{:02d}.json".format(position))
                      for position, record in enumerate(runs["records"])]
    copied_extra = [copy_small_json(Path(path), bundle,
                                    "runjson_{:02d}_{}".format(position, Path(path).name))
                    for position, path in enumerate(runs["extra_jsons"])]
    copied_reports = [copy_small_json(source, bundle) for source in args.report]
    write_json(bundle, "copied_reports.json",
               {"records": copied_records, "run_jsons": copied_extra,
                "explicit": copied_reports})

    budget_candidates = [Path(record.parent) / "composed_prompt_budget.json"
                         for record in map(Path, runs["records"])]
    budget_candidates.append(Path(args.checkout) / "composed_prompt_budget.json")
    if not any(path.is_file() for path in budget_candidates):
        missing.append({"item": "composed_prompt_budget.json (original PC measurement)",
                        "status": "NOT_ON_DEVICE",
                        "detail": "expected on the Windows PC; cannot be fetched from the Jetson"})


def token_budget_payload(args):
    checkout = Path(args.checkout).resolve()
    sys.path.insert(0, str(checkout))
    try:
        from pytorch_tokenizers import CppTiktoken
    except ImportError:
        return {"error": "pytorch_tokenizers not importable with this interpreter",
                "hint": "rerun this mode with the pinned runtime venv python"}
    from forgesight.assist_contract import prepare_advisory
    from forgesight.edge_jetson_smoke import chat_prompt

    tokenizer = CppTiktoken()
    tokenizer.load(str(Path(args.resolved_llama_root).resolve() / "model" / "tokenizer.model"))

    def count(text):
        return len(tokenizer.encode(text, 0, 0))

    record = json.loads(Path(args.record).read_text())
    # prepare_advisory pins the guidance by canonical hash, so a wrong or
    # edited document fails closed here instead of miscounting.
    guidance = json.loads((checkout / "guidance" / "prototype_inspection_v2.json").read_text())
    profile = json.loads((Path(args.resolved_llama_root) / "model" / "edge_llama.json").read_text())
    try:
        context = prepare_advisory(record, guidance)
    except Exception as exc:  # the refusal reason is itself useful evidence
        return {"record": str(args.record),
                "prepare_advisory_refusal": "{}: {}".format(type(exc).__name__, exc)}
    prompt_text = context["prompt_text"]
    system_text = context["system_text"]
    formatted = chat_prompt(prompt_text, system_text)
    full = count(formatted)
    prompt_only = count(prompt_text)
    system_only = count(system_text)
    max_new = context.get("max_new_tokens", 64)
    return {
        "note": "Tokenizer-only measurement; no model was loaded and no "
                "generation ran. Part counts are exact for each text; the "
                "template overhead is full-minus-parts and approximate "
                "because of BPE boundary effects.",
        "record": str(args.record),
        "prompt_tokens_full_formatted": full,
        "prompt_tokens_prompt_text_only": prompt_only,
        "prompt_tokens_system_text_only": system_only,
        "approx_template_overhead": full - prompt_only - system_only,
        "composed_max_new_tokens": max_new,
        "deployed_max_seq_length": profile.get("max_seq_length"),
        "deployed_max_context_length": profile.get("max_context_length"),
        "prefill_check_prompt_lt_max_seq_length": (full < profile["max_seq_length"])
        if isinstance(profile.get("max_seq_length"), int) else None,
        "context_check_prompt_plus_new_le_max_context":
        (full + max_new <= profile["max_context_length"])
        if isinstance(profile.get("max_context_length"), int) else None,
        "prompt_text": prompt_text,
        "system_text": system_text,
    }


def run_token_children(args, bundle, llama_root, runs, missing):
    """Re-run this script with the pinned venv python for each saved record.

    Tokenizer-only: the child imports pytorch_tokenizers and the repo's
    contract code, never the model. Failures are recorded, never fatal.
    """
    index = {"child_python": None, "results": []}
    venv_python = Path(llama_root) / "venv" / "bin" / "python"
    if not runs["records"]:
        index["skipped"] = "no saved records discovered"
        write_json(bundle, "token_budget_index.json", index)
        return
    if not venv_python.is_file():
        missing.append({"item": "pinned runtime venv python",
                        "status": "MISSING",
                        "detail": "{} not found; token counts not collected".format(venv_python)})
        index["skipped"] = "venv python missing"
        write_json(bundle, "token_budget_index.json", index)
        return
    index["child_python"] = str(venv_python)
    for position, record in enumerate(runs["records"]):
        out_name = "token_budget_{:02d}.json".format(position)
        command = [str(venv_python), "-B", str(Path(__file__).resolve()),
                   "--child-token-mode", "--checkout", str(Path(args.checkout).resolve()),
                   "--llama-root", str(llama_root), "--record", record,
                   "--out", str(bundle / out_name)]
        entry = {"record": record, "output": out_name}
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=CHILD_TIMEOUT_SECONDS)
            entry["returncode"] = result.returncode
            if result.returncode != 0:
                entry["stderr_tail"] = result.stderr.strip()[-2000:]
                missing.append({"item": "token count for {}".format(record),
                                "status": "CHILD_FAILED",
                                "detail": entry["stderr_tail"][-500:]})
        except (OSError, subprocess.SubprocessError) as exc:
            entry["error"] = "{}: {}".format(type(exc).__name__, exc)
            missing.append({"item": "token count for {}".format(record),
                            "status": "CHILD_FAILED", "detail": entry["error"]})
        index["results"].append(entry)
    write_json(bundle, "token_budget_index.json", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", default=None,
                        help="Path to the ForgeSight checkout under test (read-only). "
                             "Defaults to the current working directory.")
    parser.add_argument("--llama-root", default=None,
                        help="Deployed Llama runtime root containing model/. "
                             "Optional: without it, exactly one Stage 2.9-validated "
                             "install under ~/.local/share/forgesight is selected; "
                             "zero or several matches are reported, never guessed.")
    parser.add_argument("--report", action="append", default=[],
                        help="Existing advice/verification report JSON to copy in (repeatable).")
    parser.add_argument("--record", default=None,
                        help="Saved inspection record.json: single-record token-count mode "
                             "(run with the pinned venv python).")
    parser.add_argument("--child-token-mode", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.checkout is None:
        args.checkout = Path.cwd()

    if args.child_token_mode:
        if not (args.llama_root and args.record and args.out):
            parser.error("--child-token-mode needs --llama-root, --record and --out.")
        args.resolved_llama_root = Path(args.llama_root)
        Path(args.out).write_text(json.dumps(token_budget_payload(args), indent=2,
                                             sort_keys=True) + "\n")
        return

    llama_root, discovery = resolve_llama_root(args)
    args.resolved_llama_root = llama_root
    # Capture git state BEFORE creating the bundle directory, so the status
    # snapshot is not polluted by the bundle itself when run from a checkout.
    git_payload = git_checkout_payload(Path(args.checkout))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle = Path("forgesight_diag_{}".format(stamp))
    if bundle.exists():
        parser.error("{} already exists; wait a second and rerun.".format(bundle))
    bundle.mkdir()
    missing = []
    runs = discover_saved_runs(args)
    collect_inventory(args, bundle, git_payload, llama_root, discovery, runs, missing)
    if discovery.get("status") in ("NONE_VALIDATED", "AMBIGUOUS_NOT_SELECTED"):
        print("NOTE: {} - details in llama_root_discovery.json inside the bundle."
              .format(discovery["status"]))
    if args.record:
        if llama_root is None:
            write_json(bundle, "token_budget.json", {
                "error": "no unique validated Llama runtime root; see llama_root_discovery.json"})
        else:
            write_json(bundle, "token_budget.json", token_budget_payload(args))
    elif llama_root is not None:
        run_token_children(args, bundle, llama_root, runs, missing)
    write_json(bundle, "missing_items.json", {"missing": missing,
                                              "collected_everything": not missing})
    # Pack the whole bundle into one zip for chat attachment; only files
    # this run created inside the fresh bundle directory are included.
    archive = Path("{}.zip".format(bundle))
    if archive.exists():
        parser.error("{} already exists; not overwriting.".format(archive))
    with zipfile.ZipFile(str(archive), "w", zipfile.ZIP_DEFLATED) as packed:
        for item in sorted(bundle.iterdir()):
            packed.write(str(item), arcname="{}/{}".format(bundle.name, item.name))
    print("Bundle directory: {}".format(bundle.resolve()))
    print("Attach this file: {}".format(archive.resolve()))


if __name__ == "__main__":
    main()
