#!/usr/bin/env python3
"""Read-only ForgeSight Jetson diagnostic bundle collector.

Collects deployment evidence for the composed-advice prompt-budget failure
(207 prompt tokens vs the exported prefill bound) into a new timestamped
directory. Nothing is installed, modified, generated or sent anywhere; the
script only reads and copies files into the bundle. No model is loaded and
no generation runs. Two modes:

1) Inventory mode (any Python 3.8+):
     /usr/bin/python3 forgesight_diag_collect.py --checkout /path/to/ForgeSight \
         --llama-root "$HOME/.local/share/forgesight/llama_cpu_v1" \
         [--report /path/to/advice_report.json]...

2) Token-count mode (pinned runtime venv python; loads ONLY the tokenizer):
     "$HOME/.local/share/forgesight/llama_cpu_v1/venv/bin/python" \
         forgesight_diag_collect.py --checkout /path/to/ForgeSight \
         --llama-root "$HOME/.local/share/forgesight/llama_cpu_v1" \
         --record /path/to/saved/record.json

Mode 2 re-derives the composed prompt for the saved record with the merged
code and counts tokens with the pinned tokenizer, so the fix can be sized
against the deployment instead of estimated. If the venv python differs,
use the interpreter that can import pytorch_tokenizers on the device.
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


def copy_small_json(source, directory):
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
    target = directory / source.name
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


def collect_inventory(args, bundle, git_payload):
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

    root = Path(args.llama_root) / "model"
    names = ["edge_llama.json", "llama3_2_1b_xnnpack.pte", "tokenizer.model",
             "llama3_2_1b_export.json", "llama3_2_1b_assets.json"]
    assets = {name: file_entry(root / name) for name in names}
    write_json(bundle, "deployed_llama_assets.json", assets)
    # The small profile/export JSONs define the actual prefill and context
    # budgets; copy them verbatim so the fix is sized on the real values.
    copied = {name: copy_small_json(root / name, bundle)
              for name in ("edge_llama.json", "llama3_2_1b_export.json",
                           "llama3_2_1b_assets.json")}
    write_json(bundle, "copied_profiles.json", copied)

    reports = [copy_small_json(source, bundle) for source in args.report]
    write_json(bundle, "copied_reports.json", reports)


def collect_token_budget(args, bundle):
    checkout = Path(args.checkout).resolve()
    sys.path.insert(0, str(checkout))
    try:
        from pytorch_tokenizers import CppTiktoken
    except ImportError:
        write_json(bundle, "token_budget.json", {
            "error": "pytorch_tokenizers not importable with this interpreter",
            "hint": "rerun this mode with the pinned runtime venv python"})
        return
    from forgesight.assist_contract import prepare_advisory
    from forgesight.edge_jetson_smoke import chat_prompt

    tokenizer = CppTiktoken()
    tokenizer.load(str(Path(args.llama_root).resolve() / "model" / "tokenizer.model"))

    def count(text):
        return len(tokenizer.encode(text, 0, 0))

    record = json.loads(Path(args.record).read_text())
    # prepare_advisory pins the guidance by canonical hash, so a wrong or
    # edited document fails closed here instead of miscounting.
    guidance = json.loads((checkout / "guidance" / "prototype_inspection_v2.json").read_text())
    profile = json.loads((Path(args.llama_root) / "model" / "edge_llama.json").read_text())
    try:
        context = prepare_advisory(record, guidance)
    except Exception as exc:  # the refusal reason is itself useful evidence
        write_json(bundle, "token_budget.json", {
            "prepare_advisory_refusal": "{}: {}".format(type(exc).__name__, exc)})
        return
    prompt_text = context["prompt_text"]
    system_text = context["system_text"]
    formatted = chat_prompt(prompt_text, system_text)
    full = count(formatted)
    prompt_only = count(prompt_text)
    system_only = count(system_text)
    max_new = context.get("max_new_tokens", 64)
    result = {
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
    write_json(bundle, "token_budget.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", required=True,
                        help="Path to the ForgeSight checkout under test (read-only).")
    parser.add_argument("--llama-root", required=True,
                        help="Deployed Llama runtime root containing model/.")
    parser.add_argument("--report", action="append", default=[],
                        help="Existing advice/verification report JSON to copy in (repeatable).")
    parser.add_argument("--record", default=None,
                        help="Saved inspection record.json: enables token-count mode.")
    args = parser.parse_args()
    # Capture git state BEFORE creating the bundle directory, so the status
    # snapshot is not polluted by the bundle itself when run from a checkout.
    git_payload = git_checkout_payload(Path(args.checkout))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle = Path("forgesight_diag_{}".format(stamp))
    if bundle.exists():
        parser.error("{} already exists; wait a second and rerun.".format(bundle))
    bundle.mkdir()
    collect_inventory(args, bundle, git_payload)
    if args.record:
        collect_token_budget(args, bundle)
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
