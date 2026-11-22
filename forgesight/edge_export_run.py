"""Run one pinned re-export into a fresh attempt directory; never overwrite.

Wraps the documented EDGE_RUNTIME.md 2.5/2.6 flow in one checked step:
readiness pins (ExecuTorch checkout commit, recorded package versions, licensed
weight hashes against the assets manifest), recipe rendering, the export
subprocess with retained logs and resource facts, artifact inspection, and a
staged export manifest written inside the new attempt directory only. Repository
manifests are updated by a separate reviewed commit from these staged facts;
this script never edits them, so the existing deployed hash chain keeps working
until the new chain is committed coherently.

Typical use in the pinned export environment (documented WSL context):
    /root/.cache/forgesight/et1.5.1/venv/bin/python -B -m forgesight.edge_export_run \
        --checkout /root/.cache/forgesight/et1.5.1/executorch \
        --output artifacts/edge/export_attempt_02
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from forgesight.edge_assets import fingerprint
from forgesight.edge_recipe import create_recipe

PACKAGE_PINS = ('torch', 'executorch', 'torchao')
RSS_SAMPLE_SECONDS = 0.1


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'NOT_INSTALLED'


def _file_sha256(path):
    return fingerprint(Path(path))[1]


def readiness(profile, reference, checkout, weights, assets, output):
    """Return the list of blockers; empty means the export may start."""
    blockers = []
    if output.exists():
        return [f'Attempt output exists: {output} (attempts are never reused)']
    for name in PACKAGE_PINS:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        expected = reference['packages'].get(name)
        if installed != expected:
            blockers.append(f'{name} {installed!r} installed, need the recorded {expected!r}')
    if not (checkout / '.git').exists():
        blockers.append(f'No pinned ExecuTorch checkout at {checkout}')
    else:
        result = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=15, check=False)
        if result.returncode != 0 or result.stdout.strip() != profile['executorch_commit']:
            blockers.append('ExecuTorch checkout is not at the pinned commit '
                            + profile['executorch_commit'])
    for name in profile['source_files']:
        path = weights / name
        if not path.is_file():
            blockers.append(f'Missing licensed model file: {name}')
            continue
        if fingerprint(path)[1] != assets['files'][name]['sha256']:
            blockers.append(f'{name} sha256 differs from the pinned assets manifest')
    return blockers


def _sample_rss(pid, samples, stop):
    while not stop.is_set():
        try:
            with open(f'/proc/{pid}/status') as stream:
                for line in stream:
                    if line.startswith('VmHWM:'):
                        samples.append(int(line.split()[1]))
                        break
        except OSError:
            return
        stop.wait(RSS_SAMPLE_SECONDS)


def run_export(config_path, checkout, attempt_dir):
    argv = [sys.executable, '-B', '-m', 'extension.llm.export.export_llm',
            '--config', str(config_path)]
    env = dict(os.environ, OMP_NUM_THREADS='8')
    started = time.monotonic()
    with open(attempt_dir / 'export.stdout.log', 'xb') as out, \
            open(attempt_dir / 'export.stderr.log', 'xb') as err:
        process = subprocess.Popen(argv, cwd=checkout, stdout=out, stderr=err, env=env)
        samples, stop = [], threading.Event()
        sampler = threading.Thread(target=_sample_rss, args=(process.pid, samples, stop),
                                   daemon=True)
        sampler.start()
        returncode = process.wait()
        stop.set()
        sampler.join(timeout=2)
    return {'argv': argv, 'returncode': returncode, 'completion_reason': 'EXITED',
            'process_wall_seconds': time.monotonic() - started,
            'sampled_peak_rss_kib': max(samples) if samples else None,
            'rss_sampling_interval_seconds': RSS_SAMPLE_SECONDS,
            'memory_note': 'Sampled process VmHWM; may miss peaks and excludes other processes.',
            'stdout_bytes': (attempt_dir / 'export.stdout.log').stat().st_size,
            'stderr_bytes': (attempt_dir / 'export.stderr.log').stat().st_size,
            'export_working_directory': str(checkout), 'omp_num_threads': 8}


def build_manifest(profile, profile_sha, assets_sha, pylock_sha, config_key, config_sha,
                   process, pte_path, pte_relative, validation):
    size, digest = fingerprint(pte_path)
    return {
        'schema_version': 1,
        'status': 'EXPORTED_NOT_INFERENCE_VERIFIED',
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'executorch_commit': profile['executorch_commit'],
        'backend_requested': profile['backend'],
        'quantization': {'qmode': profile['quantization'], 'group_size': profile['group_size'],
                         'embedding_quantize': profile['embedding_quantize']},
        'max_seq_length': profile['max_seq_length'],
        'max_context_length': profile['max_context_length'],
        'artifact': {'bytes': size, 'sha256': digest, 'local_relative_path': pte_relative},
        'packages': {name: _package_version(name) for name in
                     ('torch', 'torchvision', 'executorch', 'torchao', 'torchtune', 'transformers')},
        'inputs_sha256': {'configs/edge_llama.json': profile_sha,
                          'manifests/llama3_2_1b_assets.json': assets_sha,
                          'requirements/pylock.export.toml': pylock_sha,
                          config_key: config_sha},
        'export_process': process,
        'device_inference': 'BLOCKED_NOT_VERIFIED',
        'model_inference': 'NOT_RUN',
        'quality': 'NOT_VERIFIED',
        'offline_operation': 'NOT_VERIFIED',
        'artifact_validation': validation,
        'file_identifier': 'ET12',
    }


def export_attempt(profile_path, schema_path, reference_path, checkout, weights,
                   assets_path, pylock_path, output):
    profile = json.loads(Path(profile_path).read_text())
    reference = json.loads(Path(reference_path).read_text())
    assets = json.loads(Path(assets_path).read_text())
    blockers = readiness(profile, reference, Path(checkout), Path(weights), assets, Path(output))
    if blockers:
        return {'status': 'BLOCKED_NOT_RUN', 'blockers': blockers}
    output = Path(output)
    recipe = create_recipe(profile, json.loads(Path(schema_path).read_text()),
                           Path(checkout), Path(weights), output)
    config_path = output / 'export_config.yaml'
    process = run_export(config_path, Path(checkout), output)
    result = {'status': 'EXPORT_FAILED_RETAINED_LOGS', 'recipe_status': recipe['status'],
              'export_process': process}
    if process['returncode'] != 0:
        return result
    pte_path = output / 'exported/llama3_2_1b_xnnpack.pte'
    if not pte_path.is_file():
        result['error'] = f'Export exited 0 but no artifact at {pte_path}'
        return result
    size, digest = fingerprint(pte_path)
    reference_draft = {'schema_version': 1, 'model_id': profile['model_id'],
                       'model_revision': profile['model_revision'],
                       'executorch_commit': profile['executorch_commit'],
                       'artifact': {'bytes': size, 'sha256': digest}}
    try:
        from forgesight.edge_artifact import inspect_artifact
        validation = inspect_artifact(pte_path, reference_draft, profile)
    except importlib.metadata.PackageNotFoundError:
        validation = {'status': 'NOT_RUN_IN_THIS_ENV', 'method_execution': 'NOT_RUN'}
    except Exception as error:
        result['status'] = 'EXPORT_FAILED_ARTIFACT_REJECTED'
        result['artifact_validation'] = {'status': 'FAILED_NOT_VERIFIED',
                                         'error': f'{type(error).__name__}: {error}',
                                         'method_execution': 'NOT_RUN'}
        return result
    manifest = build_manifest(
        profile, _file_sha256(profile_path), _file_sha256(assets_path),
        _file_sha256(pylock_path), os.path.relpath(config_path), _file_sha256(config_path),
        process, pte_path, os.path.relpath(pte_path), validation)
    manifest_path = output / 'llama3_2_1b_export.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    result.update({'status': 'EXPORTED_STAGED_NOT_COMMITTED',
                   'staged_manifest': str(manifest_path)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path('configs/edge_llama.json'))
    parser.add_argument('--schema', type=Path, default=Path('configs/edge_upstream_schema.json'))
    parser.add_argument('--assets', type=Path, default=Path('manifests/llama3_2_1b_assets.json'))
    parser.add_argument('--reference', type=Path, default=Path('manifests/llama3_2_1b_export.json'))
    parser.add_argument('--pylock', type=Path, default=Path('requirements/pylock.export.toml'))
    parser.add_argument('--checkout', type=Path, default=Path('data/tools/executorch'))
    parser.add_argument('--weights', type=Path, default=Path('models/llama3_2_1b'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/edge/export_attempt_02'))
    args = parser.parse_args()
    result = export_attempt(args.profile, args.schema, args.reference, args.checkout,
                            args.weights, args.assets, args.pylock, args.output)
    summary = {'status': result['status']}
    if 'blockers' in result:
        summary['blockers'] = result['blockers']
    if 'staged_manifest' in result:
        summary['staged_manifest'] = result['staged_manifest']
        summary['next'] = ('Review the staged manifest, commit it as '
                           'manifests/llama3_2_1b_export.json together with the profile '
                           'change in ONE coherent commit, then stage the device update.')
    print(json.dumps(summary, indent=2))
    return 0 if result['status'] == 'EXPORTED_STAGED_NOT_COMMITTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
