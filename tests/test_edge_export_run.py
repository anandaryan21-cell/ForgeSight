"""Host tests for the pinned re-export runner: readiness gates, non-overwrite,
staged manifest shape and hash fidelity. No real export runs here."""

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from forgesight import edge_export_run
from forgesight.edge_export_run import export_attempt, readiness

REPO = Path(__file__).resolve().parent.parent


def make_world(root):
    root = Path(root)
    checkout = root / 'et'
    checkout.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=checkout, check=True)
    (checkout / 'marker').write_text('x')
    subprocess.run(['git', 'add', '-A'], cwd=checkout, check=True)
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x'],
                   cwd=checkout, check=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=checkout, check=True,
                            capture_output=True, text=True).stdout.strip()
    profile = json.loads((REPO / 'configs/edge_llama.json').read_text())
    profile['executorch_commit'] = commit
    weights = root / 'weights'
    (weights / 'original').mkdir(parents=True)
    assets = {'files': {}}
    for name in profile['source_files']:
        content = ('weights:' + name).encode()
        (weights / name).write_bytes(content)
        assets['files'][name] = {'sha256': hashlib.sha256(content).hexdigest(),
                                 'bytes': len(content)}
    reference = {'packages': {}, 'inputs_sha256': {}}
    for name in edge_export_run.PACKAGE_PINS:
        try:
            reference['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass  # absent pin matches absent install
    schema = json.loads((REPO / 'configs/edge_upstream_schema.json').read_text())
    schema['executorch_commit'] = commit
    return profile, reference, checkout, weights, assets, schema


class ReadinessTests(unittest.TestCase):
    def test_existing_output_is_the_only_blocker_reported_first(self):
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, _ = make_world(tmp)
            output = Path(tmp) / 'attempt'
            output.mkdir()
            blockers = readiness(profile, reference, checkout, weights, assets, output)
            self.assertEqual(len(blockers), 1)
            self.assertIn('never reused', blockers[0])

    def test_clean_world_has_no_blockers(self):
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, _ = make_world(tmp)
            self.assertEqual(readiness(profile, reference, checkout, weights, assets,
                                       Path(tmp) / 'new'), [])

    def test_wrong_checkout_commit_blocks(self):
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, _ = make_world(tmp)
            profile['executorch_commit'] = '0' * 40
            blockers = readiness(profile, reference, checkout, weights, assets, Path(tmp) / 'n')
            self.assertTrue(any('pinned commit' in b for b in blockers))

    def test_missing_and_mismatched_weights_block(self):
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, _ = make_world(tmp)
            (weights / 'original/params.json').unlink()
            (weights / 'original/tokenizer.model').write_bytes(b'tampered')
            blockers = readiness(profile, reference, checkout, weights, assets, Path(tmp) / 'n')
            self.assertTrue(any('Missing licensed model file' in b for b in blockers))
            self.assertTrue(any('differs from the pinned assets manifest' in b for b in blockers))


class AttemptTests(unittest.TestCase):
    def test_successful_attempt_stages_manifest_without_touching_repo(self):
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, schema = make_world(tmp)
            schema_path = Path(tmp) / 'schema.json'
            schema_path.write_text(json.dumps(schema))
            profile_path = Path(tmp) / 'edge_llama.json'
            profile_path.write_text(json.dumps(profile, indent=2) + '\n')
            assets_path = Path(tmp) / 'assets.json'
            assets_path.write_text(json.dumps(assets))
            reference_path = Path(tmp) / 'reference.json'
            reference_path.write_text(json.dumps(reference))
            pylock = Path(tmp) / 'pylock.export.toml'
            pylock.write_text('# lock\n')
            output = Path(tmp) / 'attempt_02'

            def fake_export(config_path, checkout_dir, attempt_dir):
                # The subprocess runs with cwd=checkout; a relative --config
                # would resolve against the checkout, not the repo.
                if not Path(config_path).is_absolute():
                    raise AssertionError('config path must be absolute')
                pte = attempt_dir / 'exported/llama3_2_1b_xnnpack.pte'
                pte.parent.mkdir(parents=True)
                pte.write_bytes(b'\x00\x00\x00\x00ET12' + b'fake pte bytes for staging test')
                return {'argv': ['fake'], 'returncode': 0, 'completion_reason': 'EXITED',
                        'process_wall_seconds': 0.1, 'sampled_peak_rss_kib': 1,
                        'rss_sampling_interval_seconds': 0.1, 'memory_note': 'stub',
                        'stdout_bytes': 0, 'stderr_bytes': 0,
                        'export_working_directory': str(checkout_dir), 'omp_num_threads': 8}

            with patch.object(edge_export_run, 'run_export', fake_export):
                result = export_attempt(profile_path, schema_path,
                                        reference_path, checkout, weights, assets_path,
                                        pylock, output)
            self.assertEqual(result['status'], 'EXPORTED_STAGED_NOT_COMMITTED', result)
            manifest = json.loads((output / 'llama3_2_1b_export.json').read_text())
            self.assertEqual(manifest['max_seq_length'], 256)
            self.assertEqual(manifest['max_context_length'], 512)
            # Hash fidelity: profile input hash is the hash of the actual profile bytes,
            # exactly what the device's verify_inputs recomputes over its deployed copy.
            self.assertEqual(manifest['inputs_sha256']['configs/edge_llama.json'],
                             hashlib.sha256(profile_path.read_bytes()).hexdigest())
            self.assertEqual(manifest['inputs_sha256']['manifests/llama3_2_1b_assets.json'],
                             hashlib.sha256(assets_path.read_bytes()).hexdigest())
            self.assertEqual(manifest['inputs_sha256']['requirements/pylock.export.toml'],
                             hashlib.sha256(pylock.read_bytes()).hexdigest())
            staged_pte = output / 'exported' / 'llama3_2_1b_xnnpack.pte'
            self.assertEqual(manifest['artifact']['sha256'],
                             hashlib.sha256(staged_pte.read_bytes()).hexdigest())
            self.assertEqual(manifest['artifact_validation']['status'], 'NOT_RUN_IN_THIS_ENV')
            self.assertEqual(manifest['device_inference'], 'BLOCKED_NOT_VERIFIED')
            # A second attempt into the same directory refuses rather than overwriting.
            with patch.object(edge_export_run, 'run_export', fake_export):
                again = export_attempt(profile_path, schema_path,
                                       reference_path, checkout, weights, assets_path,
                                       pylock, output)
            self.assertEqual(again['status'], 'BLOCKED_NOT_RUN')
            self.assertIn('never reused', again['blockers'][0])

    def test_failed_export_keeps_logs_and_stages_no_manifest(self):
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, schema = make_world(tmp)
            for name, content in (('p', profile), ('a', assets), ('r', reference), ('s', schema)):
                (Path(tmp) / name).write_text(json.dumps(content))
            (Path(tmp) / 'lock').write_text('#\n')

            def failing_export(config_path, checkout_dir, attempt_dir):
                return {'argv': ['fake'], 'returncode': 3, 'completion_reason': 'EXITED',
                        'process_wall_seconds': 0.1, 'sampled_peak_rss_kib': None,
                        'rss_sampling_interval_seconds': 0.1, 'memory_note': 'stub',
                        'stdout_bytes': 0, 'stderr_bytes': 0,
                        'export_working_directory': str(checkout_dir), 'omp_num_threads': 8}

            with patch.object(edge_export_run, 'run_export', failing_export):
                result = export_attempt(Path(tmp) / 'p', Path(tmp) / 's',
                                        Path(tmp) / 'r', checkout, weights, Path(tmp) / 'a',
                                        Path(tmp) / 'lock', Path(tmp) / 'attempt')
            self.assertEqual(result['status'], 'EXPORT_FAILED_RETAINED_LOGS')
            self.assertFalse((Path(tmp) / 'attempt/llama3_2_1b_export.json').exists())


STUB_EXPORT_MODULE = '''"""Stand-in for extension.llm.export.export_llm."""
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
out_dir = Path(config["export"]["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / (config["export"]["output_name"] + ".pte")
artifact.write_bytes(b"\\x00\\x00\\x00\\x00ET12" + b"stub artifact bytes")
print("stub export stdout")
print("stub export stderr", file=sys.stderr)
'''


def add_stub_export_module(checkout):
    module_dir = Path(checkout) / 'extension/llm/export'
    module_dir.mkdir(parents=True)
    for package in (Path(checkout) / 'extension', Path(checkout) / 'extension/llm',
                    module_dir):
        (package / '__init__.py').write_text('')
    (module_dir / 'export_llm.py').write_text(STUB_EXPORT_MODULE)


class SubprocessWiringTests(unittest.TestCase):
    def test_relative_output_with_spaces_runs_real_stub_subprocess(self):
        """The export child runs with the pinned checkout as cwd; a relative
        --config would resolve there and fail. Run the real subprocess path
        (no run_export mock) from a cwd with spaces, with a relative --output,
        to prove the config path handed to the child is absolute."""
        with TemporaryDirectory() as tmp:
            profile, reference, checkout, weights, assets, schema = make_world(tmp)
            add_stub_export_module(checkout)
            for name, content in (('p', profile), ('a', assets), ('r', reference),
                                  ('s', schema)):
                (Path(tmp) / name).write_text(json.dumps(content))
            (Path(tmp) / 'lock').write_text('#\n')
            spaced = Path(tmp) / 'dir with spaces'
            spaced.mkdir()
            previous_cwd = os.getcwd()
            os.chdir(spaced)
            try:
                result = export_attempt(Path(tmp) / 'p', Path(tmp) / 's',
                                        Path(tmp) / 'r', checkout, weights,
                                        Path(tmp) / 'a', Path(tmp) / 'lock',
                                        Path('relative attempt 02'))
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(result['status'], 'EXPORTED_STAGED_NOT_COMMITTED', result)
            attempt = spaced / 'relative attempt 02'
            config_arg = result['export_process']['argv'][
                result['export_process']['argv'].index('--config') + 1]
            self.assertTrue(Path(config_arg).is_absolute(), config_arg)
            self.assertTrue((attempt / 'export.stdout.log').read_text().strip())
            manifest = json.loads((attempt / 'llama3_2_1b_export.json').read_text())
            pte = attempt / 'exported/llama3_2_1b_xnnpack.pte'
            self.assertEqual(manifest['artifact']['sha256'],
                             hashlib.sha256(pte.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
