"""Bounded pins for the root diagnostic collector's discovery and packing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / 'forgesight_diag_collect.py'
MARKER = {'purpose': 'FORGESIGHT_STAGE_2_9_CPU_RUNTIME'}


def make_install(root, valid=True):
    root = Path(root)
    (root / 'model').mkdir(parents=True)
    (root / 'venv' / 'bin').mkdir(parents=True)
    marker = dict(MARKER, prefix=str(root))
    (root / '.forgesight-runtime.json').write_text(json.dumps(marker))
    for name in ('edge_llama.json', 'llama3_2_1b_xnnpack.pte', 'tokenizer.model'):
        (root / 'model' / name).write_text('{}')
    (root / 'venv' / 'bin' / 'python').write_text('')
    if not valid:
        (root / 'model' / 'llama3_2_1b_xnnpack.pte').unlink()


class DiagCollectDiscoveryTest(unittest.TestCase):
    def run_collector(self, workdir, home):
        env = dict(os.environ, HOME=str(home))
        return subprocess.run([sys.executable, '-B', str(SCRIPT)],
                              cwd=workdir, capture_output=True, text=True,
                              timeout=60, env=env)

    def bundle(self, workdir):
        dirs = [p for p in Path(workdir).iterdir()
                if p.is_dir() and p.name.startswith('forgesight_diag_')]
        self.assertEqual(len(dirs), 1)
        return dirs[0]

    def test_unique_validated_install_is_selected(self):
        with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as home:
            make_install(Path(home) / '.local/share/forgesight/llama_cpu_v1')
            result = self.run_collector(workdir, home)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = self.bundle(workdir)
            discovery = json.loads((bundle / 'llama_root_discovery.json').read_text())
            self.assertEqual(discovery['status'], 'SELECTED_UNIQUE_VALIDATED')
            self.assertTrue(discovery['selected'].endswith('llama_cpu_v1'))
            # Privacy pin: no remote listing (remote URLs can carry credentials).
            git_payload = json.loads((bundle / 'git_checkout.json').read_text())
            self.assertEqual(sorted(git_payload.keys()),
                             ['checkout', 'head', 'status_porcelain', 'tree'])
            self.assertEqual(len(list(Path(workdir).glob('forgesight_diag_*.zip'))), 1)

    def test_two_validated_installs_are_reported_never_guessed(self):
        with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as home:
            make_install(Path(home) / '.local/share/forgesight/llama_cpu_v1')
            make_install(Path(home) / '.local/share/forgesight/llama_cpu_v1_b')
            result = self.run_collector(workdir, home)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = self.bundle(workdir)
            discovery = json.loads((bundle / 'llama_root_discovery.json').read_text())
            self.assertEqual(discovery['status'], 'AMBIGUOUS_NOT_SELECTED')
            self.assertIsNone(discovery['selected'])
            missing = json.loads((bundle / 'missing_items.json').read_text())
            self.assertFalse(missing['collected_everything'])

    def test_same_named_records_and_run_jsons_all_copied_uniquely(self):
        with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as home:
            make_install(Path(home) / '.local/share/forgesight/llama_cpu_v1')
            for run in ('run_a', 'run_b'):
                run_dir = Path(home) / '.local/share/forgesight/vision_cpu_v1/runs' / run
                run_dir.mkdir(parents=True)
                (run_dir / 'record.json').write_text(json.dumps({'run': run}))
                (run_dir / 'advice_report.json').write_text(json.dumps({'run': run}))
            result = self.run_collector(workdir, home)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = self.bundle(workdir)
            copied = json.loads((bundle / 'copied_reports.json').read_text())
            record_files = [entry.get('bundle_file') for entry in copied['records']]
            self.assertEqual(sorted(record_files), ['record_00.json', 'record_01.json'])
            self.assertTrue(all(entry['copied'] for entry in copied['records']))
            json_files = [entry.get('bundle_file') for entry in copied['run_jsons']]
            self.assertEqual(len(json_files), len(set(json_files)))
            self.assertTrue(all(entry['copied'] for entry in copied['run_jsons']))
            sources = [entry['source'] for entry in copied['records']]
            self.assertTrue(any('run_a' in s for s in sources))
            self.assertTrue(any('run_b' in s for s in sources))

    def test_unmarked_or_incomplete_install_is_not_selected(self):
        with tempfile.TemporaryDirectory() as workdir, tempfile.TemporaryDirectory() as home:
            make_install(Path(home) / '.local/share/forgesight/llama_cpu_v1', valid=False)
            result = self.run_collector(workdir, home)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = self.bundle(workdir)
            discovery = json.loads((bundle / 'llama_root_discovery.json').read_text())
            self.assertEqual(discovery['status'], 'NONE_VALIDATED')


if __name__ == '__main__':
    unittest.main()
