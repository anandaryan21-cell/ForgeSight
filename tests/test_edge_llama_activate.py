"""Host fixture tests for the Llama hash-chain staging/activation/rollback tool.

No real PTE, device or venv here: tiny fixture bytes stand in for the model,
and the process scan is monkeypatched. These tests prove the swap mechanics,
verification gates and rollback, not device behavior.
"""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from forgesight import edge_llama_activate as act


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pte(tag):
    # Fixture bytes with the pinned ET12 schema identifier at offset 4.
    return b'\x00' * 4 + b'ET12' + (b'fixture-pte:' + tag) * 8


def make_chain(model_dir, tag, seq):
    """Write one coherent five-file chain into model_dir; return its facts."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pte = _pte(tag)
    tokenizer = b'fixture-tokenizer:' + tag
    assets = {
        'schema_version': 1,
        'model_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'model_revision': '9213176726f574b556790deb65791e0c5aa438b6',
        'files': {'original/tokenizer.model': {'bytes': len(tokenizer),
                                               'sha256': _sha(tokenizer)}},
    }
    profile = {
        'schema_version': 1,
        'model_id': assets['model_id'],
        'model_revision': assets['model_revision'],
        'executorch_commit': '3b60683923245cf472b7323426920e15623ba361',
        'backend': 'xnnpack',
        'max_seq_length': seq,
        'max_context_length': 512,
    }
    export = {
        'schema_version': 1,
        'model_id': assets['model_id'],
        'model_revision': assets['model_revision'],
        'executorch_commit': profile['executorch_commit'],
        'backend_requested': 'xnnpack',
        'quantization': {'qmode': '8da4w', 'group_size': 128,
                         'embedding_quantize': '4,32'},
        'max_seq_length': seq,
        'max_context_length': 512,
        'artifact': {'bytes': len(pte), 'sha256': _sha(pte)},
        'inputs_sha256': {},  # filled after the pinned files are written
    }
    (model_dir / act.PTE_NAME).write_bytes(pte)
    (model_dir / act.TOKENIZER_NAME).write_bytes(tokenizer)
    (model_dir / act.ASSETS_NAME).write_text(json.dumps(assets, indent=2) + '\n')
    (model_dir / act.PROFILE_NAME).write_text(json.dumps(profile, indent=2) + '\n')
    export['inputs_sha256'] = {
        act.PROFILE_PIN_KEY: _sha((model_dir / act.PROFILE_NAME).read_bytes()),
        act.ASSETS_PIN_KEY: _sha((model_dir / act.ASSETS_NAME).read_bytes()),
    }
    (model_dir / act.EXPORT_NAME).write_text(json.dumps(export, indent=2) + '\n')
    return {'pte': pte, 'tokenizer': tokenizer, 'profile': profile, 'export': export}


def make_world(tmp):
    """A live llama root on the 128 chain plus a coherent 256 candidate trio."""
    root = Path(tmp)
    llama_root = root / 'llama_cpu_v1'
    old = make_chain(llama_root / act.MODEL_DIR, b'old128', 128)
    staging = root / 'staging'
    candidate = staging / 'candidate'
    candidate.mkdir(parents=True)
    new = make_chain(candidate, b'new256', 256)
    # The candidate ships only the three replaced files.
    (candidate / act.TOKENIZER_NAME).unlink()
    (candidate / act.ASSETS_NAME).unlink()
    # make_chain pins the candidate's own assets file; re-pin the manifest to
    # the LIVE assets/tokenizer, which is the unchanged-set contract.
    export = dict(new['export'])
    export['inputs_sha256'] = {
        act.PROFILE_PIN_KEY: _sha((candidate / act.PROFILE_NAME).read_bytes()),
        act.ASSETS_PIN_KEY: _sha((llama_root / act.MODEL_DIR / act.ASSETS_NAME).read_bytes()),
    }
    (candidate / act.EXPORT_NAME).write_text(json.dumps(export, indent=2) + '\n')
    return llama_root, candidate, old, new


QUIET_SCAN = {'matching': [], 'unreadable': 0, 'scan': 'procfs'}
BUSY_SCAN = {'matching': [{'pid': 4242, 'cmdline': 'python -m forgesight.jetson_live_ui'}],
             'unreadable': 0, 'scan': 'procfs'}


class PreflightTests(unittest.TestCase):
    def test_coherent_candidate_passes_and_plans_unchanged_set(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, old, new = make_world(tmp)
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertEqual(report['status'], 'PREFLIGHT_OK')
            self.assertEqual(report['blockers'], [])
            plan = report['plan']
            self.assertEqual(plan['max_seq_length'], {'live': 128, 'candidate': 256})
            self.assertEqual(plan['keeps_unchanged'][act.TOKENIZER_NAME]['sha256'],
                             _sha(old['tokenizer']))
            self.assertNotEqual(plan['replaces'][act.PTE_NAME]['sha256'],
                                plan['replaces_live'][act.PTE_NAME]['sha256'])

    def test_candidate_profile_pin_mismatch_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            profile = json.loads((candidate / act.PROFILE_NAME).read_text())
            profile['max_seq_length'] = 999  # no longer matches the manifest pin
            (candidate / act.PROFILE_NAME).write_text(json.dumps(profile, indent=2) + '\n')
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('Profile differs' in b for b in report['blockers']))

    def test_candidate_pte_hash_mismatch_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            pte = candidate / act.PTE_NAME
            data = bytearray(pte.read_bytes())
            data[-1] ^= 0xFF  # same size, different hash
            pte.write_bytes(bytes(data))
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('artifact checksum mismatch' in b for b in report['blockers']))

    def test_candidate_pte_without_et12_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            pte = candidate / act.PTE_NAME
            data = bytearray(pte.read_bytes())
            data[4:8] = b'XXXX'
            pte.write_bytes(bytes(data))
            manifest = json.loads((candidate / act.EXPORT_NAME).read_text())
            manifest['artifact']['sha256'] = _sha(bytes(data))
            (candidate / act.EXPORT_NAME).write_text(json.dumps(manifest, indent=2) + '\n')
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('ET12' in b or 'schema identifier' in b
                                for b in report['blockers']))

    def test_candidate_manifest_against_different_assets_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            manifest = json.loads((candidate / act.EXPORT_NAME).read_text())
            manifest['inputs_sha256'][act.ASSETS_PIN_KEY] = _sha(b'some other assets file')
            (candidate / act.EXPORT_NAME).write_text(json.dumps(manifest, indent=2) + '\n')
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('asset reference differs' in b for b in report['blockers']))

    def test_missing_and_extra_candidate_files_block(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            (candidate / act.PROFILE_NAME).unlink()
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('missing files' in b for b in report['blockers']))
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            (candidate / 'notes.txt').write_text('operator notes')
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('unexpected entries' in b for b in report['blockers']))

    def test_noop_candidate_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, old, _ = make_world(tmp)
            model_dir = llama_root / act.MODEL_DIR
            for name in act.REPLACED_NAMES:
                (candidate / name).write_bytes((model_dir / name).read_bytes())
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('nothing to activate' in b for b in report['blockers']))

    def test_broken_live_set_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            model_dir = llama_root / act.MODEL_DIR
            (model_dir / act.PROFILE_NAME).write_text('{"tampered": true}\n')
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.preflight(llama_root, candidate)
            self.assertTrue(any('Live set is not a coherent chain' in b
                                for b in report['blockers']))


class ActivateTests(unittest.TestCase):
    def test_happy_path_swaps_trio_keeps_unchanged_and_writes_evidence(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, old, new = make_world(tmp)
            model_dir = llama_root / act.MODEL_DIR
            before_unchanged = act._hashes(model_dir, act.UNCHANGED_NAMES)
            report_path = Path(tmp) / 'reports' / 'activate.json'
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                      report_path)
            self.assertEqual(report['status'], 'ACTIVATED_VERIFIED_PENDING_APP_RESTART')
            self.assertEqual(report['live_chain']['max_seq_length'], 256)
            self.assertEqual(act.sha256_file(model_dir / act.PTE_NAME), _sha(new['pte']))
            self.assertEqual(act._hashes(model_dir, act.UNCHANGED_NAMES), before_unchanged)
            backup_dir = Path(report['backup_dir'])
            self.assertEqual(act.sha256_file(backup_dir / act.PTE_NAME), _sha(old['pte']))
            record = json.loads((backup_dir / 'backup.json').read_text())
            self.assertEqual(record['files'][act.PTE_NAME]['sha256'], _sha(old['pte']))
            journal = json.loads((backup_dir / 'activation_journal.json').read_text())
            self.assertEqual(journal['state'], 'ACTIVATED')
            self.assertTrue(report_path.is_file())
            # Reports never overwrite: an existing report path refuses up front.
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                with self.assertRaises(act.StageError) as raised:
                    act.activate(llama_root, candidate, Path(tmp) / 'backups', report_path)
            self.assertIn('choose a new file', str(raised.exception))

    def test_running_app_blocks_before_any_change(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, old, _ = make_world(tmp)
            model_dir = llama_root / act.MODEL_DIR
            before = act._hashes(model_dir, act.ALL_NAMES)
            with patch.object(act, 'scan_app_processes', return_value=BUSY_SCAN):
                report = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                      Path(tmp) / 'activate.json')
            self.assertEqual(report['status'], 'BLOCKED')
            self.assertIn('running', report['error'])
            self.assertEqual(act._hashes(model_dir, act.ALL_NAMES), before)
            self.assertFalse((Path(tmp) / 'backups').exists()
                             and any((Path(tmp) / 'backups').iterdir()))

    def test_stale_staging_temp_blocks(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            model_dir = llama_root / act.MODEL_DIR
            before = act._hashes(model_dir, act.ALL_NAMES)
            (model_dir / (act.TEMP_PREFIX + act.PTE_NAME + '.tmp')).write_bytes(b'stale')
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                      Path(tmp) / 'activate.json')
            self.assertEqual(report['status'], 'BLOCKED')
            self.assertIn('Stale staging temp', report['error'])
            self.assertEqual(act._hashes(model_dir, act.ALL_NAMES), before)

    def test_same_second_reactivate_after_rollback_gets_fresh_backup(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, old, new = make_world(tmp)
            frozen = '2026-11-22T10:00:00Z'
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN), \
                    patch.object(act, '_utc', return_value=frozen):
                first = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                     Path(tmp) / 'a1.json')
                back = act.rollback(llama_root, Path(first['backup_dir']),
                                    Path(tmp) / 'r1.json')
                second = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                      Path(tmp) / 'a2.json')
            self.assertEqual(first['status'], 'ACTIVATED_VERIFIED_PENDING_APP_RESTART')
            self.assertEqual(back['status'], 'ROLLED_BACK_VERIFIED')
            self.assertEqual(second['status'], 'ACTIVATED_VERIFIED_PENDING_APP_RESTART')
            self.assertNotEqual(first['backup_dir'], second['backup_dir'])
            model_dir = llama_root / act.MODEL_DIR
            self.assertEqual(act.sha256_file(model_dir / act.PTE_NAME), _sha(new['pte']))

    def test_post_swap_verify_failure_rolls_back_automatically(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, old, _ = make_world(tmp)
            model_dir = llama_root / act.MODEL_DIR
            real_verify = act.verify_chain
            calls = {'n': 0}

            def fail_after_swap(*args, **kwargs):
                calls['n'] += 1
                if calls['n'] == 3:  # live preflight, candidate preflight, post-swap live
                    raise act.StageError('simulated post-swap incoherence')
                return real_verify(*args, **kwargs)

            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN), \
                    patch.object(act, 'verify_chain', side_effect=fail_after_swap):
                report = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                      Path(tmp) / 'activate.json')
            self.assertEqual(report['status'], 'FAILED_ROLLED_BACK_VERIFIED')
            self.assertEqual(act.sha256_file(model_dir / act.PTE_NAME), _sha(old['pte']))
            self.assertEqual(act.sha256_file(model_dir / act.PROFILE_NAME),
                             _sha((llama_root / act.MODEL_DIR / act.PROFILE_NAME).read_bytes()))
            self.assertEqual(act._hashes(model_dir, act.REPLACED_NAMES)[act.PTE_NAME]['sha256'],
                             _sha(old['pte']))


class RollbackTests(unittest.TestCase):
    def _activated(self, tmp):
        llama_root, candidate, old, new = make_world(tmp)
        with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
            report = act.activate(llama_root, candidate, Path(tmp) / 'backups',
                                  Path(tmp) / 'activate.json')
        assert report['status'] == 'ACTIVATED_VERIFIED_PENDING_APP_RESTART', report
        return llama_root, Path(report['backup_dir']), old, new

    def test_rollback_restores_old_chain_and_preserves_live(self):
        with TemporaryDirectory() as tmp:
            llama_root, backup_dir, old, new = self._activated(tmp)
            model_dir = llama_root / act.MODEL_DIR
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.rollback(llama_root, backup_dir, Path(tmp) / 'rollback.json')
            self.assertEqual(report['status'], 'ROLLED_BACK_VERIFIED')
            self.assertEqual(act.sha256_file(model_dir / act.PTE_NAME), _sha(old['pte']))
            self.assertEqual(report['live_chain']['max_seq_length'], 128)
            preserved = Path(report['preserved_live_to'])
            self.assertEqual(act.sha256_file(preserved / act.PTE_NAME), _sha(new['pte']))

    def test_second_rollback_is_a_verified_noop(self):
        with TemporaryDirectory() as tmp:
            llama_root, backup_dir, old, _ = self._activated(tmp)
            model_dir = llama_root / act.MODEL_DIR
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                first = act.rollback(llama_root, backup_dir, Path(tmp) / 'r1.json')
                second = act.rollback(llama_root, backup_dir, Path(tmp) / 'r2.json')
            self.assertEqual(first['status'], 'ROLLED_BACK_VERIFIED')
            self.assertEqual(second['status'], 'ALREADY_MATCHES_BACKUP')
            self.assertEqual(act.sha256_file(model_dir / act.PTE_NAME), _sha(old['pte']))

    def test_tampered_backup_blocks_without_touching_live(self):
        with TemporaryDirectory() as tmp:
            llama_root, backup_dir, _, new = self._activated(tmp)
            model_dir = llama_root / act.MODEL_DIR
            (backup_dir / act.PTE_NAME).write_bytes(b'tampered backup bytes')
            before = act._hashes(model_dir, act.ALL_NAMES)
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                report = act.rollback(llama_root, backup_dir, Path(tmp) / 'rollback.json')
            self.assertEqual(report['status'], 'BLOCKED')
            self.assertEqual(act._hashes(model_dir, act.ALL_NAMES), before)


class CliTests(unittest.TestCase):
    def test_preflight_cli_exit_codes(self):
        with TemporaryDirectory() as tmp:
            llama_root, candidate, _, _ = make_world(tmp)
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                self.assertEqual(act.main(['preflight', '--llama-root', str(llama_root),
                                           '--candidate', str(candidate)]), 0)
            (candidate / act.PTE_NAME).unlink()
            with patch.object(act, 'scan_app_processes', return_value=QUIET_SCAN):
                self.assertEqual(act.main(['preflight', '--llama-root', str(llama_root),
                                           '--candidate', str(candidate)]), 1)


if __name__ == '__main__':
    unittest.main()
