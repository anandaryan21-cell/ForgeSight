"""Stage, activate and roll back the three-file Llama advisory hash chain.

The Jetson advisory runtime pins one coherent set of files under the private
llama root's ``model/`` directory. Three of them change together when a new
export is adopted:

- ``llama3_2_1b_xnnpack.pte``      (artifact, pinned by the export manifest)
- ``edge_llama.json``              (profile, pinned by the export manifest)
- ``llama3_2_1b_export.json``      (export manifest, pins the other two)

Two must NOT change in this operation:

- ``tokenizer.model``              (pinned by the assets manifest)
- ``llama3_2_1b_assets.json``      (pinned by the export manifest)

Why they must move together: the persistent worker
(``forgesight.assist_edge_worker``) runs ``edge_jetson_smoke.verify_inputs``
at startup and stat-checks the artifact on every request, and the controller
(``assist_jetson_replay_once._pinned_asset_hashes``) re-derives the pinned
artifact/tokenizer hashes from the live manifests on every advisory request.
A mixed set (new PTE with the old manifest, or the reverse) fails closed at
both layers: requests are rejected, never answered from a mismatched chain.
That fail-closed property is the safety net; this script's job is to make the
swap deliberate, verified and reversible instead of a blind three-file copy.

Verification here mirrors the pure file checks of
``edge_jetson_smoke.verify_inputs`` exactly (provenance equality across
profile/export/assets, the profile and assets pins in the export manifest's
``inputs_sha256``, the artifact size+SHA-256 fingerprint, and the tokenizer
fingerprint from the assets manifest), plus the ET12 schema identifier check
from ``edge_artifact.check_fingerprint``. It uses only the standard library,
so it can run before the device venv is involved; it does NOT replace the
device gate. The authoritative proof remains the restarted worker's own
``verify_inputs`` plus warmup generation and the first accepted advisory.
This script never claims generation, quality or device acceptance.

Operation contract (same discipline as the rest of the edge tooling):

- ``preflight`` changes nothing and lists every blocker.
- ``activate`` refuses to touch anything while any blocker exists, including
  a detected running operator app or advisory worker (the app owns the worker
  as a child process; both must be stopped first) or any process whose
  cmdline it cannot read - if it cannot prove the app is stopped, it fails
  closed. It backs up the live trio with verified hashes before replacing
  anything, replaces via same-directory temp files + ``os.replace`` +
  directory fsync, and re-verifies the live chain. A failure before the
  first replace reports FAILED_LIVE_UNCHANGED; a failure after it restores
  the verified backup (FAILED_ROLLED_BACK_VERIFIED) or names manual recovery
  with the backup and journal (FAILED_MANUAL_INTERVENTION). It never reports
  "no changes" once a write may have happened.
- ``rollback`` verifies the backup before restoring, preserves whatever is
  currently live under ``pre_rollback_live/`` (nothing is silently destroyed),
  and re-verifies the restored chain.
- Evidence reports are written to fresh paths only (``open(..., 'x')``); an
  existing report path is an error, matching the repo's no-overwrite rule.

Crash honesty: three files cannot be replaced atomically as one operation.
The app must be stopped, so the brief mixed-set window is unobserved by
requests; if the process dies mid-swap, the journal and backup in the backup
directory plus a rerun of ``preflight``/``rollback`` recover a coherent set.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

MODEL_DIR = 'model'
PTE_NAME = 'llama3_2_1b_xnnpack.pte'
PROFILE_NAME = 'edge_llama.json'
EXPORT_NAME = 'llama3_2_1b_export.json'
ASSETS_NAME = 'llama3_2_1b_assets.json'
TOKENIZER_NAME = 'tokenizer.model'
REPLACED_NAMES = (PTE_NAME, PROFILE_NAME, EXPORT_NAME)
UNCHANGED_NAMES = (TOKENIZER_NAME, ASSETS_NAME)
ALL_NAMES = REPLACED_NAMES + UNCHANGED_NAMES

# The two pins the export manifest must carry, exactly as verify_inputs reads.
PROFILE_PIN_KEY = 'configs/edge_llama.json'
ASSETS_PIN_KEY = 'manifests/llama3_2_1b_assets.json'
TOKENIZER_ASSET_KEY = 'original/tokenizer.model'

# Processes whose open file tables make a live swap unsafe to start. The
# operator app launches the advisory worker as its child, so both match.
APP_PROCESS_MARKERS = ('forgesight.jetson_live_ui', 'forgesight.assist_edge_worker')

TEMP_PREFIX = '.activate-'


class StageError(ValueError):
    """A coherence, safety or integrity check failed; nothing was hidden."""


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _read_json(path, what):
    try:
        document = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise StageError(f'{what} is missing or not valid JSON: {path} ({error})')
    if not isinstance(document, dict):
        raise StageError(f'{what} must be a JSON object: {path}')
    return document


def _require_hex64(value, what):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise StageError(f'{what} is not a lowercase SHA-256 hex digest.')
    return value


def _check_fingerprint(path, expected, what):
    """Mirror edge_jetson_smoke.fingerprint: size first, then streaming hash."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected.get('bytes'):
        raise StageError(f'Missing or wrong-sized {what}: {path.name}')
    if expected.get('bytes') == 0:
        raise StageError(f'{what} has an empty pin: {path.name}')
    actual = sha256_file(path)
    if actual != _require_hex64(expected.get('sha256'), f'{what} sha256 pin'):
        raise StageError(f'{what} checksum mismatch: {path.name}')
    return {'bytes': path.stat().st_size, 'sha256': actual}


def verify_chain(pte, profile_path, export_path, assets_path, tokenizer_path):
    """Verify one coherent five-file chain; return the evidence. Raises StageError.

    Pure-file mirror of edge_jetson_smoke.verify_inputs (no device/venv gate),
    plus the ET12 schema identifier check from edge_artifact.check_fingerprint.
    """
    profile = _read_json(profile_path, 'Runtime profile')
    export = _read_json(export_path, 'Export manifest')
    assets = _read_json(assets_path, 'Assets manifest')
    try:
        provenance_ok = (
            profile['model_id'] == export['model_id'] == assets['model_id']
            and profile['model_revision'] == export['model_revision'] == assets['model_revision']
            and profile['executorch_commit'] == export['executorch_commit']
            and profile['backend'] == 'xnnpack'
            and export['backend_requested'] == 'xnnpack'
            and export['quantization']['qmode'] == '8da4w')
    except (KeyError, TypeError) as error:
        raise StageError(f'Profile/export/assets lack a required provenance field: {error}')
    if not provenance_ok:
        raise StageError('Model/export/profile provenance differs.')
    inputs = export.get('inputs_sha256')
    if not isinstance(inputs, dict):
        raise StageError('Export manifest carries no inputs_sha256 pins.')
    profile_pin = _require_hex64(inputs.get(PROFILE_PIN_KEY), 'profile input pin')
    assets_pin = _require_hex64(inputs.get(ASSETS_PIN_KEY), 'assets input pin')
    profile_sha = sha256_file(profile_path)
    if profile_sha != profile_pin:
        raise StageError('Profile differs from the exported model input.')
    assets_sha = sha256_file(assets_path)
    if assets_sha != assets_pin:
        raise StageError('Tokenizer asset reference differs from export input.')
    artifact = export.get('artifact')
    if not isinstance(artifact, dict):
        raise StageError('Export manifest has no artifact fingerprint.')
    artifact_fp = _check_fingerprint(pte, artifact, 'exported artifact')
    if Path(pte).suffix != '.pte':
        raise StageError('Expected an ExecuTorch .pte artifact.')
    with Path(pte).open('rb') as stream:
        header = stream.read(8)
    if header[4:8] != b'ET12':
        raise StageError('Artifact does not have the pinned program schema identifier.')
    files = assets.get('files')
    tokenizer_entry = files.get(TOKENIZER_ASSET_KEY) if isinstance(files, dict) else None
    if not isinstance(tokenizer_entry, dict):
        raise StageError('Assets manifest has no tokenizer pin.')
    tokenizer_fp = _check_fingerprint(tokenizer_path, tokenizer_entry, 'tokenizer')
    return {
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'executorch_commit': profile['executorch_commit'],
        'max_seq_length': profile.get('max_seq_length'),
        'max_context_length': profile.get('max_context_length'),
        'artifact': artifact_fp,
        'profile_sha256': profile_sha,
        'assets_sha256': assets_sha,
        'tokenizer': tokenizer_fp,
        'file_identifier': 'ET12',
        'verification': 'pure-file mirror of verify_inputs; no device, venv, load or generation gate',
    }


def scan_app_processes():
    """Best-effort /proc scan for a running operator app or advisory worker.

    Returns matching (pid, cmdline) rows plus a count of processes whose
    cmdline was unreadable to this user (a root-owned process would be).
    Matching rows block activation, and so does any unreadable cmdline: an
    unreadable process cannot be proven innocent, so activation fails closed
    rather than guessing.
    """
    result = {'matching': [], 'unreadable': 0, 'scan': 'procfs'}
    proc = Path('/proc')
    if not proc.is_dir():
        result['scan'] = 'unavailable'
        return result
    self_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == self_pid:
            continue
        try:
            cmdline = (entry / 'cmdline').read_bytes().decode(errors='replace')
        except OSError:
            result['unreadable'] += 1
            continue
        if not cmdline:
            continue
        if any(marker in cmdline for marker in APP_PROCESS_MARKERS):
            result['matching'].append({'pid': int(entry.name),
                                       'cmdline': cmdline.replace('\x00', ' ').strip()})
    return result


def _fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_verified(source, target):
    """Copy then prove the copy; raise before anything else happens on mismatch."""
    shutil.copyfile(source, target)
    if sha256_file(target) != sha256_file(source):
        raise StageError(f'Verified copy failed: {target}')


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _hashes(model_dir, names):
    return {name: {'bytes': (Path(model_dir) / name).stat().st_size,
                   'sha256': sha256_file(Path(model_dir) / name)}
            for name in names}


def preflight(llama_root, candidate_dir):
    """Read-only: verify live chain, candidate chain and drift; list blockers."""
    llama_root = Path(llama_root)
    model_dir = llama_root / MODEL_DIR
    candidate_dir = Path(candidate_dir)
    report = {'status': 'PREFLIGHT', 'llama_root': str(llama_root),
              'candidate_dir': str(candidate_dir), 'checked_utc': _utc(),
              'blockers': [], 'warnings': []}
    blockers = report['blockers']

    live = {}
    if not model_dir.is_dir():
        blockers.append(f'No live model directory at {model_dir}')
    else:
        missing = [name for name in ALL_NAMES if not (model_dir / name).is_file()]
        if missing:
            blockers.append(f'Live model directory is missing files: {missing}')
        stale = sorted(path.name for path in model_dir.glob(TEMP_PREFIX + '*.tmp'))
        if stale:
            blockers.append(f'Stale staging temp files in the live directory: {stale}; '
                            'a previous run was interrupted - inspect them before retrying')
        if not missing and not stale:
            try:
                live = verify_chain(*(model_dir / name for name in
                                      (PTE_NAME, PROFILE_NAME, EXPORT_NAME,
                                       ASSETS_NAME, TOKENIZER_NAME)))
                report['live_chain'] = live
            except StageError as error:
                blockers.append(f'Live set is not a coherent chain; resolve that first: {error}')

    candidate = {}
    if not candidate_dir.is_dir():
        blockers.append(f'No candidate directory at {candidate_dir}')
    else:
        names = sorted(path.name for path in candidate_dir.iterdir())
        missing = [name for name in REPLACED_NAMES if name not in names]
        extras = [name for name in names if name not in REPLACED_NAMES]
        if missing:
            blockers.append(f'Candidate directory is missing files: {missing}')
        if extras:
            blockers.append(f'Candidate directory holds unexpected entries: {extras}')
        if not missing and not extras and live is not None and model_dir.is_dir():
            # The candidate is verified against the CURRENT assets/tokenizer:
            # this is exactly what proves those two stay untouched and the new
            # manifest still pins them.
            try:
                candidate = verify_chain(candidate_dir / PTE_NAME,
                                         candidate_dir / PROFILE_NAME,
                                         candidate_dir / EXPORT_NAME,
                                         model_dir / ASSETS_NAME,
                                         model_dir / TOKENIZER_NAME)
                report['candidate_chain'] = candidate
            except StageError as error:
                blockers.append(f'Candidate is not coherent with the live assets/tokenizer: {error}')

    if live and candidate:
        if candidate['artifact']['sha256'] == live['artifact']['sha256']:
            blockers.append('Candidate artifact is identical to the live artifact; nothing to activate.')
        report['plan'] = {
            'replaces': dict(_hashes(candidate_dir, REPLACED_NAMES)),
            'replaces_live': dict(_hashes(model_dir, REPLACED_NAMES)),
            'keeps_unchanged': dict(_hashes(model_dir, UNCHANGED_NAMES)),
            'max_seq_length': {'live': live['max_seq_length'],
                               'candidate': candidate['max_seq_length']},
            'max_context_length': {'live': live['max_context_length'],
                                   'candidate': candidate['max_context_length']},
        }

    processes = scan_app_processes()
    report['process_scan'] = processes
    if processes['matching']:
        report['warnings'].append(
            'Operator app/advisory worker appears to be RUNNING; activation will refuse '
            'until it is stopped: ' + '; '.join(f"pid {row['pid']}: {row['cmdline']}"
                                                for row in processes['matching']))
    if processes['scan'] != 'procfs' or processes['unreadable']:
        report['warnings'].append(
            f"Process scan is {processes['scan']} with {processes['unreadable']} unreadable "
            'processes; activation will refuse until the app is stopped and every process '
            'cmdline is readable to the user running the tool.')
    if not blockers:
        report['status'] = 'PREFLIGHT_OK'
    return report


def _refuse_if_running(report):
    processes = scan_app_processes()
    report['process_scan'] = processes
    if processes['matching']:
        raise StageError(
            'Refusing to change the live set while the operator app/advisory worker is '
            'running: ' + '; '.join(f"pid {row['pid']}" for row in processes['matching']))
    if processes['scan'] != 'procfs':
        raise StageError('Cannot confirm the app is stopped: no procfs on this host.')
    if processes['unreadable']:
        raise StageError(
            f"Cannot prove the operator app/worker is stopped: {processes['unreadable']} "
            'process cmdlines are unreadable to this user. Stop the app and rerun, or '
            'rerun with read access to all processes (for example via sudo).')


def _install(model_dir, source_dir, names):
    """Replace files via same-dir temp + os.replace; fsync files and directory."""
    model_dir = Path(model_dir)
    for name in names:
        if (model_dir / (TEMP_PREFIX + name + '.tmp')).exists():
            raise StageError(
                f'Stale staging temp {TEMP_PREFIX}{name}.tmp in {model_dir}; '
                'a previous run was interrupted - inspect it before retrying.')
    temps = []
    try:
        for name in names:
            temp = model_dir / (TEMP_PREFIX + name + '.tmp')
            _copy_verified(Path(source_dir) / name, temp)
            with temp.open('rb') as stream:
                os.fsync(stream.fileno())
            temps.append(temp)
        for temp, name in zip(temps, names):
            os.replace(temp, model_dir / name)
    finally:
        for temp in temps:
            if temp.exists():
                temp.unlink()
    _fsync_dir(model_dir)


def _restore_from_backup(model_dir, backup_dir, record, report, reason):
    """Best-effort restore of the recorded trio; verify the chain afterwards."""
    report['rollback_reason'] = reason
    try:
        _install(model_dir, backup_dir, REPLACED_NAMES)
        verify_chain(*(Path(model_dir) / name for name in
                       (PTE_NAME, PROFILE_NAME, EXPORT_NAME, ASSETS_NAME, TOKENIZER_NAME)))
        report['status'] = 'FAILED_ROLLED_BACK_VERIFIED'
        report['rollback_detail'] = ('The live set was mixed or unverified after the failed '
                                     'swap; it was restored from the verified backup and the '
                                     'restored chain passes verification.')
    except (StageError, OSError) as error:
        report['status'] = 'FAILED_MANUAL_INTERVENTION'
        report['rollback_detail'] = (
            f'Automatic rollback did not complete cleanly: {error}. THE LIVE SET MAY BE '
            f'MIXED RIGHT NOW. Recover by hand: keep the app stopped, copy '
            f'{REPLACED_NAMES} from the verified backup at {backup_dir} into {model_dir}, '
            f'then rerun preflight against any candidate to prove the chain. The journal at '
            f'{backup_dir}/activation_journal.json records where the swap stopped. Do not '
            'start the app until the chain verifies.')
    return report


def activate(llama_root, candidate_dir, backup_parent, report_path):
    """Verified, reversible swap of the live trio for the candidate trio."""
    llama_root = Path(llama_root)
    model_dir = llama_root / MODEL_DIR
    report = {'status': 'BLOCKED', 'operation': 'activate', 'started_utc': _utc(),
              'llama_root': str(llama_root), 'candidate_dir': str(candidate_dir)}
    report_path = Path(report_path)
    if report_path.exists():
        # Nowhere honest to write the refusal; fail before touching anything.
        raise StageError(f'Evidence report exists; choose a new file: {report_path}')
    try:
        pre = preflight(llama_root, candidate_dir)
        report['preflight'] = pre
        if pre['blockers']:
            raise StageError('Preflight blockers: ' + ' | '.join(pre['blockers']))
        _refuse_if_running(report)

        backup_parent = Path(backup_parent)
        backup_parent.mkdir(parents=True, exist_ok=True)
        # Timestamp + candidate suffix: a rollback-then-reactivate in the same
        # second still gets a fresh, never-reused directory.
        stamp = _utc().replace(':', '').replace('-', '')
        suffix = pre['candidate_chain']['artifact']['sha256'][:8]
        backup_dir = None
        for attempt in [''] + [f'-{n}' for n in range(2, 10)]:
            trial = backup_parent / f'llama-chain-{stamp}-{suffix}{attempt}'
            try:
                trial.mkdir()
                backup_dir = trial
                break
            except FileExistsError:
                continue
        if backup_dir is None:
            raise StageError(f'No fresh backup directory available under {backup_parent}')
        report['backup_dir'] = str(backup_dir)

        # 1. Verified backup of the live trio before anything changes.
        for name in REPLACED_NAMES:
            _copy_verified(model_dir / name, backup_dir / name)
        record = {'created_utc': _utc(), 'llama_root': str(llama_root),
                  'model_dir': str(model_dir),
                  'files': _hashes(backup_dir, REPLACED_NAMES),
                  'candidate': _hashes(candidate_dir, REPLACED_NAMES),
                  'kept_unchanged': _hashes(model_dir, UNCHANGED_NAMES)}
        (backup_dir / 'backup.json').write_text(json.dumps(record, indent=2) + '\n')
        _fsync_dir(backup_dir)

        journal_path = backup_dir / 'activation_journal.json'
        journal_path.write_text(json.dumps(
            {'state': 'REPLACING', 'started_utc': _utc(),
             'note': ('If this state is found after a crash, the live set may be mixed; '
                      'run preflight, then rollback from this backup directory.')},
            indent=2) + '\n')

        # 2. Replace, then prove the new live chain. From here a failure can
        # leave a MIXED live set, so nothing in the handler may claim "no
        # changes": it restores the verified backup, or names manual recovery.
        try:
            _install(model_dir, candidate_dir, REPLACED_NAMES)
            report['live_chain'] = verify_chain(
                *(model_dir / name for name in
                  (PTE_NAME, PROFILE_NAME, EXPORT_NAME, ASSETS_NAME, TOKENIZER_NAME)))
        except Exception as error:
            journal_path.write_text(json.dumps(
                {'state': 'SWAP_FAILED', 'error': f'{type(error).__name__}: {error}'},
                indent=2) + '\n')
            live_now = _hashes(model_dir, REPLACED_NAMES)
            if all(live_now[name]['sha256'] == record['files'][name]['sha256']
                   for name in REPLACED_NAMES):
                # The failure happened before the first replace: the live trio
                # still equals the verified backup of the original chain.
                report['status'] = 'FAILED_LIVE_UNCHANGED'
                report['error'] = f'{type(error).__name__}: {error}'
                report['detail'] = ('The swap failed before any live file was replaced; '
                                    'the live trio still matches the verified backup of '
                                    'the original chain. Fix the cause and retry with a '
                                    'fresh report path.')
            else:
                report = _restore_from_backup(model_dir, backup_dir, record, report,
                                              f'{type(error).__name__}: {error}')
            _write_report(report_path, report)
            return report
        journal_path.write_text(json.dumps({'state': 'ACTIVATED', 'finished_utc': _utc()},
                                           indent=2) + '\n')
        report['status'] = 'ACTIVATED_VERIFIED_PENDING_APP_RESTART'
        report['next'] = ('Start the operator app; its advisory worker re-runs verify_inputs '
                          'and the bounded warmup on the new chain. The first accepted '
                          'advisory remains the authoritative device proof; this script '
                          'proves only the hash chain swap.')
    except (StageError, OSError) as error:
        # Reachable only before the journal/install phase: the swap handler
        # above owns every failure after the first possible write.
        report['status'] = 'BLOCKED'
        report['error'] = str(error)
        report['detail'] = 'No live file was changed.'
    _write_report(report_path, report)
    return report


def rollback(llama_root, backup_dir, report_path):
    """Restore the recorded trio; nothing live is destroyed without a copy."""
    llama_root = Path(llama_root)
    model_dir = llama_root / MODEL_DIR
    backup_dir = Path(backup_dir)
    report_path = Path(report_path)
    report = {'status': 'BLOCKED', 'operation': 'rollback', 'started_utc': _utc(),
              'llama_root': str(llama_root), 'backup_dir': str(backup_dir)}
    if report_path.exists():
        raise StageError(f'Evidence report exists; choose a new file: {report_path}')
    try:
        try:
            record = json.loads((backup_dir / 'backup.json').read_text())
            recorded = record['files']
        except (OSError, ValueError, KeyError) as error:
            raise StageError(f'Backup record is missing or unreadable: {error}')
        for name in REPLACED_NAMES:
            expected = recorded.get(name)
            if not isinstance(expected, dict):
                raise StageError(f'Backup record lacks {name}.')
            _check_fingerprint(backup_dir / name, expected, 'backup file')
        _refuse_if_running(report)

        live_before = _hashes(model_dir, REPLACED_NAMES)
        report['live_before'] = live_before
        if all(live_before[name]['sha256'] == recorded[name]['sha256']
               for name in REPLACED_NAMES):
            report['status'] = 'ALREADY_MATCHES_BACKUP'
            report['detail'] = 'Live trio already equals the backup; nothing was changed.'
        else:
            preserved = backup_dir / 'pre_rollback_live'
            try:
                preserved.mkdir()
            except FileExistsError:
                raise StageError(f'Pre-rollback preservation directory exists: {preserved}')
            for name in REPLACED_NAMES:
                _copy_verified(model_dir / name, preserved / name)
            report['preserved_live_to'] = str(preserved)
            # Same honesty rule as activate: once a replace may have happened,
            # the report must never imply the live set is simply unchanged.
            try:
                _install(model_dir, backup_dir, REPLACED_NAMES)
                report['live_chain'] = verify_chain(
                    *(model_dir / name for name in
                      (PTE_NAME, PROFILE_NAME, EXPORT_NAME, ASSETS_NAME, TOKENIZER_NAME)))
                report['status'] = 'ROLLED_BACK_VERIFIED'
                report['next'] = ('Start the operator app; its worker re-verifies the '
                                  'restored chain at startup.')
            except Exception as error:
                live_now = _hashes(model_dir, REPLACED_NAMES)
                if all(live_now[name]['sha256'] == recorded[name]['sha256']
                       for name in REPLACED_NAMES):
                    report['status'] = 'FAILED_MANUAL_INTERVENTION'
                    report['error'] = (f'The trio was restored from the backup but the chain '
                                       f'does not verify; assets/tokenizer may have drifted '
                                       f'independently: {type(error).__name__}: {error}')
                elif all(live_now[name]['sha256'] == live_before[name]['sha256']
                         for name in REPLACED_NAMES):
                    report['status'] = 'FAILED_LIVE_UNCHANGED'
                    report['error'] = f'{type(error).__name__}: {error}'
                    report['detail'] = ('Rollback failed before any live file was replaced; '
                                        'the live set is unchanged, and a verified copy is '
                                        f'also preserved at {preserved}. Fix the cause and '
                                        'retry with a fresh report path.')
                else:
                    report['status'] = 'FAILED_MANUAL_INTERVENTION'
                    report['error'] = (
                        f'Rollback failed mid-install: {type(error).__name__}: {error}. '
                        'THE LIVE SET MAY BE MIXED RIGHT NOW. Recover by hand: choose one '
                        f'coherent chain - the verified backup at {backup_dir} or the '
                        f'preserved previous live set at {preserved} - copy its '
                        f'{REPLACED_NAMES} into {model_dir}, then run preflight. Keep the '
                        'app stopped until the chain verifies.')
    except (StageError, OSError) as error:
        report['status'] = 'BLOCKED'
        report['error'] = str(error)
    _write_report(report_path, report)
    return report


def _write_report(report_path, report):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('preflight', 'activate', 'rollback'))
    parser.add_argument('--llama-root', type=Path, required=True,
                        help='Private llama runtime root holding the model/ directory.')
    parser.add_argument('--candidate', type=Path,
                        help='Fresh directory holding exactly the three new files.')
    parser.add_argument('--backup-parent', type=Path,
                        default=Path('backups'),
                        help='Parent for the fresh backup directory (activate).')
    parser.add_argument('--backup-dir', type=Path,
                        help='The recorded backup directory to restore (rollback).')
    parser.add_argument('--report', type=Path,
                        help='Fresh evidence output path (activate/rollback).')
    args = parser.parse_args(argv)

    if args.operation == 'preflight':
        if args.candidate is None:
            parser.error('preflight needs --candidate')
        report = preflight(args.llama_root, args.candidate)
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'PREFLIGHT_OK' else 1
    if args.report is None:
        parser.error(f'{args.operation} needs --report (a fresh path)')
    if args.operation == 'activate':
        if args.candidate is None:
            parser.error('activate needs --candidate')
        report = activate(args.llama_root, args.candidate, args.backup_parent, args.report)
    else:
        if args.backup_dir is None:
            parser.error('rollback needs --backup-dir')
        report = rollback(args.llama_root, args.backup_dir, args.report)
    print(json.dumps({'status': report['status'],
                      **({'error': report['error']} if 'error' in report else {}),
                      **({'next': report['next']} if 'next' in report else {}),
                      'report': str(args.report)}, indent=2))
    return 0 if report['status'] in ('ACTIVATED_VERIFIED_PENDING_APP_RESTART',
                                     'ROLLED_BACK_VERIFIED', 'ALREADY_MATCHES_BACKUP') else 1


if __name__ == '__main__':
    raise SystemExit(main())
