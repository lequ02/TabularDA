"""Prepare reviewable cleanup lists and preserve small historical ZIP snapshots.

Does not delete, move, rewrite, or load any experiment asset.
"""
import csv
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('D:/SummerResearch')
AUDIT = ROOT / 'audit'
SNAPSHOTS = AUDIT / 'experiment_cleanup_archive_snapshots_20261010'
SNAPSHOTS.mkdir(exist_ok=True)
digest_cache = {}


def load(name):
    return json.loads((AUDIT / name).read_text(encoding='utf-8-sig'))


def digest(path):
    before = path.stat()
    key = (str(path.resolve()), before.st_size, before.st_mtime_ns)
    if key in digest_cache:
        return digest_cache[key]
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Artifact changed during verification: ' + str(path))
    digest_cache[key] = value.hexdigest()
    return digest_cache[key]


remote = load('experiment_cleanup_dependency_audit_2026_10_10.json')
duplicates = load('experiment_cleanup_local_duplicate_hashes_2026_10_10.json')
backups = load('experiment_cleanup_loose_backup_audit_2026_10_10.json')
archives = load('experiment_cleanup_backup_archive_audit_2026_10_10.json')
retention = load('experiment_cleanup_archive_retention_plan_2026_10_10.json')
actions = []

for group in duplicates:
    for old in group['paths']:
        if not (old.endswith('.pkl') and ('/old/' in old or '/old_split/' in old)):
            continue
        keep = next(path for path in group['paths'] if '/old/' not in path and '/old_split/' not in path)
        assert digest(ROOT / old) == digest(ROOT / keep) == group['sha256']
        actions.append({'host': 'local', 'path': str(ROOT / old), 'relative_path': old,
                        'action': 'REMOVE_DUPLICATE_AFTER_FINAL_RECHECK', 'bytes': group['bytes_per_file'],
                        'sha256': group['sha256'], 'keep_path': str(ROOT / keep),
                        'keep_sha256': group['sha256'],
                        'reason': 'Identical ignored legacy checkpoint; targeted source/notebook search found no consumer of the old model directory.'})
assert len(actions) == 4

for row in backups['files']:
    assert row['exact_restore_from_current_with_crlf']
    assert digest(ROOT / row['path']) == row['sha256']
    assert digest(ROOT / row['original_path']) == row['current_sha256']
    actions.append({'host': 'local', 'path': str(ROOT / row['path']), 'relative_path': row['path'],
                    'action': 'REMOVE_REDUNDANT_BACKUP_AFTER_RESTORATION_CHECK', 'bytes': row['bytes'],
                    'sha256': row['sha256'], 'keep_path': str(ROOT / row['original_path']),
                    'keep_sha256': row['current_sha256'], 'reason': 'Entire original file exactly reproduced from retained bytes by LF-to-CRLF conversion.',
                    'restoration': backups['restoration_rule']})

protected = {'.cache/remote_intrusion/latest_comparison.zip',
             '.cache/remote_intrusion/latest_comparison_mix.zip',
             '.cache/remote_intrusion/comparison_d2_scores.zip'}
protected_members = {}
for archive in archives['archives']:
    if archive['path'] in protected:
        for entry in archive['entries']:
            protected_members[(entry['member'], entry['sha256'])] = archive['path']

archive_manifest = []
historical_checker_archives = {
    '.cache/remote_intrusion/comparison_before_refresh_20261007_233801/.cache/remote_intrusion/latest_comparison.zip',
    '.cache/remote_intrusion/comparison_before_refresh_20261007_233801/.cache/remote_intrusion/latest_comparison_mix.zip',
}
for index, path in enumerate(retention['remaining_archives'], 1):
    if path in historical_checker_archives:
        continue
    archive = next(a for a in archives['archives'] if a['path'] == path)
    mappings = []
    with zipfile.ZipFile(ROOT / path) as source:
        for entry in archive['entries']:
            name = entry['member']
            if name in {'audit/latest_comparison_snapshot.json', 'audit/latest_comparison_mix_snapshot.json'}:
                payload = source.read(name)
                assert hashlib.sha256(payload).hexdigest() == entry['sha256']
                saved = SNAPSHOTS / f'{index:02d}_{Path(name).name}'
                saved.write_bytes(payload)
                mappings.append({'member': name, 'sha256': entry['sha256'], 'keep_file': str(saved)})
            elif entry['matches_current_file_bytes']:
                assert digest(ROOT / name) == entry['sha256']
                mappings.append({'member': name, 'sha256': entry['sha256'], 'keep_file': str(ROOT / name)})
            else:
                retained_archive = protected_members[(name, entry['sha256'])]
                mappings.append({'member': name, 'sha256': entry['sha256'],
                                 'keep_archive': str(ROOT / retained_archive), 'keep_member': name})
    archive_hash = digest(ROOT / path)
    archive_manifest.append({'original_archive': str(ROOT / path), 'original_sha256': archive_hash,
                             'members': mappings,
                             'restoration_limit': 'Member files and snapshots restore exactly. Repacked ZIP container bytes need not reproduce the original ZIP hash.'})
    actions.append({'host': 'local', 'path': str(ROOT / path), 'relative_path': path,
                    'action': 'REMOVE_OLD_ARCHIVE_AFTER_SNAPSHOT_AND_MEMBER_CHECK', 'bytes': archive['bytes'],
                    'sha256': archive_hash, 'keep_path': str(SNAPSHOTS),
                    'keep_sha256': '', 'reason': 'Every member has a retained byte-identical source; unique snapshot metadata preserved separately. Historical ZIP hashes are retained in the manifest.'})

(AUDIT / 'experiment_cleanup_archive_member_manifest_2026_10_10.json').write_text(
    json.dumps(archive_manifest, indent=2) + '\n', encoding='utf-8')

# Preserve the distinction between no static reference and an actually unused asset.
cache_candidates = load('experiment_cleanup_remote_duplicate_candidates_2026_10_10.json')
for row in cache_candidates:
    if '/.cache/news_log_v1/' in row['path']:
        row['decision'] = 'KEEP_DYNAMIC_RUNNER_DEPENDENCY'
        row['reason'] = 'run_news_log_experiment.py:194-209 constructs this path and verifies it against the completion marker; removing it breaks cached labeler reuse.'
    elif '/preserved_partial/downstream/' in row['path']:
        row['decision'] = 'REVIEW_EXACT_DUPLICATE_PARTIAL_BACKUP'
        row['reason'] = 'Identical completed canonical checkpoint; preserve recovery ledger/logs and recheck closed queue before considering this small backup.'
    else:
        row['decision'] = 'HOLD_VALIDATION_FIXTURE'
        row['reason'] = 'Identical data copies, but absence of a literal path reference does not establish the diagnostic fixture is retired.'
(AUDIT / 'experiment_cleanup_remote_duplicate_candidates_2026_10_10.json').write_text(
    json.dumps(cache_candidates, indent=2) + '\n', encoding='utf-8')

pilot_decisions = []
for namespace in remote['pilot_namespaces']:
    name = namespace['namespace']
    for path, meta in remote['inventory'].items():
        if not any(path.startswith(area + '/' + name + '/') for area in ('data', 'output', 'sdv trained model')):
            continue
        if name == 'census_weighted_pilot_20261004':
            reason = 'Three verified imports cite original pilot records/hashes; preserve import and historical analysis evidence.'
        elif name == 'pilot_ctgan_v1':
            reason = '45 full-budget historical runs; saved dependencies and leakage-audit consumers remain.'
        elif name == 'credit_weighted_macro_f1_pilot_20261006':
            reason = 'Three completed weighted diagnostic runs; preserve the distinct procedure and findings.'
        elif name == 'news_log_pilot_20261008_0048':
            reason = 'Three distinct log-target diagnostic runs and historical report snapshots; superseded main-panel scores do not retire evidence.'
        else:
            reason = 'Smoke evidence, not production. No scientific deletion justification; tiny collection retained separately.'
        pilot_decisions.append({'host': 'remote', 'path': str(Path(remote['root']) / path),
                                'bytes': meta['bytes'], 'decision': 'KEEP', 'namespace': name,
                                'reason': reason, 'literal_reference_count': len(meta['consumers'])})

with (AUDIT / 'experiment_cleanup_pilot_file_decisions_2026_10_10.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(pilot_decisions[0]))
    writer.writeheader()
    writer.writerows(pilot_decisions)

manifest = {'prepared_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
            'status': 'PROPOSED_ONLY_NO_DELETIONS', 'local_proposed_files': len(actions),
            'logical_bytes_in_proposal': sum(row['bytes'] for row in actions),
            'new_snapshot_bytes': sum(p.stat().st_size for p in SNAPSHOTS.iterdir() if p.is_file()),
            'requirements_before_execution': ['Recheck current file/keeper hashes, relevant working-tree changes, live jobs, and path consumers.',
                'Preserve retained canonical files and current ZIP archives used by the restoration maps.',
                'Preserve all saved snapshot/member manifests; an old ZIP container cannot necessarily be restored byte-for-byte.',
                'Do not delete pilot namespaces, runtime labeler caches, completed weights, raw source caches, or feature-stage inputs.'],
            'actions': actions}
(AUDIT / 'experiment_cleanup_review_manifest_2026_10_10.json').write_text(
    json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
with (AUDIT / 'experiment_cleanup_proposed_files_2026_10_10.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=('host', 'path', 'action', 'bytes', 'sha256', 'keep_path', 'keep_sha256', 'reason'), extrasaction='ignore')
    writer.writeheader()
    writer.writerows(actions)
print(json.dumps({'local_proposed_files': len(actions),
                  'logical_GiB': manifest['logical_bytes_in_proposal'] / 2**30,
                  'preserved_snapshot_bytes': manifest['new_snapshot_bytes'],
                  'pilot_files_kept': len(pilot_decisions),
                  'actions': {kind: sum(r['action'] == kind for r in actions) for kind in {r['action'] for r in actions}}}, indent=2))
