"""Build review catalogs from saved remote metadata and lightweight local checks."""
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(20 * 1024 * 1024)
DEST = ROOT / 'audit' / 'cleanup_review_20261013'
DEST.mkdir(exist_ok=True)
snapshot = json.loads((ROOT / 'audit/cleanup_recheck_20261013_remote_snapshot.json').read_text(encoding='utf-8-sig'))
assets = snapshot['assets']
rows = snapshot['rows']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_csv(name, data, fields):
    with (DEST / name).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def link(path, label):
    return f'[{label}](<{Path(path).as_posix()}>)'


for item in assets.values():
    local = ROOT / item['relative_path'] if item['relative_path'] else None
    item['local_path'] = str(local) if local else ''
    item['local_present'] = bool(local and local.is_file())

local_record_checks = []
for row in rows:
    local = ROOT / row['record_relative_path']
    row['local_record_present'] = local.is_file()
    row['local_record_hash_matches_remote'] = None
    if local.is_file() and row.get('record_sha256'):
        row['local_record_hash_matches_remote'] = sha(local) == row['record_sha256']
        row['local_record_lf_hash_matches_remote'] = row['local_record_hash_matches_remote'] or hashlib.sha256(local.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == row['record_sha256']
        local_record_checks.append({'path': str(local), 'matches': row['local_record_hash_matches_remote'], 'matches_after_lf_normalization': row['local_record_lf_hash_matches_remote']})

asset_rows = [{**{k: item[k] for k in ('remote_path', 'resolved_remote_path', 'relative_path', 'remote_present', 'bytes', 'local_path', 'local_present')},
               'roles': ';'.join(item['roles']),
               'local_link': link(item['local_path'], Path(item['local_path']).name) if item['local_present'] else '',
               'retention': 'keep_current_evidence_or_replay_dependency',
               'validation': 'presence_only; artifact_bytes_not_compared'} for item in assets.values()]
write_csv('artifact_locations.csv', asset_rows, list(asset_rows[0]))

run_fields = ['dataset', 'seed', 'generator', 'construction', 'labeler', 'mode', 'method', 'scope',
              'matrix_cell', 'status', 'record_relative_path', 'record_sha256', 'selected_dev_epoch',
              'local_record_present', 'local_record_hash_matches_remote', 'errors', 'test_scores',
              'local_record_lf_hash_matches_remote', 'target_normalization', 'generator_parameters', 'assets']
run_rows = [{k: json.dumps(row.get(k), sort_keys=True) if isinstance(row.get(k), (dict, list)) else row.get(k, '')
             for k in run_fields} for row in rows]
write_csv('matrix_runs.csv', run_rows, run_fields)
missing = [r for r in run_rows if r['matrix_cell'] and r['status'] != 'complete_presence_checked']
write_csv('missing_runs.csv', missing, run_fields)


def asset_links(row, roles):
    keys = list(dict.fromkeys(key for role in roles for key in row['assets'].get(role, [])))
    if not keys:
        return '—'
    links = []
    remote_count = 0
    for key in keys:
        item = assets[key]
        if item['local_present']:
            name = Path(item['local_path']).name
            stem = row['run_name'] + '.'
            label = name.removeprefix(stem)
            if len(label) > 60:
                label = label[-60:]
            links.append(link(item['local_path'], label))
        elif item['remote_present']:
            remote_count += 1
        else:
            links.append('missing: ' + Path(key).name)
    if remote_count:
        links.append(f'{remote_count} server-only')
    return '<br>'.join(links)


for dataset in snapshot['datasets']:
    lines = [f'# {dataset}: run artifacts', '',
             f"Checked {snapshot['checked_at_chicago']} (Chicago).", '',
             'Links point to existing local files. “Server-only” means the exact saved and resolved server paths are in '
             '[artifact_locations.csv](artifact_locations.csv), joined through the `assets` field in [matrix_runs.csv](matrix_runs.csv). '
             'No weights or row tables were downloaded. Local asset presence does not establish byte identity; local run-record hashes were compared.', '',
             'An absent expected record does not establish that all its generator or data inputs are absent. Separate TabDDPM real-only references are listed below the matrix and excluded from its 1,168 cells.', '']
    for reference_only in (False, True):
        selected = [r for r in rows if r['dataset'] == dataset and r['matrix_cell'] != reference_only]
        if not selected:
            continue
        lines += ['## Additional TabDDPM real-only references' if reference_only else '## Expected matrix', '',
                  '| Seed / generator / source / target / mode | Status | Record | Synthetic / predictions CSV | Training plots | Selected weights | Generator / provenance | Labeler / quality | Real splits / manifest | Logs / other outputs |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in selected:
            groups = [
                ('run_record',), ('synthetic_path', 'predictions_path'), ('training_plot',),
                ('downstream_weight_path',), ('generator_model_path', 'generator_provenance_path', 'generator_output'),
                ('predictor_model_path', 'synthetic_quality_path', 'dnn_dev_report_path'),
                tuple(k for k in r['assets'] if k.startswith('real_')) + ('split_manifest_path',),
                ('arm_log', 'run_output', 'epoch_history', 'metric_sidecar'),
            ]
            identity = f"{r['seed']} / {r['generator']} / {r['construction']} / {r['labeler']} / {r['mode']}"
            lines.append('| ' + ' | '.join([identity, r['status']] + [asset_links(r, group) for group in groups]) + ' |')
        lines.append('')
    (DEST / (dataset + '_artifacts.md')).write_text('\n'.join(lines), encoding='utf-8')

# Reconcile the earlier proposal with the receipt; never treat it as a new allowlist.
proposal = json.loads((ROOT / 'audit/experiment_cleanup_review_manifest_2026_10_10.json').read_text())
receipt = [json.loads(line) for line in (ROOT / 'audit/deletion_receipt_2026_10_10.jsonl').read_text().splitlines()]
deleted = {r['path']: r for r in receipt if r['event'] == 'deleted'}
cleanup_rows = []
for item in proposal['actions']:
    keeper = Path(item['keep_path']) if item.get('keep_path') else None
    target = Path(item['path'])
    keeper_hash = sha(keeper) if keeper and keeper.is_file() else ''
    restoration = None
    if item['action'] == 'REMOVE_REDUNDANT_BACKUP_AFTER_RESTORATION_CHECK' and keeper and keeper.is_file():
        original_bytes = keeper.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        restoration = hashlib.sha256(original_bytes).hexdigest() == item['sha256']
    cleanup_rows.append({
        'original_path': str(target), 'original_action': item['action'], 'bytes': item['bytes'],
        'in_deletion_receipt': str(target) in deleted, 'target_present_now': target.is_file(),
        'keeper_path': str(keeper) if keeper else '', 'keeper_present_now': bool(keeper and keeper.exists()),
        'keeper_hash_matches_saved': keeper_hash == item['keep_sha256'] if item.get('keep_sha256') else None,
        'backup_original_bytes_recoverable': restoration,
        'decision_now': 'already_deleted_not_a_new_candidate' if str(target) in deleted and not target.exists() else 'review_changed_state',
    })
write_csv('prior_cleanup_recheck.csv', cleanup_rows, list(cleanup_rows[0]))
members = json.loads((ROOT / 'audit/experiment_cleanup_archive_member_manifest_2026_10_10.json').read_text())
unique_members = {(m['keep_file'], m['sha256']) for archive in members for m in archive['members']}
member_checks = []
for path, expected in sorted(unique_members):
    exists = Path(path).is_file()
    member_checks.append({'retained_path': path, 'present': exists, 'expected_sha256': expected,
                          'matches': exists and sha(path) == expected})
write_csv('retained_archive_members_recheck.csv', member_checks, list(member_checks[0]))

# Refresh CSV discovery, keeping the old purpose annotations as historical evidence.
# Inspect headers only; no local experiment/data diagnostics or model operations.
with (ROOT / 'audit/local_csv_purpose_catalog_2026_10_10.csv').open(encoding='utf-8-sig', newline='') as handle:
    old_catalog = {str(Path(r['path']).resolve()).lower(): r for r in csv.DictReader(handle)}
paths = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', '*.csv',
                        '-g', '!.git/**', '-g', '!.worktrees/**', '-g', '!**/node_modules/**',
                        '-g', '!**/site-packages/**', '-g', '!**/.git/**',
                        '-g', '!audit/cleanup_review_20261013/**'], cwd=ROOT,
                       capture_output=True, text=True, check=True).stdout.splitlines()
csv_rows = []
hash_groups = defaultdict(list)
for rel in sorted(paths):
    path = ROOT / rel
    if not path.is_file():
        continue
    old = old_catalog.get(str(path.resolve()).lower())
    size = path.stat().st_size
    # Header inspection is bounded even for large tables and invalid encodings surface.
    with path.open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle), [])
    digest = sha(path) if size <= 5 * 1024 * 1024 else ''
    if digest:
        hash_groups[digest].append(rel)
    csv_rows.append({'relative_path': rel.replace('\\', '/'), 'local_path': str(path),
                     'local_link': link(path, path.name), 'bytes': size,
                     'in_prior_catalog': bool(old), 'prior_study': old['study'] if old else '',
                     'prior_scientific_role': old['scientific_role'] if old else '',
                     'prior_treatment': old['proposed_treatment'] if old else '',
                     'header_json': json.dumps(header), 'sha256_if_at_most_5mib': digest,
                     'current_decision': 'retain_current_or_historical_evidence_pending_consumer_review'})
write_csv('current_csv_catalog.csv', csv_rows, list(csv_rows[0]))
duplicate_rows = [{'sha256': digest, 'copies': len(paths), 'relative_paths_json': json.dumps(paths),
                   'disposition': 'candidate_only_update_readers_and_preserve_provenance_before_consolidation'}
                  for digest, paths in sorted(hash_groups.items()) if len(paths) > 1]
write_csv('small_csv_duplicate_candidates.csv', duplicate_rows, ['sha256', 'copies', 'relative_paths_json', 'disposition'])

report_checks = {}
prior_validation = json.loads((ROOT / 'audit/comparison_tabddpm_refresh_2026_10_13_final.json').read_text())
report_source_sets = []
for mode in ('synthetic', 'mix'):
    stem = 'dataset_pipeline_comparison' + ('_mix' if mode == 'mix' else '')
    expected = prior_validation['report_validation'][mode]
    report_checks[mode] = {}
    for suffix, key in (('csv', 'csv_sha256'), ('md', 'markdown_sha256'), ('png', 'plot_sha256')):
        path = ROOT / 'output/comparisons' / (stem + '.' + suffix)
        report_checks[mode][suffix + '_hash_matches'] = sha(path) == expected[key]
    with (ROOT / 'output/comparisons' / (stem + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        data = list(reader)
        refs = {r[k] for r in data for k in reader.fieldnames if k.endswith('_record') and r[k]}
        report_source_sets.append(refs)
        report_checks[mode].update({'data_rows': len(data), 'columns': len(reader.fieldnames), 'distinct_source_records': len(refs)})

matrix = [r for r in rows if r['matrix_cell']]
summary = {
    'checked_at_chicago': snapshot['checked_at_chicago'], 'matrix_status': snapshot['summary'],
    'matrix_cells': len(matrix), 'additional_tabddpm_real_only_references': len(rows) - len(matrix),
    'missing_groups': dict(Counter(r['generator'] + '/' + r['dataset'] for r in matrix if r['status'] == 'absent_record')),
    'local_record_checks': {'checked': len(local_record_checks), 'mismatches': [r for r in local_record_checks if not r['matches']]},
    'artifact_locations': {'saved_path_entries': len(assets), 'unique_resolved_paths': len({a['resolved_remote_path'] for a in assets.values()}),
                           'local_present_path_entries': sum(a['local_present'] for a in assets.values()),
                           'server_only_path_entries': sum(a['remote_present'] and not a['local_present'] for a in assets.values()),
                           'local_artifact_bytes_compared': False},
    'prior_cleanup': {'receipt_deleted_files': len(deleted), 'receipt_deleted_bytes': sum(r['bytes'] for r in deleted.values()),
                      'all_231_targets_still_absent': all(not r['target_present_now'] for r in cleanup_rows),
                      'keeper_hash_mismatches': [r['keeper_path'] for r in cleanup_rows if r['keeper_hash_matches_saved'] is False],
                      'loose_backup_restoration_failures': [r['original_path'] for r in cleanup_rows if r['backup_original_bytes_recoverable'] is False],
                      'retained_member_mappings_checked': len(member_checks),
                      'retained_member_failures': [r for r in member_checks if not r['matches']]},
    'csv_discovery': {'research_csvs': len(csv_rows), 'not_in_prior_catalog': sum(not r['in_prior_catalog'] for r in csv_rows),
                      'small_exact_duplicate_groups': len(duplicate_rows),
                      'redundant_copies_in_small_hash_groups': sum(r['copies'] - 1 for r in duplicate_rows),
                      'full_csv_content_classification_refreshed': False, 'large_csv_hashes_computed': False},
    'comparison_report_checks': report_checks,
    'distinct_source_records_across_both_current_comparisons': len(set.union(*report_source_sets)),
    'active_research_processes': snapshot['active_research_processes'],
    'no_deletions_moves_training_or_restarts_in_this_review': True,
}
(DEST / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
