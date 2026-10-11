"""Prepare proposed paths from saved inventories; never move experiment assets."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit'
inventory_source = AUDIT / 'experiment_cleanup_dependency_audit_2026_10_10.json'
snapshot = json.loads(inventory_source.read_text(encoding='utf-8-sig'))
inventory = snapshot['inventory']
selected = set()
comparison_counts = {}
for name in ('dataset_pipeline_comparison.csv', 'dataset_pipeline_comparison_mix.csv'):
    with (ROOT / 'output/comparisons' / name).open(newline='', encoding='utf-8-sig') as source:
        records = {v for row in csv.DictReader(source) for v in row.values()
                   if v and v.endswith('.run.json')}
    comparison_counts[name] = len(records)
    selected.update(records)
statistical = json.loads((AUDIT / 'statistical_328_20261010/snapshot/snapshot.json').read_text())
assert selected == {r['path'] for r in statistical['records']}
assert len(selected) == 406
assert selected <= inventory.keys()

# Saved JSON consumers include nested prepared paths and original import records.
# This graph does not infer that historical source-code references require retention.
edges = defaultdict(set)
for path, item in inventory.items():
    for consumer in item['consumers']:
        kind, origin = consumer.split(':', 1)
        if kind != 'source' and origin in inventory:
            edges[origin].add(path)
active = set(selected)
active.update(p.replace('.run.json', '.d2.json') for p in selected
              if p.replace('.run.json', '.d2.json') in inventory)
for name in ('comparison_rerun_verification_latest.json',
             'comparison_news_log_fit_table_hashes.json',
             'comparison_housing_fit_table_hashes.json'):
    active.add('audit/' + name)
queue = deque(sorted(active))
while queue:
    for dependency in edges[queue.popleft()]:
        if dependency not in active:
            active.add(dependency)
            queue.append(dependency)

hashes = {}
duplicate_group = {}
for i, group in enumerate(snapshot['duplicate_groups'], 1):
    for member in group['members']:
        hashes[member['path']] = group['sha256']
        duplicate_group[member['path']] = i
for r in statistical['records']:
    hashes[r['path']] = r['sha256']

run_findings = {}
with (AUDIT / 'completed_run_preprocessing_findings_2026_10_09.csv').open(newline='', encoding='utf-8-sig') as source:
    for finding in csv.DictReader(source):
        if finding['implementation_defects']:
            relative = finding['record'].removeprefix(snapshot['root'] + '/')
            run_findings[relative] = finding['implementation_defects']

def owner(path):
    parts = PurePosixPath(path).parts
    if parts[0] in ('data', 'output', 'sdv trained model'):
        return parts[1]
    return parts[1] if len(parts) > 1 else 'unclassified'

def role(path):
    name = PurePosixPath(path).name
    if name.endswith('.run.json'): return 'run_record'
    if name.endswith('.d2.json'): return 'derived_metric_provenance'
    if name == 'split_manifest.json': return 'split_manifest'
    if name.endswith('.provenance.json'): return 'generator_provenance'
    if name.endswith('.quality.json'): return 'synthetic_quality'
    if path.startswith('.cache/news_log_runtime_20261008/.cache/news_log_v1/') and name.endswith('.complete.json'):
        return 'labeler_completion_provenance'
    if name.endswith('.predictor.pkl') or name.endswith('.predictor.pt'): return 'labeler_model'
    if path.startswith('sdv trained model/') and name.endswith('.pkl'): return 'generator_model'
    if name.endswith('.weights.pth'): return 'downstream_checkpoint'
    if name.endswith(('.pkl', '.pt', '.pth')): return 'model_or_serialized_cache_review'
    if name.endswith('.predictions.csv'): return 'test_predictions'
    if name.endswith('.epochs.csv'): return 'epoch_history'
    if re.search(r'_real_(train|dev|test)_(raw|onehot)\.csv$', name): return 'prepared_split'
    if path.startswith('data/') and re.search(r'_seed\d+_(ctgan|tvae|tabddpm)_.+_100k\.csv$', name): return 'synthetic_table'
    if '/features/' in path and name.endswith('.csv'): return 'features_only_table'
    if path.startswith('.cache/news_log_runtime_20261008/.cache/news_log_v1/'):
        if name.endswith('.csv'): return 'labeler_combined_table'
        if name.endswith(('.pt', '.pkl')): return 'labeler_model'
    if path.startswith('output/simulated_all_20260930/') and name.endswith('.csv'):
        if '/seed_' in path: return 'simulated_split_or_sample'
        return 'simulated_result_table'
    if path.startswith('output/comparisons/') and name.endswith('.csv'): return 'comparison_table'
    if '/results/' in path or name in ('per_run.csv', 'summary.csv', 'scores.csv'):
        return 'result_table'
    if path.startswith('audit/') and name.endswith('.csv'): return 'audit_table_review'
    if path.startswith('data/') and name.endswith('.csv'): return 'legacy_table_review'
    return 'table_review'

def lifecycle(path):
    scope = owner(path)
    if path in active:
        if scope == 'census_weighted_pilot_20261004': return 'current_import_origin'
        return 'current_result_dependency'
    if path.startswith('.cache/news_log_runtime_20261008/.cache/news_log_v1/'):
        return 'reusable_news_labeler_asset'
    # Include companion epoch histories, reports and plots in a retained run bundle.
    peer = re.sub(r'\.(epochs\.csv|predictions\.csv|weights\.pth)$', '.run.json', path)
    if '/weight/' in peer: peer = peer.replace('/weight/', '/acc/')
    if peer in selected: return 'current_run_companion'
    if scope == 'simulated_all_20260930': return 'current_simulated_study'
    if scope in ('mnist_head_fixed_20261009', 'housing_no_faker_20261009', 'news_log_v1',
                 'census_kdd_weighted_macro_f1_20261005'):
        return 'completed_study_extension_not_report_selected'
    if scope in ('pilot_ctgan_v1', 'census_weighted_pilot_20261004'):
        return 'historical_pilot_bundle'
    if 'pilot' in scope or 'dropout' in scope: return 'diagnostic_study'
    if 'smoke' in scope or 'validation' in scope or 'test_tools' in scope:
        return 'validation_fixture_review'
    if path.startswith('data/corrected_v2/') or path.startswith('output/corrected_v2/'):
        dataset = PurePosixPath(path).parts[2]
        if dataset in ('mnist12', 'mnist28', 'news', 'california_housing'):
            return 'superseded_or_shared_asset_review'
        return 'completed_original_matrix_extension_review'
    if path.startswith('data/corrected_v2_seed42_mnist28_news/') or path.startswith('output/corrected_v2_seed42_mnist28_news/'):
        return 'superseded_or_shared_asset_review'
    if 'statistical' in path: return 'analysis_release_or_snapshot_review'
    if path.startswith('output/comparisons/'):
        return 'report_or_historical_reference_review'
    if path.startswith('audit/'): return 'historical_evidence_or_validator_input_review'
    if path.startswith('.cache/'): return 'runtime_asset_or_intermediate_review'
    return 'legacy_or_unknown_review'

def propose(path, kind):
    p = PurePosixPath(path)
    parts = p.parts
    if path.startswith('.cache/news_log_runtime_20261008/.cache/news_log_v1/'):
        seed = parts[-2]
        if kind == 'labeler_model': prefix = 'artifacts/models/labelers'
        elif kind == 'labeler_combined_table': prefix = 'artifacts/tables/real'
        elif kind == 'labeler_completion_provenance': prefix = 'artifacts/provenance'
        else: return ''
        tail = ['combined_relabeling'] if kind == 'labeler_combined_table' else []
        return str(PurePosixPath(prefix, 'news_log_v1', 'news', seed, *tail, p.name))
    if kind in ('prepared_split', 'synthetic_table', 'synthetic_quality', 'split_manifest'):
        category = {'prepared_split': 'prepared', 'split_manifest': 'prepared',
                    'synthetic_table': 'synthetic', 'synthetic_quality': 'synthetic'}[kind]
        return str(PurePosixPath('artifacts/tables/real', *parts[1:-1], category, p.name))
    if kind in ('labeler_model', 'generator_model', 'generator_provenance') and parts[0] in ('data', 'sdv trained model'):
        category = 'labelers' if kind == 'labeler_model' else 'generators'
        return str(PurePosixPath('artifacts/models', category, *parts[1:]))
    if kind in ('test_predictions', 'epoch_history', 'downstream_checkpoint', 'run_record', 'derived_metric_provenance') and parts[0] == 'output' and len(parts) >= 5:
        match = re.search(r'_seed(\d+)_', p.name)
        if match:
            seed = 'seed_' + match.group(1)
            if kind == 'downstream_checkpoint': prefix = 'artifacts/models/downstream'
            elif kind in ('run_record', 'derived_metric_provenance'): prefix = 'artifacts/records'
            else: prefix = 'artifacts/tables/real'
            tail = [] if kind in ('downstream_checkpoint', 'run_record', 'derived_metric_provenance') else [kind]
            return str(PurePosixPath(prefix, parts[1], parts[2], seed, *tail, p.name))
    if parts[:2] == ('output', 'simulated_all_20260930') and len(parts) == 5 and kind == 'simulated_split_or_sample':
        dataset, seed_dir, name = parts[2:]
        seed = seed_dir.removeprefix('seed_')
        stem = p.stem
        if stem in ('train', 'test'):
            category, renamed = 'prepared', f'{dataset}_seed{seed}_real_{stem}_raw.csv'
        elif stem == 'original':
            category, renamed = 'evaluation_tables', f'{dataset}_seed{seed}_real_original_10k.csv'
        elif stem in ('ctgan', 'tvae'):
            category, renamed = 'synthetic', f'{dataset}_seed{seed}_{stem}_full_generated_10k.csv'
        else:
            match = re.fullmatch(r'(ctgan|tvae)-(joint|features)(?:-(.+))?', stem)
            if not match: return ''
            generator, construction, labeler = match.groups()
            fit = 'full' if construction == 'joint' else 'xonly'
            category = 'synthetic' if labeler else 'features'
            renamed = f'{dataset}_seed{seed}_{generator}_{fit}_{labeler or "features"}_10k.csv'
        return str(PurePosixPath('artifacts/tables/simulated', parts[1], dataset, seed_dir, category, renamed))
    if kind == 'simulated_result_table':
        return str(PurePosixPath('artifacts/tables/results/simulated_all_20260930', p.name))
    if kind == 'comparison_table':
        return str(PurePosixPath('artifacts/tables/results/real_comparison', *parts[2:]))
    # A review classification is deliberately not a mechanical migration instruction.
    return ''

rows = []
included_suffixes = ('.csv', '.pkl', '.pth', '.pt', '.run.json', '.d2.json',
                     '.provenance.json', '.quality.json')
for path, item in sorted(inventory.items()):
    news_marker = path.startswith('.cache/news_log_runtime_20261008/.cache/news_log_v1/') and path.endswith('.complete.json')
    if not path.endswith(included_suffixes) and PurePosixPath(path).name != 'split_manifest.json' and not news_marker:
        continue
    kind = role(path)
    proposed = propose(path, kind)
    state = lifecycle(path)
    peer = re.sub(r'\.(epochs\.csv|predictions\.csv|weights\.pth)$', '.run.json', path).replace('/weight/', '/acc/')
    if kind == 'model_or_serialized_cache_review' or 'review' in kind:
        proposed = ''
    rows.append(dict(host='research_server', old_path=path, role=kind,
                     lifecycle=state, proposed_path=proposed, bytes=item['bytes'],
                     known_sha256=hashes.get(path, ''), exact_duplicate_group=duplicate_group.get(path, ''),
                     saved_consumer_count=len(item['consumers']),
                     current_dependency=path in active,
                     historical_run_findings=run_findings.get(peer, ''),
                     decision='PROPOSE_LOCATION_ONLY' if proposed else 'REVIEW_OWNER_AND_PURPOSE',
                     reason='Reachable from selected records/current validation evidence' if path in active else
                     'Classified by inspected writer/path pattern; final retention not authorized'))

# Local-only current analysis files were added after the comprehensive remote snapshot.
# Include CSV locations without equating copied tables with independent observations.
local_extra = []
for folder in ('output/statistical_analysis_328_20261010', 'audit/statistical_328_20261010'):
    for path in sorted((ROOT / folder).rglob('*.csv')):
        relative = path.relative_to(ROOT).as_posix()
        proposed = ''
        if folder.startswith('output/'):
            proposed = str(PurePosixPath('artifacts/tables/results/real_statistics_20261010', *PurePosixPath(relative).parts[2:]))
        kind = 'frozen_analysis_input' if '/snapshot/' in relative or '/frozen_inputs/' in relative else 'analysis_table_review'
        current = (any(part in PurePosixPath(relative).parts for part in ('synthetic', 'mix', 'separate_results', 'snapshot', 'frozen_inputs'))
                   or path.name in ('contrast_weights.csv', 'original_references.csv', 'regression_d2_scores.csv'))
        row = dict(host='local_workspace', old_path=relative, role=kind,
                   lifecycle='current_analysis_bundle_review' if current else 'historical_pooled_analysis_table_review', proposed_path=proposed,
                   bytes=path.stat().st_size, known_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   exact_duplicate_group='', saved_consumer_count='', current_dependency='',
                   historical_run_findings='',
                   decision='PROPOSE_LOCATION_ONLY' if proposed else 'REVIEW_OWNER_AND_PURPOSE',
                   reason='Local analysis input/output inspected; distinguish current separate analysis from superseded pooled tables')
        rows.append(row)
        local_extra.append(relative)

destinations = Counter((r['host'], r['proposed_path']) for r in rows if r['proposed_path'])
assert all(n == 1 for n in destinations.values()), 'Proposed path collision'
with (AUDIT / 'repository_organization_file_catalog_2026_10_10.csv').open('w', newline='', encoding='utf-8') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = dict(status='REVIEW_ONLY_NO_ASSET_MOVES_OR_DELETIONS',
               inventory_at_chicago=snapshot['checked_at_chicago'],
               refreshed_remote_at_chicago=json.loads((AUDIT / 'repository_organization_fresh_remote_2026_10_10.json').read_text(encoding='utf-8-sig'))['checked_at_chicago'],
               comparison_selected_records=comparison_counts, distinct_selected_records=len(selected),
               statistical_snapshot_matches_comparison_union=True,
               saved_dependency_closure_files=len(active),
               catalog_rows=len(rows), remote_csv_rows=sum(r['host']=='research_server' and r['old_path'].endswith('.csv') for r in rows),
               local_analysis_csv_rows=len(local_extra),
               proposed_locations=sum(bool(r['proposed_path']) for r in rows),
               unresolved_owner_or_purpose=sum(not r['proposed_path'] for r in rows),
               lifecycle_counts=dict(Counter(r['lifecycle'] for r in rows)),
               role_counts=dict(Counter(r['role'] for r in rows)),
               remote_output_selected_weight_files=sum(p.startswith('output/') and p.endswith('.weights.pth') for p in inventory),
               inspected_exact_duplicate_groups=len(snapshot['duplicate_groups']),
               repeated_logical_bytes=sum(g['bytes_per_file'] * (len(g['members']) - 1) for g in snapshot['duplicate_groups']),
               repeated_allocated_bytes_after_inode_accounting=sum(
                   sum(sizes.values()) - max(sizes.values())
                   for g in snapshot['duplicate_groups']
                   for sizes in [{(inventory[m['path']]['device'], m['inode']): m['allocated_bytes'] for m in g['members']}]),
               proposed_destination_collisions=0,
               scope_limits=['Saved graph is not proof of all external/dynamic consumers.',
                             'Historical source references do not require preserving original layout.',
                             'Remote inventory predates the new separate statistical runtime; local release CSVs supplement it.',
                             'Known hashes reuse prior byte checks; no migration-time rehash is implied.',
                             'Repeated allocated bytes are not guaranteed reclaimable; uninspected links, consumers and retention can reduce savings.',
                             'No CSV payload, model, or run record was changed. Proposed paths are not deletion approval.'])
(AUDIT / 'repository_organization_review_validation_2026_10_10.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
