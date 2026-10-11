"""Inspect local artifacts and final report membership; no experiments or moves."""
import ast
import csv
import hashlib
import itertools
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit'
HERE = AUDIT / 'statistical_328_20261010'
OUT = ROOT / 'output/statistical_analysis_328_20261010'
SCOPES = {'adult': 'corrected_v2', 'census_kdd': 'census_kdd_weighted_macro_f1_20261005',
          'covertype': 'corrected_v2', 'mnist12': 'mnist_head_fixed_20261009',
          'mnist28': 'mnist_head_fixed_20261009', 'news': 'news_log_v1',
          'california_housing': 'housing_no_faker_20261009'}

def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def read_bytes(path):
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    return payload

def sha(payload):
    return hashlib.sha256(payload).hexdigest()

def relative(path):
    return path.relative_to(ROOT).as_posix()

# Use the inspected, pure method-name mapping without importing experiment modules.
tree = ast.parse((ROOT / 'src/modeling/constants.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'method_parts')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<method-name mapping>', 'exec'), namespace)
method_parts = namespace['method_parts']

comparison_sets = {}
comparison_rows = {}
selected = set()
for mode, name in [('synthetic', 'dataset_pipeline_comparison.csv'), ('mix', 'dataset_pipeline_comparison_mix.csv')]:
    rows = read_csv(ROOT / 'output/comparisons' / name)
    paths = {v for row in rows for k, v in row.items() if k.endswith('_record') and v}
    assert len(paths) == 210
    comparison_sets[mode] = paths
    comparison_rows[mode] = rows
    selected.update(paths)

expected_paths = set()
expected_keys = set()
for dataset, seed in itertools.product(SCOPES, (42, 43)):
    expected_keys.add((dataset, seed, 'real', 'real', 'original', 'original'))
    expected_paths.add(f'output/{SCOPES[dataset]}/{dataset}/acc/{dataset}_seed{seed}_real_original.run.json')
    for generator, fit in itertools.product(('ctgan', 'tvae'), ('full', 'xonly')):
        for labeler in (('generated', 'rf', 'xgb', 'dnn') if fit == 'full' else ('rf', 'xgb', 'dnn')):
            for mode in ('synthetic', 'mix'):
                expected_keys.add((dataset, seed, generator, fit, labeler, mode))
                expected_paths.add(f'output/{SCOPES[dataset]}/{dataset}/acc/{dataset}_seed{seed}_{generator}_{fit}_{labeler}_{mode}.run.json')
assert selected == expected_paths
snapshot_path = HERE / 'snapshot/snapshot.json'
snapshot = json.loads(read_bytes(snapshot_path))
snapshot_index = {r['path']: r for r in snapshot['records']}
assert selected == snapshot_index.keys()
assert len(selected) == len(expected_keys) == 406
shared = comparison_sets['synthetic'] & comparison_sets['mix']
assert len(shared) == 14 and all(p.endswith('_real_original.run.json') for p in shared)

file_reason = defaultdict(set)
artifact_roles = defaultdict(set)
result_assets = set()
origin_records = set()
origin_byte_checks = []
missing_mirrors = defaultdict(set)

def add(path, why, role=''):
    path = str(path).replace('\\', '/')
    file_reason[path].add(why)
    if role:
        artifact_roles[path].add(role)

def local_mirror(value):
    if not isinstance(value, str):
        return None
    value = value.replace('\\', '/')
    if value.startswith('D:/SummerResearch/'):
        return value[len('D:/SummerResearch/'):]
    prefix = '/home/thuy/Research/minh_data_synth/TabularDA/'
    if value.startswith(prefix):
        value = value[len(prefix):]
        # Analyze documented runtime directory links; this does not modify readers.
        match = re.match(r'\.cache/[^/]+/(data|output|sdv trained model)/(.+)', value)
        if match:
            value = match.group(1) + '/' + match.group(2)
        return value
    if value.startswith(('data/', 'output/', 'sdv trained model/', 'audit/')):
        return value
    return None

run_rows = []
raw_equal = normalized_equal = 0
current_raw_changed = []
key_set = set()
metric_lookup = {}
per_dataset = defaultdict(Counter)
run_cache = {}
for name in sorted(selected):
    path = ROOT / name
    current_bytes = read_bytes(path)
    frozen_path = HERE / 'snapshot/records' / name
    frozen_bytes = read_bytes(frozen_path)
    assert sha(frozen_bytes) == snapshot_index[name]['sha256'], frozen_path
    record = json.loads(current_bytes)
    assert record == json.loads(frozen_bytes), path
    if current_bytes == frozen_bytes:
        raw_equal += 1
    elif current_bytes.replace(b'\r\n', b'\n') == frozen_bytes.replace(b'\r\n', b'\n'):
        normalized_equal += 1
        current_raw_changed.append(name)
    else:
        raise AssertionError(f'Selected record differs beyond line endings: {name}')
    dataset, seed, mode = record['dataset'], record['seed'], record['train_option']
    generator, fit, labeler = ('real', 'real', 'original') if mode == 'original' else method_parts(record['augment_option'])
    key = (dataset, seed, generator, fit, labeler, mode)
    assert key in expected_keys and key not in key_set
    key_set.add(key)
    assert record['selected_dev_epoch'] is not None
    per_dataset[dataset][mode] += 1
    add(name, 'Selected by current real comparison and statistical analysis', 'source_record')
    add(relative(frozen_path), 'Frozen byte version of selected source record', 'source_record_copy')
    run_rows.append(dict(dataset=dataset, seed=seed, generator=generator, has_y='' if fit == 'real' else int(fit == 'full'),
                         construction=fit, labeler=labeler, training_mode=mode, record=name,
                         frozen_sha256=sha(frozen_bytes), local_sha256=sha(current_bytes),
                         frozen_bytes_equal=current_bytes == frozen_bytes))
    for metric, value in record['test_scores'].items():
        metric_lookup[name, metric] = value
    for field, value in record.items():
        if field.endswith('_path') and value:
            mirrored = local_mirror(value)
            if mirrored:
                role = {'predictions_path': 'heldout_predictions', 'downstream_weight_path': 'selected_checkpoint',
                        'synthetic_path': 'synthetic_table', 'generator_model_path': 'generator_model',
                        'predictor_model_path': 'labeler_model'}.get(field, 'artifact_provenance')
                result_assets.add(mirrored)
                artifact_roles[mirrored].add(role)
                if (ROOT / mirrored).is_file(): add(mirrored, f'Used by selected run ({field})', role)
                else: missing_mirrors[role].add(mirrored)
    for split, views in record.get('prepared_real_paths', {}).items():
        for view, value in views.items():
            mirrored = local_mirror(value)
            if mirrored:
                result_assets.add(mirrored)
                artifact_roles[mirrored].add('prepared_table')
                if (ROOT / mirrored).is_file(): add(mirrored, f'Selected run prepared {split}/{view}', 'prepared_table')
                else: missing_mirrors['prepared_table'].add(mirrored)
    protocol = record.get('evaluation_protocol', {})
    if protocol.get('source_record'):
        origin = local_mirror(protocol['source_record'])
        origin_records.add(origin)
        original_bytes = read_bytes(ROOT / origin)
        original_sha = protocol['source_record_sha256']
        exact = sha(original_bytes) == original_sha
        restored_lf = sha(original_bytes.replace(b'\r\n', b'\n')) == original_sha
        assert exact or restored_lf, origin
        origin_byte_checks.append(dict(path=origin, raw_bytes_match=exact,
                                      recorded_bytes_restorable_as_lf=restored_lf,
                                      original_sha256=original_sha, local_sha256=sha(original_bytes)))
        add(origin, 'Original record cited by a selected Census pilot import', 'import_origin')
    if dataset in ('news', 'california_housing'):
        sidecar_name = name.replace('.run.json', '.d2.json')
        sidecar_bytes = read_bytes(ROOT / sidecar_name)
        sidecar = json.loads(sidecar_bytes)
        assert sidecar['run_record'] == name
        assert sidecar['run_record_sha256'] == sha(frozen_bytes)
        metric_lookup[name, 'd2_absolute_error'] = sidecar['test_scores']['d2_absolute_error']
        add(sidecar_name, 'D2 score/normalization and original-record hash used in current reports', 'd2_sidecar')
    # Keep only small fields for checking actual displayed metrics.
    run_cache[name] = dict(dataset=dataset, seed=seed, mode=mode, scores=record['test_scores'])
assert key_set == expected_keys
assert len(origin_records) == 3

checked_display_values = 0
for mode, rows in comparison_rows.items():
    for row in rows:
        for column, value in row.items():
            if not value or column == 'configuration' or column.endswith(('_record', '_d2_metadata')):
                continue
            match = re.fullmatch(r'(.+)_(f1_binary|f1_macro|accuracy|r2|d2_absolute_error|nmae_sigma)(_percent)?', column)
            if not match:
                continue
            panel, metric, percent = match.groups()
            record_name = row.get(panel + '_record')
            if not record_name:
                continue  # Published paper reference rows are not our model runs.
            expected = metric_lookup[record_name, metric] * (100 if percent else 1)
            assert abs(float(value) - expected) < 1.000001e-6, (column, record_name)
            checked_display_values += 1
assert checked_display_values == 780

scores = read_csv(HERE / 'snapshot/scores.csv')
assert {r['record'] for r in scores} == selected
assert len({(r['record'], r['metric']) for r in scores}) == len(scores)
for row in scores:
    assert abs(float(row['value']) - metric_lookup[row['record'], row['metric']]) < 1e-12
analysis_tables = {}
for arm in ('synthetic', 'mix'):
    for name, count in [('classification_scores.csv', 120), ('generated_baselines.csv', 20), ('regression_scores.csv', 112), ('regression_d2_scores.csv', 56)]:
        rows = read_csv(OUT / arm / name)
        assert len(rows) == count
        for row in rows:
            assert row['record'] in selected
            assert run_cache[row['record']]['mode'] == arm
            assert abs(float(row['value']) - metric_lookup[row['record'], row['metric']]) < 1e-12
        analysis_tables[arm + '/' + name] = count

# Final statistical package membership is explicit, rather than inferred from its folder.
manifest_path = HERE / 'deliverable_manifest.json'
manifest = json.loads(read_bytes(manifest_path))
final_files = {p.replace('\\', '/') for p in manifest['files']}
manifest_hash_failures = []
for name, digest in manifest['files'].items():
    if sha(read_bytes(ROOT / name)) != digest:
        manifest_hash_failures.append(name)
assert not manifest_hash_failures
final_files.add('output/statistical_analysis_328_20261010/real_data_328_analysis_package.zip')
for stem in ('dataset_pipeline_comparison', 'dataset_pipeline_comparison_mix'):
    final_files.update('output/comparisons/' + stem + extension for extension in ('.csv', '.md', '.png'))
for p in (ROOT / 'output/pdf/synthetic_benchmark').rglob('*'):
    if p.is_file(): final_files.add(relative(p))
final_files.add('output/pdf/synthetic_benchmark_source_and_data.zip')

core_files = {
    'scripts/plot_recent_comparison.py', 'scripts/build_corrected_results.py', 'scripts/compute_regression_d2.py',
    'src/modeling/constants.py', 'audit/statistical_328_20261010/build_report.py',
    'audit/statistical_328_20261010/build_explicit_report.py', 'audit/statistical_328_20261010/analyze_explicit.py',
    'audit/statistical_328_20261010/verify_explicit.py', 'audit/statistical_328_20261010/remote_explicit_io.py',
    'audit/statistical_review_20261006/analyze_results.py', 'tmp/pdfs/synthetic_benchmark/build_report.py',
    'audit/sdv_api_smoke.py', 'audit/mnist_head_rerun_20261009/run_plan.py',
    'audit/mnist_head_rerun_20261009/run_worker.py', 'audit/housing_faker_rerun_20261009/worker.py',
    'audit/experiment_plan_20261007/experiment_task.py',
    'tmp/pdfs/synthetic_benchmark/report_template.tex',
    '.cache/remote_intrusion/check_comparison.py', '.cache/remote_intrusion/check_comparison_mix.py',
    '.cache/remote_intrusion/check_comparison_d2.py', '.cache/remote_intrusion/check_latest_comparison_20261010.py'}
for name in core_files:
    assert (ROOT / name).is_file(), name
    add(name, 'Current report/analysis/metric/verification source', 'current_source')

evidence_files = {
    'audit/comparison_rerun_verification_latest.json', 'audit/comparison_news_log_fit_table_hashes.json',
    'audit/comparison_housing_fit_table_hashes.json', 'audit/latest_comparison_snapshot.json',
    'audit/latest_comparison_mix_snapshot.json', 'audit/statistical_328_20261010/snapshot/snapshot.json',
    'audit/statistical_328_20261010/deliverable_manifest.json', 'audit/statistical_328_20261010/validation.json',
    'audit/statistical_328_20261010/original_final_328_project.Rmd',
    'audit/statistical_328_20261010/results/d2_verification.json',
    'audit/remote_simulated_methods_per_run.csv',
    'audit/leakage_mnist_direct_confirmation_2026_09_30.jsonl',
    'audit/mnist_tvae_match_investigation_2026_09_30.jsonl',
    'audit/leakage_affected_runs_2026_09_30.csv'}
for name in evidence_files:
    assert (ROOT / name).is_file(), name
    add(name, 'Current final builder/check or cited reference provenance', 'current_evidence')
for folder in ('audit/statistical_328_20261010/explicit_results',):
    for path in (ROOT / folder).rglob('*'):
        if path.is_file(): add(relative(path), 'Current separate-analysis report input', 'analysis_input')
for name in ('audit/statistical_328_20261010/snapshot/scores.csv',
             'audit/statistical_328_20261010/results/regression_d2_scores.csv'):
    add(name, 'Frozen scores used by current independent analysis', 'analysis_input')
for name in ('per_run.csv', 'summary.csv', 'config.json'):
    add('output/simulated_all_20260930/' + name, 'Current standalone simulated report input', 'simulated_report_input')

# Follow explanatory links from current final Markdown, not all historical notes.
for name in sorted(final_files):
    path = ROOT / name
    if path.suffix.lower() != '.md': continue
    text = path.read_text(encoding='utf-8-sig')
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in link or link.startswith('#'): continue
        linked = (path.parent / link.split('#')[0]).resolve()
        if linked.is_file() and linked.is_relative_to(ROOT):
            add(relative(linked), f'Explanation/asset linked from {name}', 'final_linked_explanation')

EXCLUDED_TOP = {'.git', '.worktrees', 'summer_research', 'env', 'venv', 'node_modules', 'GLRM', 'pyglrm', '.agents', '.codex', '.aws'}
files = {}
for base, directories, names in os.walk(ROOT, followlinks=False):
    parent = Path(base)
    directories[:] = [d for d in directories if d not in EXCLUDED_TOP and not (parent / d).is_symlink()]
    for name in names:
        p = parent / name
        if p.is_symlink(): continue
        rel = relative(p)
        if rel.startswith('audit/local_final_results_') or name == 'LOCAL_FINAL_RESULTS_AUDIT_2026_10_10.md':
            continue
        files[rel] = p.stat().st_size

canonical_run_files = {p for p in files if p.startswith('output/') and p.endswith('.run.json')}
unselected = canonical_run_files - selected

def classify(path):
    if path == '.cache/remote_intrusion/server_key' or path in ('audit/ssh_known_hosts', 'AGENTS.md'):
        return 'LOCAL_OPERATIONAL_KEEP', 'Local access/instructions; outside result publication'
    if path in final_files:
        return 'FINAL_DELIVERABLE_OR_PACKAGE_MEMBER', 'Explicit current output/package membership'
    if path in selected:
        return 'FINAL_SOURCE_RECORD', 'One of the exact 406 selected runs'
    if path.startswith('audit/statistical_328_20261010/snapshot/records/'):
        name = path.removeprefix('audit/statistical_328_20261010/snapshot/records/')
        if name in selected:
            return 'FINAL_RECORD_COPY_CONSOLIDATE', 'Same experiment as canonical output; preserve snapshot hash bytes once'
        if name.endswith('.d2.json') and name.replace('.d2.json', '.run.json') in selected:
            return 'FINAL_SIDECAR_COPY_CONSOLIDATE', 'Frozen copy of current derived-metric evidence'
    if path in origin_records:
        return 'FINAL_IMPORT_ORIGIN', 'One of three source records cited by selected Census runs'
    if path in file_reason:
        roles = artifact_roles[path]
        if roles & {'selected_checkpoint', 'generator_model', 'labeler_model', 'synthetic_table', 'prepared_table'}:
            return 'SELECTED_EXPERIMENT_ASSET', 'Supports a selected experiment; distinguish report rebuild from retraining'
        return 'FINAL_REBUILD_OR_EVIDENCE', '; '.join(sorted(file_reason[path]))
    if path in ('.cache/remote_intrusion/latest_comparison.zip', '.cache/remote_intrusion/latest_comparison_mix.zip'):
        return 'CURRENT_VERIFIER_TRANSPORT_COPY', 'Current checker uses export ZIP; consolidate after archive-based checks are replaced'
    if path == 'audit/statistical_328_20261010/remote_snapshot.zip':
        return 'FROZEN_SNAPSHOT_TRANSPORT_COPY', 'Imported source archive; extracted selected records/scores already retained'
    if path.startswith(('output/statistical_analysis_328_20261010/', 'audit/statistical_328_20261010/results/')):
        return 'SUPERSEDED_ANALYSIS_NOT_FINAL', 'Excluded from current explicit synthetic/mix package and numerical fits'
    if path in unselected:
        return 'UNSELECTED_EXPERIMENT_RECORD_NOT_FINAL', 'Not in final 406 grid; preserve scientific history separately as appropriate'
    if path.startswith('data/'):
        if path.split('/')[1] in ('corrected_v2', 'corrected_v2_seed42_mnist28_news', 'pilot_ctgan_v1'):
            return 'UNSELECTED_PREPARED_EVIDENCE_NOT_FINAL', 'Not used by selected final result sources/builders'
        return 'LEGACY_DATA_NOT_FINAL', 'Final builders use corrected records/frozen scores; current acquisition does not use these legacy splits'
    if path.startswith('sdv trained model/'):
        return 'UNSELECTED_OR_LEGACY_MODEL_NOT_FINAL', 'Not among selected record model references locally'
    if any(part in ('__pycache__', '.ipynb_checkpoints', '.pytest_cache') for part in Path(path).parts):
        return 'RECREATABLE_INTERMEDIATE', 'Generated runtime/notebook/test cache'
    if path.startswith('audit/statistical_328_20261010/'):
        return 'EARLIER_ANALYSIS_OR_COLLECTION_NOT_FINAL', 'Earlier analysis/presentation/transfer tool outside active selected workflow'
    if path.startswith(('audit/', 'SDGym-research/', '.cache/', 'output/', 'plots/')):
        return 'HISTORICAL_OR_DIAGNOSTIC_NOT_FINAL', 'Outside inspected current final output/input/source sets; archive by study if useful'
    if path.startswith('tmp/'):
        return 'RECREATABLE_BUILD_INTERMEDIATE', 'Current build sources retained explicitly; this generated file is not a final input'
    if path.startswith(('src/', 'scripts/', 'simulated_eval/')) or path in ('requirements.txt', '.gitignore', '.gitattributes', '.gitleaks.toml'):
        return 'MAINTAINED_RESEARCH_CODE_OR_CONFIG', 'Useful implementation/configuration; not a dependency of every final figure'
    if path.startswith(('papers/', 'SDGym/')):
        return 'REFERENCE_OR_LEGACY_BENCHMARK_NOT_FINAL', 'Reference library/legacy benchmark outside current report builds'
    return 'ROOT_LEGACY_OR_LOCAL_TOOL_NOT_FINAL', 'Not read by current final result/report workflow'

catalog = []
for path, size in sorted(files.items()):
    category, why = classify(path)
    catalog.append(dict(path=path, bytes=size, disposition=category, reason=why,
                        final_roles=';'.join(sorted(artifact_roles[path]))))

for suffix, rows in [('selected_runs_2026_10_10.csv', run_rows), ('file_dispositions_2026_10_10.csv', catalog)]:
    with (AUDIT / ('local_final_results_' + suffix)).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

counts = Counter(row['disposition'] for row in catalog)
sizes = Counter()
for row in catalog: sizes[row['disposition']] += row['bytes']
legacy_data_bytes = sum(size for path, size in files.items() if path.startswith('data/') and classify(path)[0] == 'LEGACY_DATA_NOT_FINAL')
legacy_models_bytes = sum(size for path, size in files.items() if path.startswith('sdv trained model/') and classify(path)[0] == 'UNSELECTED_OR_LEGACY_MODEL_NOT_FINAL')
summary = dict(status='LOCAL_READ_ONLY_AUDIT_NO_ASSET_MOVES', checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
               roots=['current real comparison', 'current separate real statistical analysis', 'current standalone simulated report'],
               remote_contacted=False, datasets=list(SCOPES), generators=['ctgan', 'tvae'], labelers=['rf', 'xgb', 'dnn'],
               synthetic_conditions_per_generator=7, training_modes=['synthetic', 'mix'], baseline_per_dataset_seed=1,
               formula='7 datasets * 2 seeds * (2 generators * (1 generated + 2 has_y * 3 labelers) * 2 training modes + 1 original) = 406',
               selected_runs=len(selected), selected_classification_runs=290, selected_regression_runs=116,
               per_dataset={d:dict(per_dataset[d], total=sum(per_dataset[d].values())) for d in SCOPES},
               per_mode=dict(Counter(r['training_mode'] for r in run_rows)),
               generated_target_runs=sum(r['labeler']=='generated' for r in run_rows), relabeled_runs=sum(r['labeler'] in ('rf','xgb','dnn') for r in run_rows),
               has_y_counts=dict(Counter(str(r['has_y']) for r in run_rows)),
               comparison_records={k:len(v) for k,v in comparison_sets.items()}, shared_real_baselines=len(shared),
               no_missing_or_extra_configurations=True, local_selected_record_files=len(run_rows),
               frozen_record_hashes_verified=len(run_rows), canonical_frozen_byte_identical=raw_equal,
               canonical_frozen_newline_only_difference=normalized_equal,
               selected_record_total_bytes=sum(files[p] for p in selected),
               selected_frozen_record_copy_total_bytes=sum(files['audit/statistical_328_20261010/snapshot/records/'+p] for p in selected),
               comparison_metric_cells_checked=checked_display_values, frozen_score_rows_checked=len(scores),
               analysis_score_tables=analysis_tables, manifest_files_hashed=len(manifest['files']),
               current_census_import_source_hashes_checked=len(origin_records),
               census_import_source_byte_checks=origin_byte_checks,
               local_referenced_asset_counts=dict(Counter(role for p in result_assets if (ROOT/p).is_file() for role in artifact_roles[p])),
               not_local_referenced_asset_counts={k:len(v) for k,v in missing_mirrors.items()},
               inventoried_local_files=len(files), disposition_counts=dict(counts), disposition_bytes=dict(sizes),
               legacy_data_not_final_bytes=legacy_data_bytes, legacy_models_not_final_bytes=legacy_models_bytes,
               existing_canonical_run_records=len(canonical_run_files), unselected_canonical_run_records=len(unselected),
               excluded_roots=sorted(EXCLUDED_TOP),
               limitations=['Not-final does not mean scientifically worthless or approved for deletion.',
                            'Classification is relative to the three declared final output families, not arbitrary external consumers.',
                            'Dataset tables/model weights were not loaded, diagnostics were not run, and models were not executed.',
                            'Selected record metadata contains nonlocal artifact paths; absent local copies do not invalidate saved result tables.',
                            'Generic historical files were classified by final-set exclusion; individual archive retention can be smaller than whole bundles.'])
(AUDIT / 'local_final_results_validation_2026_10_10.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary,indent=2))
