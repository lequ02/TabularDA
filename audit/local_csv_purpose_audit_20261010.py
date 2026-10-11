"""Inspect local CSV artifacts only. No experiments, moves, or deletions."""
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit'
PREFIX = 'local_csv_purpose_'
SKIP = {'.git', '.worktrees', 'node_modules', '.agents', '.codex', '.aws'}
ENV = {'summer_research', 'env', 'venv', '.venv', 'site-packages'}
SCOPE_COLUMNS = {'dataset', 'seed', 'experiment_seed', 'method', 'generator',
                 'labeler', 'training_mode', 'mode', 'metric', 'source',
                 'status', 'split', 'condition', 'construction', 'benchmark',
                 'dropout', 'has_y', 'config', 'pipeline', 'namespace',
                 'train_option', 'augment_option', 'configuration', 'domain',
                 'family', 'table', 'transform', 'task', 'arm'}


def inspect_all():
    items = []
    excluded_dirs = []
    for base, dirs, files in os.walk(ROOT):
        for d in dirs:
            if d in SKIP or (Path(base) / d).is_symlink():
                excluded_dirs.append((Path(base) / d).relative_to(ROOT).as_posix())
        dirs[:] = [d for d in dirs if d not in SKIP and not (Path(base) / d).is_symlink()]
        for name in files:
            if not name.lower().endswith('.csv') or name.startswith(PREFIX):
                continue
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            before = path.stat()
            environment = bool(set(path.relative_to(ROOT).parts) & ENV)
            # Full small-file inspection, bounded structural inspection of large
            # data. Never compute distributions, replay models, or load training.
            full = before.st_size <= 2_000_000
            with path.open('rb') as f:
                payload = f.read() if full else f.read(131072)
            if not full:
                # Drop an incomplete last line, preserving complete CSV records.
                payload = payload[:payload.rfind(b'\n') + 1]
            content = payload.decode('utf-8-sig')
            reader = csv.reader(io.StringIO(content, newline=''))
            header = next(reader, [])
            scope = defaultdict(set)
            rows = 0
            widths = Counter()
            sample = []
            for row in reader:
                if not row:
                    continue
                rows += 1
                widths[len(row)] += 1
                if len(sample) < 2:
                    sample.append(row)
                for i, key in enumerate(header):
                    if key.lower() in SCOPE_COLUMNS and i < len(row):
                        scope[key].add(row[i])
                if not full and rows >= 10:
                    break
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), rel
            items.append({
                'path': rel, 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
                'environment_dependency': environment,
                'inspection': 'complete CSV parse' if full else 'header and up to ten records',
                'inspected_rows': rows, 'total_rows': rows if full else None,
                'header': header, 'row_widths': dict(widths),
                'sample': sample, 'scope': {k: sorted(v) for k, v in scope.items()},
                'sha256': hashlib.sha256(payload).hexdigest() if full else None,
            })
    result = {'inspected_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
              'remote_contacted': False, 'excluded_directories': excluded_dirs,
              'files': sorted(items, key=lambda x: x['path'])}
    (AUDIT / 'local_csv_purpose_inspection_2026_10_10.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'csv_files': len(items),
                      'environment_csv_files': sum(x['environment_dependency'] for x in items),
                      'full_parse': sum(x['total_rows'] is not None for x in items),
                      'header_sample': sum(x['total_rows'] is None for x in items)}))


def source_index(items):
    """Literal references are evidence of a reader/writer, not retention rules."""
    names = {Path(x['path']).name for x in items if not x['environment_dependency']}
    matcher = re.compile('|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
    index = defaultdict(list)
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP | ENV and not (Path(base) / d).is_symlink()]
        for name in files:
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() not in {'.py', '.md', '.r', '.rmd', '.ipynb'}:
                continue
            if name.startswith(('local_csv_purpose_', 'local_final_results_')) or name in {
                'LOCAL_CSV_PURPOSE_REVIEW_2026_10_10.md', 'LOCAL_FINAL_RESULTS_AUDIT_2026_10_10.md',
                'REPOSITORY_ORGANIZATION_REVIEW_2026_10_10.md', 'EXPERIMENT_CLEANUP_PLAN_2026_10_10.md'}:
                continue
            if path.stat().st_size > 3_000_000:
                continue
            content = path.read_text(encoding='utf-8-sig')
            if path.suffix == '.ipynb' and content.strip():
                notebook = json.loads(content)
                content = '\n'.join(''.join(c.get('source', [])) for c in notebook.get('cells', []))
            for line_number, line in enumerate(content.splitlines(), 1):
                for matched in set(matcher.findall(line)):
                    index[matched].append({'source': rel, 'line': line_number,
                                          'text': line.replace('\\\\', '/').replace('\\', '/')})
    return index


STAT_PURPOSE = {
    'classification_scores': 'Per-seed macro F1 observations for the explicit generator/has_y/labeler/seed model',
    'generated_baselines': 'Generated-target comparators, kept outside the relabeled factorial fit',
    'original_references': 'Real-only baseline metrics shared by the two training-mode analyses',
    'regression_scores': 'Saved regression R2 and test-normalized NMAE values with record hashes',
    'regression_d2_scores': 'Saved absolute-error D2 values with run and sidecar hashes',
    'regression_summary': 'Descriptive seed/method averages in their original metric units',
    'factorial_anova': 'Primary explicit-factor Type II ANOVA terms and nominal significance',
    'seed_block_anova': 'Supporting model with seed as an additive block',
    'dataset_interaction_anova': 'Supporting model checking dataset-dependent treatment effects',
    'joint_tests': 'Nested-model tests of seed and dataset interactions',
    'model_summary': 'Formula, observation count, rank, residual degrees of freedom and diagnostics',
    'independent_F_checks': 'Independent restricted-model verification of ANOVA F values',
    'estimated_means': 'Model estimated marginal means and confidence intervals for treatment cells',
    'factorial_effects': 'Model-based factor contrasts, intervals and multiplicity-adjusted p values',
    'planned_contrasts': 'Prespecified matched comparisons with inference over five dataset means',
    'labeler_effects': 'Generator/labeler/has_y comparisons against matched generated-target baselines',
    'dataset_effects': 'Dataset-specific matched treatment differences',
    'contrast_weights': 'Explicit contrast coefficients showing how treatment cells are averaged',
    'matched_relabeling_scores': 'Per-run relabeled scores joined to their exact baseline identities',
    'scores': 'Frozen saved metrics across the 406 distinct selected run records',
    'per_run': 'One aggregate row per selected downstream run',
    'summary': 'Grouped aggregate metrics derived from selected run records',
}


def classify(item, final_paths, log_origins):
    p = item['path']; name = Path(p).name; stem = Path(p).stem
    h = set(item['header']); scope = item['scope']
    if item['environment_dependency']:
        return ('ENVIRONMENT_DEPENDENCY', 'Installed-package data or test fixture',
                'Belongs to the Python environment rather than the research artifact collection; leave the installed package intact.',
                '', '', 'leave environment intact')
    if p in final_paths:
        if p.startswith('output/comparisons/'):
            purpose = 'Current real-data comparison table; 18 configuration/reference rows point to 210 records per mode'
            evidence = 'scripts/plot_recent_comparison.py; audit/COMPARISON_HANDOFF.md'
            study = 'real_comparison'
        elif p.startswith(('output/pdf/synthetic_benchmark/', 'tmp/pdfs/synthetic_benchmark/')):
            purpose = 'Current simulated report publication table: ' + stem.replace('_', ' ')
            evidence = 'tmp/pdfs/synthetic_benchmark/build_report.py'
            study = 'simulated_benchmark'
        elif p == 'audit/remote_simulated_methods_per_run.csv':
            purpose = 'Historical 406-row simulated results; current report uses 42 paper-baseline rows, while other rows retain the pre-fix extension evidence'
            evidence = 'tmp/pdfs/synthetic_benchmark/build_report.py; audit/simulated_benchmark_audit.md'
            study = 'simulated_paper_replication'
        elif p.startswith('output/simulated_all_20260930/'):
            purpose = 'Current standalone simulated production ' + ('378 per-run observations' if stem == 'per_run' else '189 seed-averaged method/dataset observations')
            evidence = 'simulated_eval/README.md; tmp/pdfs/synthetic_benchmark/build_report.py'
            study = 'simulated_benchmark'
        else:
            purpose = STAT_PURPOSE.get(stem, 'Current statistical-report table: ' + stem.replace('_', ' '))
            evidence = 'audit/statistical_328_20261010/EXPLICIT_ANALYSIS_HANDOFF.md; audit/statistical_328_20261010/analyze_explicit.py; audit/statistical_328_20261010/build_explicit_report.py'
            study = 'real_statistics'
        return ('CURRENT_RESULT_OR_REBUILD', purpose,
                'Supports a current deliverable or its numerical/provenance verification; preserve a canonical copy and its documented recipe.',
                study, evidence, 'keep canonical table')
    special = {
        'audit/completed_run_preprocessing_findings_2026_10_09.csv': (
            'preprocessing_defects', '871-row artifact-level defect inventory for MNIST heads, binary-indicator loss, numerical PCA and other checked mechanisms',
            'Explains why MNIST downstream results were replaced and why some NB/PCA arms are not evidence for the current comparison; defect flags can overlap.',
            'audit/COMPLETED_RUN_PREPROCESSING_AUDIT_2026_10_09.md; audit/mnist_head_rerun_20261009/README.md'),
        'audit/leakage_affected_runs_2026_09_30.csv': (
            'mnist_feature_collisions', 'Run-specific counts of generated feature matches with held-out MNIST12 rows',
            'Supports the surviving collision caveat. Matching generated features does not by itself prove a generator fitted held-out rows.',
            'audit/LEAKAGE_REVIEW_2026_09_30.md; scripts/confirm_mnist_leakage_overlap.py'),
        'audit/pca_results_requiring_repair_2026_10_01.csv': (
            'pca_column_repair', 'Twelve historical Adult/Covertype downstream records identified for numerical-only PCA repair',
            'Records which observations the PCA column-selection defect invalidated; its queued status is historical, not current completion evidence.',
            'audit/EXPERIMENT_REPAIRS_2026_10_01.md'),
        'output/comparisons/census_kdd_training_histories_2026_10_03/histories.csv': (
            'census_unweighted_collapse', '369 epoch rows for eleven selected Census configurations, including train/dev loss and binary F1',
            'Demonstrates the difference between an early loss-selected all-negative checkpoint and later positive predictions; motivates the weighted/macro-F1 protocol.',
            'audit/CREDIT_CENSUS_DIAGNOSIS_2026_10_03.md; audit/CENSUS_DNN_OUTPUT_LAYER_DIAGNOSIS_2026_10_04.md'),
        'output/comparisons/census_kdd_training_histories_2026_10_03/checkpoint_summary.csv': (
            'census_unweighted_collapse', 'Eleven selected-versus-final Census checkpoint summaries',
            'Compact evidence that development-loss checkpoint selection and final-epoch behavior differed; preserve with epoch histories and original record hashes.',
            'audit/CREDIT_CENSUS_DIAGNOSIS_2026_10_03.md; audit/CENSUS_WEIGHTED_PILOT_2026_10_04.md'),
        'output/comparisons/simulated_methods/dataset_likelihood_comparison.csv': (
            'simulated_paper_discrepancy', '42 dataset/method/likelihood comparisons between published supplementary numbers and the historical replication',
            'Supports the paper-versus-supplement discrepancy analysis and evaluator limitations; cannot replace the standalone corrected benchmark.',
            'audit/simulated_benchmark_audit.md; scripts/plot_simulated_method_comparison.py'),
        'output/comparisons/simulated_methods/all_methods_comparison.csv': (
            'simulated_paper_discrepancy', '58 family/method aggregates for the historical paper replication and labeling extension',
            'Retains prior findings and the separation of paper baselines from pre-fix relabeling; disclose absent sample provenance and historical defects.',
            'audit/simulated_benchmark_audit.md; scripts/plot_simulated_method_comparison.py'),
    }
    if p in special:
        study, purpose, reason, evidence = special[p]
        return ('DECISION_OR_VALIDITY_EVIDENCE', purpose, reason, study, evidence, 'keep decision evidence')
    if p.startswith('output/news_log_pilot_20261008_0048/'):
        purpose = {'pilot_comparison': 'Three-arm seed-43 log-target pilot outcomes and selected development checkpoints',
                   'per_run': 'Three per-run pilot summaries with metric values and source-record identities',
                   'summary': 'Three pilot arm aggregates; each has only one seed'}[stem]
        return ('DECISION_OR_VALIDITY_EVIDENCE', purpose,
                'Explains the News log/no-BatchNorm rerun: MAE improved but R2 did not uniformly improve. The three pilots were explicitly excluded from final rerun reuse.',
                'news_log_pilot', 'audit/NEWS_LOG_PILOT_2026_10_08.md; audit/NEWS_LOG_RERUN_2026_10_08.md; audit/news_log_pilot_results_2026_10_08.json', 'keep decision evidence')
    if p == 'April29.csv' or p.startswith('audit/reconstruction/'):
        hypothetical = 'hypothetical' in stem or 'substitution' in stem or 'substitution' in ' '.join(item['header']).lower()
        raw = '/inputs/' in p or p == 'April29.csv' or stem in {'Mar23_tidy','April02_tidy','April29_tidy','report_matches','reconstructed_final_results','deduplicated_method_scores','deduplicated_dataset_differences'}
        if hypothetical:
            return ('HISTORICAL_SENSITIVITY_SCENARIO', 'Hypothetical replacement/sensitivity calculation: ' + stem.replace('_',' '),
                    'Explains historical analytical alternatives; substituted paper/reference values are scenarios, not newly observed experiments. Keep a labeled reproducible scenario archive.',
                    'historical_reconstruction', 'audit/reconstruction/RECONSTRUCTION.md; audit/reconstruction/paper_comparison_update_2026_09_18/outputs/paper_comparison/PAPER_COMPARISON_AND_SENSITIVITY.md', 'archive labeled scenario and recipe')
        if raw:
            return ('DECISION_OR_VALIDITY_EVIDENCE', 'Historical aggregation/source reconciliation: ' + stem.replace('_',' '),
                    'Documents why the 2025 maximum-test-score, mixed-condition and Adult/Census-alias results were replaced. Preserve source observations and reconciliations, with their validity limits.',
                    'historical_reconstruction', 'audit/reconstruction/RECONSTRUCTION.md; audit/reconstruction/reconstruct_analysis.py', 'keep compact historical evidence')
        readme = next((q / 'README.md' for q in (ROOT / p).parents
                       if ROOT in q.parents and (q / 'README.md').is_file()),
                      ROOT / 'audit/reconstruction/RECONSTRUCTION.md')
        return ('HISTORICAL_ANALYSIS_EXPORT', 'Historical analysis result: ' + stem.replace('_',' '),
                'Supports a prior reconstruction, figure or statistical alternative, not the corrected main grid. Preserve meaningful releases and their source/recipe; consolidate repeated exports.',
                'historical_reconstruction', 'audit/reconstruction/RECONSTRUCTION.md; ' + readme.relative_to(ROOT).as_posix(), 'archive meaningful release')
    if p.startswith('audit/results_review/'):
        if '/logs/' in p or '/local_logs/' in p:
            origin = log_origins[p]
            return ('DECISION_OR_VALIDITY_EVIDENCE', 'Original historical epoch log: ' + origin,
                    'Primary evidence for wrong aggregation, test-maximum selection, split-version distinctions and retrospective development selection. Keep unique original bytes plus the filename/source manifest.',
                    'historical_result_credibility', 'audit/results_review/RESULTS_REVIEW.md; audit/results_review/local_log_manifest.json; audit/results_review/drive/log_manifest.json', 'keep unique original historical log')
        return ('DECISION_OR_VALIDITY_EVIDENCE', 'Historical workbook-equivalent result export: ' + stem.replace('_',' '),
                'Documents the flawed reported summary and its exact relation to original logs; useful for explaining the corrected protocol, not a current score source.',
                'historical_result_credibility', 'audit/results_review/RESULTS_REVIEW.md; audit/results_review/workbook_comparison.py', 'keep compact historical evidence')
    if p.startswith(('audit/statistical_review_20261006/', 'output/statistical_review_20261006/')):
        reason = 'Earlier analysis of incomplete real-data scope and/or simulated results, superseded for current real-data inference; retain a labeled release and recipe to document analysis choices.'
        if 'credit' in stem:
            reason = 'Credit-inclusion sensitivity evidence for scope choice; the small minority holdout, not unfavorable scores, explains exclusion. Historical inputs differ from the current corrected final scope.'
        return ('HISTORICAL_ANALYSIS_EXPORT', 'Earlier statistical ' + stem.replace('_',' '), reason,
                'statistics_20261006', 'audit/statistical_review_20261006/analyze_results.py; audit/RESEARCH_DIRECTION_2026_10_06.md', 'archive meaningful release')
    if p.startswith(('audit/statistical_328_20261010/', '.cache/statistical_328_before_')):
        if p in {'audit/statistical_328_20261010/separate_results/mix/planned_contrasts.csv',
                 'audit/statistical_328_20261010/separate_results/synthetic/planned_contrasts.csv'}:
            return ('DECISION_OR_VALIDITY_EVIDENCE', 'Previous-version planned contrasts used to verify unchanged effects in the explicit-factor revision',
                    'The current verify_explicit.py compares these four contrast effects with the new analysis. Preserve this small cross-version reference; the entire prior report dependency tree is unnecessary.',
                    'statistics_prior_versions', 'audit/statistical_328_20261010/verify_explicit.py:80; audit/statistical_328_20261010/EXPLICIT_ANALYSIS_HANDOFF.md', 'keep cross-version verification reference')
        return ('HISTORICAL_ANALYSIS_EXPORT', 'Superseded pooled/composite-factor statistics or preserved report inputs: ' + stem.replace('_',' '),
                'Explains the prior analysis version. The explicit-factor handoff supersedes these fits; keep one versioned release rather than numerous active duplicate tables.',
                'statistics_prior_versions', 'audit/statistical_328_20261010/EXPLICIT_ANALYSIS_HANDOFF.md; audit/statistical_328_20261010/SEPARATE_ANALYSIS_HANDOFF.md', 'archive meaningful release')
    if p.startswith('.cache/remote_intrusion/'):
        return ('HISTORICAL_REPORT_SNAPSHOT', 'Earlier real-comparison table snapshot: ' + stem.replace('_',' '),
                'Preserves an earlier displayed state, including results later corrected or scope/metric presentation changes. A dated version archive suffices; a current writer dependency does not make every backup permanent.',
                'comparison_prior_versions', 'audit/COMPARISON_HANDOFF.md', 'archive meaningful release')
    if p == 'output/comparisons/adult_mnist28_comparison.csv':
        return ('HISTORICAL_REPORT_SNAPSHOT', 'Fourteen-row early pilot/corrected comparison with heterogeneous coverage',
                'Retains the pilot-to-corrected reporting history; not the current 406 grid. Pilot differences combine protocols and incomplete scope, so do not interpret as a controlled ablation.',
                'comparison_prior_versions', 'scripts/plot_recent_comparison.py; audit/PILOT_RUN_SPEC.md', 'archive meaningful release')
    if p.startswith('output/new_simulated_eval/'):
        return ('SEPARATE_EXPLORATORY_STUDY', 'Earlier single-process simulated ' + stem.replace('_',' '),
                'Contains 54 observations across seeds 42/43, under a separate design. Preserve its configuration and results as exploratory evidence; do not add them as independent repeats of the seven-dataset benchmark.',
                'new_simulated_eval', 'audit/experiment_plan_20261007/build_plan.py; output/new_simulated_eval/config.json', 'archive separate study')
    smoke_prefixes = ('output/new_simulated_smoke/', 'output/simulated_all_smoke/',
                     'output/simulated_all_verified/', 'output/simulated_insurance_smoke/',
                     'audit/simulated_fixed_smoke/', 'SDGym-research/data/simulated_paper_smoke/',
                     'SDGym-research/data/simulated_smoke_wrapper/')
    if p.startswith(smoke_prefixes):
        table = bool({'feature_0','asia','HISTORY','BirthAsphyxia','GoodStudent'} & h) or '/seed_' in p
        if table:
            return ('VALIDATION_SAMPLE', 'Small validation table with ' + str(len(item['header'])) + ' columns: ' + stem,
                    'Exercises generation, labeling, encodings or density scoring in a smoke protocol; not production quality or a rejected-method utility result. Preserve configuration/check evidence; generated rows may be retired if recreation is documented.',
                    p.split('/')[1] if p.startswith('output/') else 'legacy_simulation_smoke',
                    'simulated_eval/README.md; audit/simulated_benchmark_audit.md; audit/experiment_plan_20261007/build_plan.py', 'retain validation summary; review generated sample retention')
        return ('VALIDATION_SUMMARY', 'Smoke-run execution/score summary: ' + stem.replace('_',' '),
                'Small evidence that the checked implementation executed, including incomplete or repaired coverage. Not evidence of production method quality; retain once with its configuration and checks.',
                p.split('/')[1] if p.startswith('output/') else 'legacy_simulation_smoke',
                'simulated_eval/README.md; audit/simulated_benchmark_audit.md; audit/experiment_plan_20261007/build_plan.py', 'keep compact validation evidence')
    if p.startswith('data/') or p == 'src/synthesize_data/draft.csv':
        dataset = p.split('/')[1] if p.startswith('data/') else 'adult'
        if dataset == 'census': dataset = 'adult_legacy_census_alias'
        if stem == 'creditcard':
            purpose = 'Legacy raw Credit source table, including Time and 28 V features plus Amount/Class'
        elif p == 'src/synthesize_data/draft.csv':
            purpose = 'Unnamespaced Adult one-hot training/synthesis draft; observed schema includes income target'
        elif '_sdv_' in stem.lower():
            gen = 'TVAE' if 'tvae' in stem.lower() else 'CTGAN'
            label = next((v for v in ('categorical','gaussian','pca_gmm','rf','xgb') if v in stem.lower()), 'generated target')
            purpose = 'Legacy ' + dataset + ' ' + gen + ' table, filename labeler ' + label + '; observed target/schema recorded separately'
        else:
            purpose = 'Legacy ' + dataset + (' one-hot' if 'onehot' in stem else ' raw') + (' training' if 'train' in stem else ' test' if 'test' in stem else '') + ' table'
        return ('LEGACY_DATA_PROVENANCE_REVIEW', purpose,
                'No selected corrected record refers to this local table. It may reproduce a historical run or support an old split/encoding diagnosis; inspect its old record binding before deciding archive retention. Header/sample inspection alone cannot establish safe deletion or exact duplication.',
                'legacy_' + dataset, 'src/modeling/constants.py; src/commons/read_train_test_csv.py; audit/results_review/RESULTS_REVIEW.md; audit/reconstruction/RECONSTRUCTION.md', 'preserve pending historical provenance decision')
    if p.startswith('SDGym/datasets/'):
        return ('LEGACY_BENCHMARK_DATA', 'Packaged legacy SDGym dataset ' + p.split('/')[2],
                'Legacy benchmark dataset export; no selected corrected record or current report consumes this local CSV. Preserve it as a separately labeled data/reference archive if that legacy study is retained.',
                'legacy_sdgym', 'audit/simulated_benchmark_audit.md; src/datasets.py; simulated_eval/README.md', 'archive with legacy benchmark')
    if p.startswith('SDGym-research/data/'):
        return ('LEGACY_BENCHMARK_DATA', 'Earlier simulated source or relabeled table: ' + stem + '; observed columns ' + ', '.join(item['header'][:5]),
                'Supports the older deterministic-target/likelihood implementation and exploratory helpers. It is scientifically distinct from the standalone noisy-target benchmark; preserve meaningful source/control data with its recipe, not as extra production repeats.',
                'legacy_simulation', 'src/synthesize_data/main.py; src/synthesize_data/test_syn.py; SDGym-research/synthetic_data_benchmark/evaluator/evaluate.py; simulated_eval/README.md', 'archive separate study and recipe')
    if p.startswith('audit/'):
        if p in {'audit/missing_experiments_2026_10_01.csv', 'audit/stash_edit_inventory_2026_10_01.csv'}:
            return ('HISTORICAL_STATUS_OR_PROVENANCE', 'Historical ' + stem.replace('_',' '),
                    'Records a dated incomplete queue or preserved source edits; useful for operational/provenance history, not proof of present completion or a method outcome.',
                    'operations_history', 'audit/EXPERIMENT_REPAIRS_2026_10_01.md; audit/RESEARCH_DIRECTION_2026_10_06.md', 'archive concise operational evidence')
        if any(v in stem for v in ('cleanup','repository_organization','local_final_results')):
            return ('CLEANUP_INVESTIGATION_INTERMEDIATE', 'Earlier cleanup/organization catalog: ' + stem.replace('_',' '),
                    'Investigation output, not experiment evidence. Some earlier classifications were superseded; preserve the final migration/name map and retire redundant working catalogs after consolidation.',
                    'repository_organization', '', 'replace with final catalog after migration')
    raise ValueError('CSV purpose has not been investigated: ' + p)


def catalog():
    inspection = json.loads((AUDIT / 'local_csv_purpose_inspection_2026_10_10.json').read_text(encoding='utf-8'))
    items = inspection['files']
    with (AUDIT / 'local_final_results_file_dispositions_2026_10_10.csv').open(encoding='utf-8-sig', newline='') as f:
        final_paths = {x['path'] for x in csv.DictReader(f) if x['disposition'] in {
            'FINAL_DELIVERABLE_OR_PACKAGE_MEMBER', 'FINAL_REBUILD_OR_EVIDENCE'}}
    # Active tables are established by inspected builders/manifests, not merely
    # the older catalog, which predates the explicit-factor report revision.
    final_paths.update(x['path'] for x in items if x['path'].startswith((
        'audit/statistical_328_20261010/explicit_results/', 'output/statistical_analysis_328_20261010/')))
    final_paths.update({'audit/statistical_328_20261010/snapshot/scores.csv',
                        'audit/statistical_328_20261010/snapshot/aggregate/per_run.csv',
                        'audit/statistical_328_20261010/snapshot/aggregate/summary.csv',
                        'audit/statistical_328_20261010/results/regression_d2_scores.csv',
                        'output/simulated_all_20260930/per_run.csv', 'output/simulated_all_20260930/summary.csv',
                        'audit/remote_simulated_methods_per_run.csv'})
    # A direct evidence CSV gets its explicit scientific role below even when
    # the former final-only catalog already marked it as final evidence.
    final_paths.difference_update({'audit/leakage_affected_runs_2026_09_30.csv',
                                  'audit/completed_run_preprocessing_findings_2026_10_09.csv',
                                  'audit/pca_results_requiring_repair_2026_10_01.csv'})
    origins = {}
    for filename in ('results_review/local_log_manifest.json', 'results_review/drive/log_manifest.json'):
        entries = json.loads((AUDIT / filename).read_text(encoding='utf-8-sig'))
        for x in entries:
            local = x['local'].replace('\\','/')
            rel = local.split('/audit/',1)[1]
            origins['audit/' + rel] = x.get('member', x.get('name'))
    index = source_index(items)
    basename_counts = Counter(Path(x['path']).name for x in items if not x['environment_dependency'])
    rows = []
    for x in items:
        role, purpose, reason, study, evidence, action = classify(x, final_paths, origins)
        exact = []
        name = Path(x['path']).name
        for ref in index.get(name, []):
            if x['path'] in ref['text'] or basename_counts[name] == 1:
                exact.append(ref['source'] + ':' + str(ref['line']))
        problem = 'data row width differs from header; preserve original and use documented historical parser' if any(int(w) != len(x['header']) for w in x['row_widths']) else ''
        rows.append({'path': x['path'], 'bytes': x['bytes'], 'study': study,
                     'scientific_role': role, 'purpose': purpose, 'why_it_matters_or_does_not': reason,
                     'proposed_treatment': action, 'proposed_path': '',
                     'inspection': x['inspection'], 'inspected_rows': x['inspected_rows'],
                     'total_rows': x['total_rows'], 'columns': len(x['header']),
                     'header_json': json.dumps(x['header'][:30] if x['environment_dependency'] else x['header'],ensure_ascii=False),
                     'observed_scope_json': json.dumps(x['scope'],ensure_ascii=False),
                     'structural_caveat': problem, 'sha256': x['sha256'],
                     'byte_identical_canonical': '', 'scientific_evidence': evidence,
                     'literal_source_references': '; '.join(sorted(set(exact))[:12]),
                     'same_basename_references_count': len(index.get(name, []))})
    by_hash = defaultdict(list)
    for row in rows:
        if row['sha256'] and row['scientific_role'] != 'ENVIRONMENT_DEPENDENCY':
            by_hash[row['sha256']].append(row)
    priority = {'CURRENT_RESULT_OR_REBUILD': 0, 'DECISION_OR_VALIDITY_EVIDENCE': 1,
                'VALIDATION_SUMMARY': 2, 'HISTORICAL_ANALYSIS_EXPORT': 3}
    for group in by_hash.values():
        if len(group) < 2: continue
        canonical = min(group, key=lambda x: (priority.get(x['scientific_role'],5),
                                             '/output/' not in '/' + x['path'], len(x['path']), x['path']))
        for row in group:
            row['byte_identical_canonical'] = canonical['path']
            if row is not canonical:
                row['proposed_treatment'] = 'consolidate exact bytes; preserve study/run identities and path aliases'
    for row in rows:
        if row['scientific_role'] == 'ENVIRONMENT_DEPENDENCY':
            row['proposed_path'] = row['path']; continue
        part = {
            'CURRENT_RESULT_OR_REBUILD': 'final', 'DECISION_OR_VALIDITY_EVIDENCE': 'evidence',
            'VALIDATION_SUMMARY': 'validation', 'VALIDATION_SAMPLE': 'validation_samples',
            'LEGACY_DATA_PROVENANCE_REVIEW': 'legacy_data', 'LEGACY_BENCHMARK_DATA': 'legacy_data',
            'HISTORICAL_SENSITIVITY_SCENARIO': 'scenarios',
            'CLEANUP_INVESTIGATION_INTERMEDIATE': 'organization',
        }.get(row['scientific_role'], 'history')
        mode = next((v for v in ('synthetic', 'mix') if v in Path(row['path']).parts), '')
        identity = '__'.join(v for v in (row['study'], mode, re.sub('[^a-z0-9]+', '_', Path(row['path']).stem.lower()).strip('_')) if v)
        # Distinct historical releases can have the same leaf name. A stable
        # path identifier keeps all old-to-new entries unambiguous and short.
        path_id = hashlib.sha256(row['path'].encode()).hexdigest()[:10]
        identity = identity[:105].rstrip('_') + '__' + path_id
        # A unified hub with study/purpose subfolders avoids a flat collection
        # of ambiguous train.csv/summary.csv names and does not mix raw data.
        hub = 'data/archive' if part == 'legacy_data' else 'artifacts/tables'
        row['proposed_path'] = f'{hub}/{row["study"]}/{part}/{identity}.csv'
    lookup = {r['path']: r for r in rows}
    for row in rows:
        if row['byte_identical_canonical'] and row['byte_identical_canonical'] != row['path']:
            row['proposed_path'] = lookup[row['byte_identical_canonical']]['proposed_path']
    with (AUDIT / 'local_csv_purpose_catalog_2026_10_10.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    count = Counter(r['scientific_role'] for r in rows)
    research = [r for r in rows if r['scientific_role'] != 'ENVIRONMENT_DEPENDENCY']
    result = {
        'inspected_at_chicago': inspection['inspected_at_chicago'], 'remote_contacted': False,
        'csv_files':len(rows),'research_csv_files':len(research),'environment_csv_files':count['ENVIRONMENT_DEPENDENCY'],
        'classification_counts':dict(count), 'research_full_parse':sum(r['total_rows'] is not None for r in research),
        'research_header_sample':sum(r['total_rows'] is None for r in research),
        'unclassified':0, 'research_structural_caveats':sum(bool(r['structural_caveat']) for r in research),
        'research_exact_duplicate_groups':sum(len(g)>1 for g in by_hash.values()),
        'research_duplicate_copies':sum(len(g)-1 for g in by_hash.values()),
        'research_duplicate_copy_bytes':sum(r['bytes'] for r in research if r['byte_identical_canonical'] and r['byte_identical_canonical'] != r['path']),
        'large_file_hashes_or_row_counts_not_claimed':True,
        'classifying_a_path_does_not_establish_its_scientific_validity':True,
        'files_moved_or_deleted':0, 'excluded_directories':inspection['excluded_directories']}
    (AUDIT / 'local_csv_purpose_validation_2026_10_10.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='excluded_directories'},indent=2))


def decision_exports():
    source = AUDIT / 'dropout_real_results_2026_10_10.json'
    saved = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    for v in saved['trials'] + saved['baselines']:
        scores = v['test_scores']; norm = v['target_normalization']
        rows.append(dict(study='dropout_real_20261009_v1', dataset=v['dataset'], seed=v['seed'],
                         dropout=v['dropout'], arm_role='existing main baseline' if v['dropout']==0.4 else 'separate real-only dropout trial',
                         selected_dev_epoch=v['selected_dev_epoch'], epochs_run=v['epochs_run'], development_mse=v['dev_mse'],
                         test_r2=scores['r2'], test_mae=scores['mae'], test_nmae_sigma=scores['nmae_sigma'],
                         sigma_y_test=norm['sigma_y'], test_rows=norm['rows'], test_target_sha256=norm['target_table_sha256'],
                         record_path=v['record_path'], record_sha256=v['record_sha256'], source_sha256=v['source_sha256'],
                         target_transform=v['target_transform']['name'], source_evidence='audit/dropout_real_results_2026_10_10.json'))
    assert len(rows)==12 and len({(v['dataset'],v['seed'],v['dropout']) for v in rows})==12
    rows.sort(key=lambda v:(v['dataset'],v['dropout'],v['seed']))
    with (AUDIT/'local_csv_purpose_dropout_evidence_2026_10_10.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    decisions = [
        ('housing_dropout', 'Housing dropout 0/0.1 versus 0.4', 'eight separate trials plus four shared main baselines across Housing/News; not twelve extra main runs',
         'Housing mean development MSE favors existing dropout 0.4 over 0 and 0.1. Preserve unsuccessful alternatives.',
         'Only real-only downstream dropout was changed; this does not isolate other architectural or training-order effects.',
         'audit/DROPOUT_REAL_TRIALS_2026_10_09.md; audit/dropout_real_results_2026_10_10.json; audit/dropout_real_trials_20261009/worker.py'),
        ('news_dropout', 'News dropout 0/0.1 versus 0.4', 'separate decision evidence; main reports keep the existing common protocol',
         'News mean development MSE favors 0.1, but held-out R2 remains near zero; cannot state that 0.1 lost to 0.4.',
         'Changing the main protocol requires consistent evaluation across compared arms. Test scores are descriptive, not the selection criterion.',
         'audit/DROPOUT_REAL_TRIALS_2026_10_09.md; audit/dropout_real_results_2026_10_10.json'),
        ('news_log', 'News raw target / log target pilot and full rerun', 'pilot excluded from main-run reuse; fresh news_log_v1 records selected',
         'The three-arm log/no-BatchNorm pilot reduced MAE but did not uniformly improve R2; records explain the exploratory repair path.',
         'Log and normalization removal were combined. Yeo-Johnson was deferred after a round-trip check, not rejected by a completed experiment.',
         'audit/NEWS_LOG_PILOT_2026_10_08.md; audit/NEWS_LOG_RERUN_2026_10_08.md; audit/news_log_pilot_results_2026_10_08.json; output/news_log_pilot_20261008_0048/pilot_comparison.csv'),
        ('news_failures', 'News extreme downstream outputs and negative labeler targets', 'historical failure/control evidence outside selected repaired records',
         'Saved checkpoint replay confirms BatchNorm amplification in downstream failures and signed regression outputs in negative labels; inverse transforms and alignment were checked.',
         'No post-hoc clipping repair was validated. Preserve failure records, controls, source/model hashes and probes separately from current scores.',
         'audit/NEWS_SEED43_DIAGNOSIS_2026_10_08.md; audit/NEWS_NEGATIVE_LABEL_DIAGNOSIS_2026_10_08.md; audit/news_seed43_prediction_diagnosis_2026_10_08.json; audit/news_seed43_checkpoint_diagnosis_2026_10_08.json; audit/news_negative_label_replay_2026_10_08.json'),
        ('housing_faker', 'Housing geographical metadata / Faker correction', 'current main housing_no_faker_20261009 replaces flawed synthetic inputs',
         'Saved generator/probe evidence establishes anonymized global coordinates and labeler extrapolation; supports correcting metadata rather than clipping labels.',
         'Coordinate-only counterfactual probes are diagnoses, not corrected generator runs or independent final observations.',
         'audit/HOUSING_DNN_DIAGNOSIS_2026_10_08.md; audit/HOUSING_FAKER_RERUN_2026_10_09.md; audit/housing_dnn_mechanism_2026_10_08.json; audit/housing_dnn_controls_2026_10_08.json; audit/housing_dnn_replay_2026_10_08.json'),
        ('census_weighted', 'Census unweighted checkpoint collapse / weighted macro-F1 protocol', 'three verified pilot origins imported into weighted matrix; not extra observations',
         'Epoch histories and output-weight bounds explain collapse; weighted-loss plus development macro-F1 selection resolved it in the pilot.',
         'Weighting and selection changed together, so neither effect is isolated. Preserve exact original pilot records and import hashes.',
         'audit/CREDIT_CENSUS_DIAGNOSIS_2026_10_03.md; audit/CENSUS_DNN_OUTPUT_LAYER_DIAGNOSIS_2026_10_04.md; audit/CENSUS_WEIGHTED_PILOT_2026_10_04.md; audit/census_weighted_pilot_verified_results_2026_10_05.json; output/comparisons/census_kdd_training_histories_2026_10_03/histories.csv; output/comparisons/census_kdd_training_histories_2026_10_03/checkpoint_summary.csv'),
        ('credit_scope', 'Credit weighting pilot and exclusion from primary scope', 'excluded from selected 406; useful case study retained',
         'The weighted pilot improved two collapsed arms, while full+DNN test binary F1 was unchanged. Only ten positive test cases motivate scope caution.',
         'Exclusion is not evidence of unfavorable results or ineffective relabeling. Two experiment seeds share the fixed holdout.',
         'audit/CREDIT_WEIGHTED_PILOT_2026_10_06.md; audit/RESEARCH_DIRECTION_2026_10_06.md; audit/credit_weighted_verified_results_2026_10_06_1234.json'),
        ('intrusion_scope', 'Intrusion omitted / incomplete large-data study', 'outside selected main scope; incomplete status must remain explicit',
         'The direction note deprioritized further computation on cost and relevance grounds, not a completed finding that the method failed.',
         'Old queue snapshots cannot establish present liveness or completion; no remote inventory was refreshed for this audit.',
         'audit/RESEARCH_DIRECTION_2026_10_06.md; audit/EXPERIMENT_REPAIRS_2026_10_01.md; audit/missing_experiments_2026_10_01.csv'),
        ('mnist_heads', 'MNIST unused output heads / fresh downstream correction', 'selected mnist_head_fixed_20261009; old scores retained as defect evidence',
         'The artifact-level table and preserved probes explain why old MNIST downstream evaluations were replaced after fixing the forward output.',
         'Generator reuse and leakage/feature-collision caveats are separate from the output-head defect.',
         'audit/COMPLETED_RUN_PREPROCESSING_AUDIT_2026_10_09.md; audit/completed_run_preprocessing_findings_2026_10_09.csv; audit/mnist_head_rerun_20261009/README.md'),
        ('nb_pca', 'Binary-indicator quantization and numerical PCA columns', 'outside the current RF/XGB/DNN report; defect and repair evidence retained',
         'The repair and preprocessing tables identify erased binary features and categorical indicators entering numerical PCA.',
         'These implementation defects do not prove that all NB/PCA methods are inherently inferior; keep unaffected and repaired studies distinguishable.',
         'audit/EXPERIMENT_REPAIRS_2026_10_01.md; audit/pca_results_requiring_repair_2026_10_01.csv; audit/COMPLETED_RUN_PREPROCESSING_AUDIT_2026_10_09.md; audit/simulated_benchmark_audit.md'),
        ('mnist_collisions', 'Generated/held-out MNIST12 feature collisions', 'surviving current-report validity caveat',
         'Run-specific counts, independent joins and train-only provenance support compressed categorical feature collisions, not unauthorized held-out fitting.',
         'Older 313-run audit scope is historical; do not use it as coverage for all current corrected namespaces or as an estimate of score bias.',
         'audit/LEAKAGE_REVIEW_2026_09_30.md; audit/leakage_affected_runs_2026_09_30.csv; audit/leakage_mnist_direct_confirmation_2026_09_30.jsonl; audit/mnist_tvae_match_investigation_2026_09_30.jsonl'),
        ('historical_aggregation', '2025 workbook and maximum-test-score reconstruction', 'historical source/evidence archive; not corrected final metrics',
         'Original epoch logs and aggregation reconciliations establish mixed-condition averaging, test maxima and Adult/Census alias duplication.',
         'Hypothetical paper-value substitutions and fixed-model removal bounds are sensitivity scenarios, not corrected training results.',
         'audit/results_review/RESULTS_REVIEW.md; audit/results_review/local_log_manifest.json; audit/results_review/drive/log_manifest.json; audit/reconstruction/RECONSTRUCTION.md; April29.csv'),
        ('statistical_specification', 'Explicit factors and independent synthetic/mix analyses', 'current explicit-factor model; older analyses archived',
         'Preserve current equations, tables and checks; previous-version planned contrasts are still small explicit verification references.',
         'Generated-target comparators stay outside the relabeled factorial fit; duplicated score exports are not independent experiments.',
         'audit/statistical_328_20261010/EXPLICIT_ANALYSIS_HANDOFF.md; audit/statistical_328_20261010/verify_explicit.py; audit/statistical_328_20261010/separate_results/synthetic/planned_contrasts.csv; audit/statistical_328_20261010/separate_results/mix/planned_contrasts.csv'),
        ('simulation_design', 'Legacy deterministic simulation / current noisy known-distribution benchmark', 'separate studies with distinct scientific interpretation',
         'Keep old paper-replication/reference evidence, including supplement discrepancies, separately from the 378-row standalone production benchmark.',
         'Smoke epochs verify execution only. Hard relabeling need not preserve target uncertainty; L_test is a refit proxy, not a generator likelihood.',
         'audit/simulated_benchmark_audit.md; simulated_eval/README.md; audit/remote_simulated_methods_per_run.csv; output/comparisons/simulated_methods/dataset_likelihood_comparison.csv'),
        ('tabddpm_status', 'Tab-DDPM implementation versus production comparison', 'no production utility outcomes in inspected local collection',
         'Preserve implementation/design/validation notes rather than inventing a negative result or incomplete production study from smoke work.',
         'The matched unconditional adapter differs from published class-conditioned classification. No production comparison was established by this audit.',
         'audit/TAB_DDPM_IMPLEMENTATION.md; audit/TAB_DDPM_IMPLEMENTATION_PLAN_2026_10_04.md; audit/RESEARCH_DIRECTION_2026_10_06.md'),
    ]
    fields=['decision_id','question_or_alternative','relation_to_final_scope','supported_conclusion','limits_on_conclusion','evidence_paths']
    with (AUDIT/'local_csv_purpose_decision_index_2026_10_10.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(decisions)
    print('Exported 12 saved dropout observations and 15 evidence-backed decision entries.')


def annotate_decision_assets():
    """Add scientific purpose beside the older, narrower final-set labels."""
    sources = defaultdict(set)
    with (AUDIT/'local_csv_purpose_decision_index_2026_10_10.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            for path in row['evidence_paths'].split('; '):
                assert (ROOT/path).is_file(), path
                sources[path].add(row['decision_id'])
    old_path = AUDIT/'local_final_results_file_dispositions_2026_10_10.csv'
    with old_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames); rows=list(reader)
    records=defaultdict(list)
    for row in rows:
        if row['path'].startswith('output/') and row['path'].endswith('.run.json'):
            records[Path(row['path']).name].append(row['path'])
    references=defaultdict(set); matched_records={}; hashes={}

    def visit(value, studies):
        if isinstance(value,dict):
            run=value.get('run'); expected=value.get('record_sha256')
            if isinstance(run,str) and isinstance(expected,str):
                name=Path(run).name
                if not name.endswith('.run.json'): name += '.run.json'
                for path in records.get(name,[]):
                    if path not in hashes:
                        before=(ROOT/path).stat(); payload=(ROOT/path).read_bytes(); after=(ROOT/path).stat()
                        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),path
                        hashes[path]=(hashlib.sha256(payload).hexdigest(),hashlib.sha256(payload.replace(b'\r\n',b'\n')).hexdigest())
                    if expected in hashes[path]:
                        references[path].update(studies)
                        matched_records[path]={'expected_record_sha256':expected,
                                               'local_sha256':hashes[path][0],
                                               'match':'exact bytes' if expected==hashes[path][0] else 'LF-restored bytes; preserve original-byte restoration rule'}
            for v in value.values(): visit(v,studies)
        elif isinstance(value,list):
            for v in value: visit(v,studies)
        elif isinstance(value,str) and '\n' not in value and len(value)<1000:
            path=value.split('/TabularDA/',1)[1] if '/TabularDA/' in value else value
            if not path.startswith(('audit/','output/','data/','sdv trained model/','.cache/','src/','scripts/')):
                return
            target=ROOT/path
            if '..' in Path(path).parts: return
            if target.is_file(): references[path].update(studies)
            if path.startswith('.cache/'):
                for anchor in ['/output/','/data/','/sdv trained model/']:
                    if anchor in path:
                        canonical=anchor[1:]+path.split(anchor,1)[1]
                        if (ROOT/canonical).is_file(): references[canonical].update(studies)

    for path,studies in sources.items():
        if path.endswith('.json'):
            visit(json.loads((ROOT/path).read_text(encoding='utf-8-sig')),studies)
    with (AUDIT/'local_csv_purpose_catalog_2026_10_10.csv').open(encoding='utf-8',newline='') as f:
        csvs={r['path']:r for r in csv.DictReader(f)}
    extra=['csv_scientific_role','scientific_retention_note','decision_studies','csv_purpose_catalog']
    for name in extra:
        if name not in fields: fields.append(name)
    for row in rows:
        path=row['path']; notes=[]
        for name in extra: row[name]=''
        if path in csvs:
            row['csv_scientific_role']=csvs[path]['scientific_role']
            row['csv_purpose_catalog']='audit/local_csv_purpose_catalog_2026_10_10.csv'
            notes.append(csvs[path]['why_it_matters_or_does_not'])
        if path in sources or path in references:
            row['decision_studies']='; '.join(sorted(sources.get(path,set())|references.get(path,set())))
            notes.append('Preserve as research-decision evidence or its cited local source asset, even when outside the final 406.')
        row['scientific_retention_note']=' '.join(notes)
    with old_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result={'decision_source_files':len(sources),'cited_local_assets':len(references),
            'source_records_matched_by_hash':matched_records,
            'decision_source_studies':{p:sorted(v) for p,v in sorted(sources.items())},
            'local_asset_studies':{p:sorted(v) for p,v in sorted(references.items())},
            'existing_catalog_rows_annotated':sum(bool(r['scientific_retention_note']) for r in rows),
            'final_membership_classification_preserved':True}
    (AUDIT/'local_csv_purpose_decision_assets_2026_10_10.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Annotated decision purpose for',result['existing_catalog_rows_annotated'],'existing catalog entries;',
          len(matched_records),'diagnostic source records matched by saved hash.')


def validate_catalog():
    inspection_path=AUDIT/'local_csv_purpose_inspection_2026_10_10.json'
    inspection=json.loads(inspection_path.read_text(encoding='utf-8'))
    # This one input is our own working catalog, deliberately annotated above.
    # Refresh its bounded inspection so the new catalog does not describe the
    # prior version of a file edited by this very audit.
    rel='audit/local_final_results_file_dispositions_2026_10_10.csv'
    item=next(x for x in inspection['files'] if x['path']==rel)
    stat=(ROOT/rel).stat()
    with (ROOT/rel).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f); header=next(reader); sample=[]
        for i,row in enumerate(reader):
            if i>=10: break
            sample.append(row)
    item.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,header=header,sample=sample[:2],
                row_widths=dict(Counter(len(r) for r in sample)),scope={},inspected_rows=len(sample),
                total_rows=None,sha256=None,inspection='header and up to ten records')
    inspection['own_catalog_refreshed_at_chicago']=datetime.now(ZoneInfo('America/Chicago')).isoformat()
    inspection_path.write_text(json.dumps(inspection,indent=2),encoding='utf-8')
    path=AUDIT/'local_csv_purpose_catalog_2026_10_10.csv'
    with path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames);rows=list(reader)
    row=next(x for x in rows if x['path']==rel)
    row.update(bytes=item['bytes'],columns=len(header),header_json=json.dumps(header),
               inspected_rows=item['inspected_rows'],total_rows='',sha256='',observed_scope_json='{}')
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    expected={x['path']:x for x in inspection['files']}
    lookup={r['path']:r for r in rows}
    assert len(lookup)==len(rows)==len(expected) and set(lookup)==set(expected)
    destinations=defaultdict(list)
    for row in rows:
        original=expected[row['path']]; stat=(ROOT/row['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(original['bytes'],original['mtime_ns']),row['path']
        assert row['purpose'] and row['why_it_matters_or_does_not'] and row['proposed_treatment']
        for e in row['scientific_evidence'].split('; '):
            if e: assert (ROOT/e.split(':',1)[0]).is_file(),e
        if row['byte_identical_canonical']:
            canonical=lookup[row['byte_identical_canonical']]
            assert row['sha256']==canonical['sha256'] and row['bytes']==canonical['bytes']
        if row['scientific_role']!='ENVIRONMENT_DEPENDENCY':
            assert len(str(ROOT/row['proposed_path']))<240
            destinations[row['proposed_path']].append(row)
    for group in destinations.values():
        if len(group)>1: assert len({r['sha256'] for r in group})==1 and group[0]['sha256']
    with (AUDIT/'local_csv_purpose_dropout_evidence_2026_10_10.csv').open(encoding='utf-8',newline='') as f:
        dropout=list(csv.DictReader(f))
    source=json.loads((AUDIT/'dropout_real_results_2026_10_10.json').read_text(encoding='utf-8'))
    source_rows={(v['dataset'],v['seed'],v['dropout']):v for v in source['trials']+source['baselines']}
    for row in dropout:
        original=source_rows[row['dataset'],int(row['seed']),float(row['dropout'])]
        assert float(row['development_mse'])==original['dev_mse']
        for key, saved in [('test_r2','r2'),('test_mae','mae'),('test_nmae_sigma','nmae_sigma')]:
            assert float(row[key])==original['test_scores'][saved]
        assert row['record_sha256']==original['record_sha256']
        assert row['test_target_sha256']==original['target_normalization']['target_table_sha256']
    assert len(dropout)==12 and sum(float(r['dropout'])==0.4 for r in dropout)==4
    with (AUDIT/'local_csv_purpose_decision_index_2026_10_10.csv').open(encoding='utf-8',newline='') as f:
        decisions=list(csv.DictReader(f))
    assert len(decisions)==15 and len({r['decision_id'] for r in decisions})==15
    for row in decisions:
        for evidence in row['evidence_paths'].split('; '): assert (ROOT/evidence).is_file(),evidence
    result_path=AUDIT/'local_csv_purpose_validation_2026_10_10.json'
    result=json.loads(result_path.read_text(encoding='utf-8'))
    result.update(validated_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                  catalog_paths_exactly_match_discovery=True,
                  source_csvs_unchanged_except_intentionally_annotated_own_catalog=True,
                  evidence_paths_exist=True, proposed_destinations_unambiguous=True,
                  duplicate_groups_require_exact_full_file_hashes=True,
                  separate_generated_csv_checks={'catalog_rows':len(rows),'decision_entries':15,
                                                 'dropout_rows':12,'shared_main_baseline_rows':4},
                  source_files_modified=['audit/local_final_results_file_dispositions_2026_10_10.csv'])
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Coverage, CSV stability, evidence paths, destination collisions, exact duplicate identities and saved-value exports verified.')


if __name__ == '__main__':
    import sys
    if '--catalog-only' not in sys.argv:
        inspect_all()
    catalog()
    decision_exports()
    annotate_decision_assets()
    validate_catalog()
