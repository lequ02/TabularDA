"""Prepare scalar report inputs and replication specifications; no model fitting."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'output/statistical_analysis_328_20261010'
EVIDENCE = ROOT / 'audit/statistical_328_20261010/snapshot/records'
STAGE = ROOT / 'tmp/statistical_328_csv_replication'
CLASS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
REG = ['news', 'california_housing']
LABELS = ['rf', 'xgb', 'dnn']
ARMS = ['synthetic', 'mix']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed(value):
    if value is None or value == '':
        return None
    if isinstance(value, (int, float, bool)):
        return value
    # Existing statistical CSVs have text identifiers and numeric scalar values.
    try:
        number = float(value)
    except ValueError:
        return value
    assert math.isfinite(number)
    return number


tables = []


def table(name, rows, purpose, origin, headers=None):
    assert rows, name
    headers = headers or list(rows[0])
    assert len(headers) == len(set(headers))
    assert not {'condition', 'source', 'mode', 'group', 'approach_has_y'} & set(headers)
    tables.append(dict(filename=name, headers=headers,
                       rows=[[typed(row.get(h)) for h in headers] for row in rows],
                       purpose=purpose, origin=origin))


snapshot = json.loads((REPORT / 'source_snapshot.json').read_text())
verification = json.loads((REPORT / 'latest_source_verification.json').read_text())
assert snapshot['selected_completed'] == 406 and not snapshot['missing']
assert verification['checked_files'] == 522 and not verification['mismatches']
raw = read_csv(REPORT / 'frozen_inputs/scores.csv')
by_record = defaultdict(dict)
record_info = {}
for row in raw:
    assert row['domain'] == 'real'
    assert row['dataset'] in CLASS + REG
    assert row['metric'] not in by_record[row['record']]
    by_record[row['record']][row['metric']] = float(row['value'])
    record_info[row['record']] = row
assert len(record_info) == 406
record_snapshot = {r['path']: r for r in snapshot['records']}
assert set(record_info) == set(record_snapshot)

# Read the saved test normalization from the four real-only regression records.
# Every scalar export uses this verified shared test partition, never a new split.
normalizations = {}
normalization_origins = {}
for path, row in record_info.items():
    if row['dataset'] in REG and row['labeler'] == 'original':
        local = EVIDENCE / path
        assert sha(local) == row['record_sha256']
        record = json.loads(local.read_text())
        norm = record['target_normalization']
        assert norm['split'] == 'test' and norm['ddof'] == 0
        assert norm['target_table_sha256'] == row['test_hash']
        normalizations[(row['dataset'], int(row['seed']))] = norm
        normalization_origins[(row['dataset'], int(row['seed']))] = path
assert len(normalizations) == 4

d2_rows = read_csv(REPORT / 'frozen_inputs/regression_d2_scores.csv')
assert len(d2_rows) == 116
d2 = {}
for row in d2_rows:
    sidecar_path = EVIDENCE / row['sidecar']
    assert sha(sidecar_path) == row['sidecar_sha256']
    sidecar = json.loads(sidecar_path.read_text())
    norm = sidecar['absolute_error_normalization']
    assert sidecar['run_record_sha256'] == row['record_sha256']
    assert sidecar['test_scores']['d2_absolute_error'] == float(row['value'])
    assert norm['split'] == 'test'
    assert norm['target_table_sha256'] == record_info[row['record']]['test_hash']
    assert math.isclose(1 - norm['prediction_mae'] / norm['baseline_mae'],
                        float(row['value']), abs_tol=1e-12)
    d2[row['record']] = (row, sidecar)

INPUT_HEADERS = ['dataset', 'task', 'seed', 'generator', 'has_y', 'labeler', 'row_role',
                 'macro_F1', 'r2', 'mae', 'nmae_sigma', 'd2_absolute_error',
                 'sigma_y', 'median_y', 'baseline_mae', 'd2_prediction_mae',
                 'record', 'record_sha256', 'test_table_sha256', 'protocol']
prepared = {}
for path, row in record_info.items():
    role = ('real_only_reference' if row['labeler'] == 'original' else
            'generated_target_baseline' if row['labeler'] == 'generated' else 'relabeled')
    task = 'classification' if row['dataset'] in CLASS else 'regression'
    item = dict(dataset=row['dataset'], task=task, seed=int(row['seed']),
                generator=None if role == 'real_only_reference' else row['generator'],
                has_y=None if role == 'real_only_reference' else {'full': 1, 'xonly': 0}[row['source']],
                labeler=row['labeler'] if role == 'relabeled' else None, row_role=role,
                record=path, record_sha256=row['record_sha256'],
                test_table_sha256=row['test_hash'], protocol=row['protocol'])
    assert item['record_sha256'] == record_snapshot[path]['sha256']
    scores = by_record[path]
    if task == 'classification':
        item['macro_F1'] = scores['f1_macro']
    else:
        norm = normalizations[(row['dataset'], int(row['seed']))]
        item.update({metric: scores[metric] for metric in ['r2', 'mae', 'nmae_sigma']})
        item['sigma_y'] = norm['sigma_y']
        assert math.isclose(item['nmae_sigma'], item['mae'] / item['sigma_y'], rel_tol=1e-12)
        dr, sidecar = d2[path]
        an = sidecar['absolute_error_normalization']
        item.update(d2_absolute_error=float(dr['value']), median_y=an['median_y'],
                    baseline_mae=an['baseline_mae'], d2_prediction_mae=an['prediction_mae'])
    prepared[path] = item

for arm in ARMS:
    rows = [item for path, item in prepared.items()
            if record_info[path]['mode'] in [arm, 'original']]
    rows.sort(key=lambda r: (r['dataset'], r['seed'], r['row_role'], r['generator'] or '',
                            r['has_y'] if r['has_y'] is not None else -1, r['labeler'] or ''))
    assert len(rows) == 210
    assert sum(r['task'] == 'classification' and r['row_role'] == 'relabeled' for r in rows) == 120
    assert sum(r['task'] == 'regression' for r in rows) == 60
    table(f'{arm}_input.csv', rows,
          'Run-level unrounded scores and references. Fit ANOVA only to classification relabeled rows.',
          'Verified frozen run scores; saved test normalization and D2 sidecars.', INPUT_HEADERS)

model_specs = []
R_FORMULAS = {
    'factorial': 'macro_F1 ~ dataset + generator * has_y * labeler * seed',
    'seed_block': 'macro_F1 ~ dataset + seed + generator * has_y * labeler',
    'dataset_interaction': 'macro_F1 ~ dataset * (has_y * labeler) + generator * has_y * labeler + seed',
}
PURPOSES = {
    'factorial': 'Primary ANOVA requested in the report.',
    'seed_block': 'Restricted model for testing all seed interactions jointly.',
    'dataset_interaction': 'Check whether has_y and labeler effects differ across datasets.',
}
for arm in ARMS:
    for row in read_csv(REPORT / arm / 'model_summary.csv'):
        model_specs.append(dict(analysis=arm, input_file=f'{arm}_input.csv', model=row['name'],
                                purpose=PURPOSES[row['name']], row_filter="task == 'classification' and row_role == 'relabeled'",
                                response='macro_F1', formula=R_FORMULAS[row['name']],
                                python_formula=row['formula'], factor_columns='dataset;generator;has_y;labeler;seed',
                                coding='All five columns categorical, sum-to-zero contrasts; intercept included.',
                                R_setup='for (v in c("dataset","generator","has_y","labeler","seed")) data[[v]] <- factor(data[[v]]); options(contrasts=c("contr.sum","contr.poly"))',
                                R_anova='car::Anova(lm(formula, data=data), type=2)',
                                Python_anova='statsmodels.stats.anova.anova_lm(statsmodels.formula.api.ols(python_formula, data=data).fit(), typ=2)',
                                n=row['n'], rank=row['rank'], residual_df=row['df_resid']))
table('model_specifications.csv', model_specs, 'Exact equations, categorical coding and 120-row model filters.',
      'Report model_summary.csv and analyze_explicit.py.')

rules = [
    ('scope', 'All inputs', 'Real-data report only: Adult, Census KDD, Covertype, MNIST12, MNIST28, News, California Housing. RF/XGB/DNN labelers. Seeds 42/43. Credit, Intrusion, NB, PCA/GMM and simulations excluded.'),
    ('two_analyses', 'All tables', 'synthetic and mix are separate analyses; analysis identifies the table origin and never enters a model as a factor. Combined real_data report contains these same two analyses.'),
    ('training_rows', 'synthetic_input.csv', '100000 synthetic rows only; real-only reference rows are trained on real rows only.'),
    ('training_rows', 'mix_input.csv', 'All real training rows plus all 100000 synthetic rows, not a fixed 50/50 mixture; real-only reference rows are trained on real rows only.'),
    ('role_filter', 'ANOVA and model intervals', "Use task='classification' and row_role='relabeled'. All 120 cells observed. Exclude generated_target_baseline and real_only_reference from the factorial fit."),
    ('baseline_matching', 'matched_relabeling_scores.csv', 'Join a relabeled row to exactly one generated-target baseline by dataset, seed and generator within its input CSV. This baseline is reused as a comparator, not counted as independent duplicated observations.'),
    ('dataset_contrasts', 'planned_contrasts.csv', 'Average seeds and generators equally within each dataset. Relabeled mean also averages has_y and the three labelers equally. For has_y-specific means, average only the selected has_y level. Then subtract and average the five dataset differences equally. See contrast_weights.csv.'),
    ('paired_t', 'planned_contrasts.csv and labeler_effects.csv', 'delta vector has five dataset differences. sd=SD(delta, ddof=1); se=sd/sqrt(5); t=mean(delta)/se; df=4; p=2*t.sf(abs(t),4); 95% CI=mean(delta) +/- t.ppf(.975,4)*se; cohen_dz=mean(delta)/sd.'),
    ('labeler_comparisons', 'labeler_effects.csv', 'For each generator/has_y/labeler cell, average the two seeds within each dataset; subtract the same-generator generated-target mean averaged over those seeds. Apply paired inference to the resulting five differences.'),
    ('exact_sign_flip', 'planned_contrasts.csv and labeler_effects.csv', 'Enumerate all 32 sign assignments of five delta values. Two-sided p=mean(abs(mean(sign*delta)) >= abs(mean(delta))-1e-12). Minimum possible p is .0625.'),
    ('wilcoxon', 'planned_contrasts.csv and labeler_effects.csv', "Supplementary only: scipy.stats.wilcoxon(delta, method='approx', zero_method='wilcox'), two-sided; p=1 when all abs(delta)<1e-12."),
    ('wins_losses_ties', 'planned_contrasts.csv and labeler_effects.csv', 'Wins: delta>1e-10. Losses: delta<-1e-10. Ties: abs(delta)<=1e-10.'),
    ('relative_percent', 'Planned and labeler comparisons', '100 * (first mean - second mean) / second mean. For labeler comparisons second mean is its matched generated-target mean. Percentage points = 100 * macro-F1 change.'),
    ('holm_families', 'Adjusted p values', 'Within each analysis separately: three factorial marginal contrasts, four planned paired-t contrasts, four planned exact-sign-flip contrasts, and twelve labeler paired-t comparisons are separate Holm families. ANOVA and nested-model p values are nominal.'),
    ('holm_formula', 'Adjusted p values', 'Sort m raw p values ascending. adjusted_sorted[i]=min(1,max over j<=i of ((m-j+1)*p_sorted[j])), using 1-based ranks. Restore the original row order.'),
    ('estimated_means', 'estimated_means.csv', 'Average primary-model prediction design rows equally over all five datasets and two seeds for each generator/has_y/labeler cell. Means equal raw cell averages on this balanced grid. Model CI uses t critical value with 92 residual df and fitted OLS covariance.'),
    ('factorial_marginals', 'factorial_effects.csv', 'Average primary-model predictions over every observed combination of the other factors; contrast TVAE-CTGAN, has_y1-has_y0, seed43-seed42. Use fitted OLS contrast covariance and t with 92 residual df.'),
    ('anova', 'anova_results.csv', 'Type II ANOVA, sum-to-zero categorical coding. partial_eta_sq=term sum_sq/(term sum_sq+model residual sum_sq). Leave residual F/p/partial_eta_sq blank.'),
    ('nested_F', 'joint_tests.csv', 'Compare seed_block with factorial (11 added df), and seed_block with dataset_interaction (20 added df). F=((SSE_restricted-SSE_full)/df_added)/(SSE_full/df_resid_full). Use F.sf with df_added and full-model residual df.'),
    ('diagnostics', 'model_summary.csv', 'R_squared=1-SSE/SST; adjusted_R_squared=1-(1-R_squared)*(n-1)/df_resid. Shapiro p=scipy.stats.shapiro(fitted residuals). Residual and QQ plots are reproducible from each model specification and its input rows.'),
    ('regression_means', 'regression_summary.csv', 'For each task and metric separately: original averages 2 seeds; generated averages 2 generators x 2 seeds; each has_y level averages 2 generators x 3 labelers x 2 seeds. Do not pool regression with classification or the two regression tasks.'),
    ('regression_metrics', 'Regression rows', 'R2 and D2: higher is better. NMAE: lower is better. NMAE=mae/saved sigma_y, using original-scale real test targets, population SD ddof=0. D2=1-d2_prediction_mae/saved baseline_mae; baseline_mae=mean(abs(y_test-saved median_y)). Read saved denominators; do not estimate them from these summary rows.'),
    ('mae_precision', 'Regression rows', 'mae is the run record value used for NMAE. d2_prediction_mae is the verified saved-prediction value used by the D2 sidecar. Small serialization/precision differences can exist; use each metric with its own saved numerator.'),
    ('significance', 'Result tables', '*** p<.001; ** p<.01; * p<.05; . p<.10; otherwise blank. significant_05=1 exactly when the stated p_for_significance<.05 (bold in PDF). CSV cannot store bold formatting; these columns preserve its meaning.'),
    ('intervals', 'All result tables', '95% confidence intervals are unadjusted even when the accompanying p value is Holm-adjusted.'),
    ('inference_limits', 'All findings', 'ANOVA/model intervals assume independent homoskedastic errors and are exploratory. Seeds share held-out test rows; seed is categorical with two fixed levels. Paired inference uses five dataset means, not 120 independent tasks. Dataset interactions and nonnormal residuals limit pooled inference.'),
    ('MNIST_caveat', 'MNIST12 TVAE', 'Repaired output heads; retained TVAE synthetic tables have documented held-out feature collisions. Preserve this caveat in replicated findings.'),
    ('selected_experiment_provenance', 'provenance.csv', 'News uses log-target reruns scored in original units; Housing uses no-Faker reruns; Census KDD uses weighted macro-F1 development checkpoint selection; other datasets use development loss. Generator production settings: 500 epochs, batch 500, CUDA, 100000 synthetic rows. No experiment reruns were performed for this export.'),
    ('reference_overlap', 'Input files', 'Each input has 210 rows, including the same 14 real-only references. Union by record contains 406 runs. Do not count shared reference rows twice.'),
    ('frozen_source_time', 'All inputs', f"Score snapshot {snapshot['checked_at_chicago']}; 406 records plus 116 D2 sidecars rechecked {verification['checked_at_chicago']}, America/Chicago; no hash mismatches."),
    ('csv_format', 'All CSVs', 'UTF-8, comma-delimited, one header row, no display rounding. Blank means not applicable, never zero. decimal point is a period. Input rows are run-level final held-out scores, not individual predictions or retraining inputs.'),
    ('R_status', 'model_specifications.csv', 'Models in this report were fitted in Python on the research server. The R equations/coding specify a replication route; the optional R route has not been executed.'),
]
table('analysis_rules.csv', [dict(rule=name, applies_to=target, definition=definition) for name, target, definition in rules],
      'Aggregation, inference, metric equations, corrections, scope and caveats.', 'Current report and analysis source.')

display = dict(adult='Adult', census_kdd='Census KDD', covertype='Covertype',
               mnist12='MNIST12', mnist28='MNIST28', news='News', california_housing='California Housing')
levels = [dict(variable='dataset', level=d, meaning=display[d], applicable_rows='classification' if d in CLASS else 'regression') for d in CLASS + REG]
levels += [dict(variable='generator', level=g, meaning=g.upper(), applicable_rows='relabeled or generated_target_baseline') for g in ['ctgan', 'tvae']]
levels += [dict(variable='has_y', level=h, meaning='Generator fitted to X only' if h == 0 else 'Generator fitted to (X,Y)', applicable_rows='relabeled; generated_target_baseline only has_y=1') for h in [0, 1]]
levels += [dict(variable='labeler', level=l, meaning=meaning, applicable_rows='relabeled') for l, meaning in [('rf', 'Random forest'), ('xgb', 'XGBoost'), ('dnn', 'Neural network')]]
levels += [dict(variable='seed', level=s, meaning='Fixed categorical experiment seed; same prepared held-out rows for both seeds', applicable_rows='all inputs') for s in [42, 43]]
levels += [dict(variable='row_role', level=r, meaning=meaning, applicable_rows='all inputs') for r, meaning in [('relabeled', 'Generated X with labeler-predicted Y'), ('generated_target_baseline', 'Generate (X,Y), retaining generated Y; no labeler'), ('real_only_reference', 'Original real training data; no generator or labeler')]]
levels += [dict(variable='analysis', level=arm, meaning='Independent synthetic-only analysis' if arm == 'synthetic' else 'Independent real plus synthetic analysis', applicable_rows='result/specification tables only; never an ANOVA factor') for arm in ARMS]
table('factor_levels.csv', levels, 'Factor names, every level and reference-row meanings.', 'Report design definitions.')

dictionary = {
    'dataset': ('categorical', 'Real-data task; see factor_levels.csv.'),
    'task': ('categorical', 'classification or regression; separate metric families.'),
    'seed': ('categorical integer', '42 or 43, treated as fixed factors, not numeric covariates.'),
    'generator': ('categorical', 'ctgan or tvae; blank for real-only references.'),
    'has_y': ('categorical integer', '0: generator fitted on X; 1: fitted on (X,Y); blank for real-only references.'),
    'labeler': ('categorical', 'rf, xgb or dnn; blank for generated targets and real-only references.'),
    'row_role': ('categorical', 'relabeled, generated_target_baseline, real_only_reference; a row filter, not a fitted factor.'),
    'macro_F1': ('number, 0-1', 'Equal-weight average of class F1 scores; the saved f1_macro score. Classification only.'),
    'r2': ('number', 'Saved original-scale regression R squared. Higher is better.'),
    'mae': ('number, target units', 'Saved run-record mean absolute test error; numerator for NMAE.'),
    'nmae_sigma': ('number', 'Saved MAE / saved population SD of real test targets. Lower is better.'),
    'd2_absolute_error': ('number', 'Saved D2=1-d2_prediction_mae/baseline_mae. Higher is better.'),
    'sigma_y': ('number, target units', 'Saved real-test target SD, ddof=0, read from the corresponding verified real-only record.'),
    'median_y': ('number, target units', 'Saved median of real test targets in D2 sidecar.'),
    'baseline_mae': ('number, target units', 'Saved real-test error of constant median_y predictor; D2 denominator.'),
    'd2_prediction_mae': ('number, target units', 'Saved mean absolute error from held-out predictions, the D2 numerator.'),
    'record': ('text path', 'Original remote repository-relative *.run.json path; join key across input and provenance tables.'),
    'record_sha256': ('text', 'SHA256 of the frozen completed run record.'),
    'test_table_sha256': ('text', 'SHA256 identifying the prepared real held-out test table.'),
    'protocol': ('text', 'Namespace/protocol recorded in frozen score snapshot; distinguishes corrected variants.'),
    'analysis': ('categorical identifier', 'synthetic or mix, identifying separate analyses in combined result tables; never modeled.'),
    'model': ('categorical identifier', 'factorial, seed_block or dataset_interaction; see model_specifications.csv.'),
    'has_y0': ('number', 'Mean score for relabeled rows with has_y=0, in regression_summary.csv.'),
    'has_y1': ('number', 'Mean score for relabeled rows with has_y=1, in regression_summary.csv.'),
    'original': ('number', 'Mean real-only reference score in regression_summary.csv.'),
    'generated': ('number', 'Mean generated-target baseline score for the relevant table row.'),
    'relabeled': ('number', 'Mean relabeled score for the relevant labeler comparison.'),
    'mean': ('number', 'Estimated score in estimated_means.csv; mean dataset-level difference in contrast tables.'),
    'ci_low': ('number', 'Lower bound of an unadjusted 95% interval; method specified in analysis_rules.csv.'),
    'ci_high': ('number', 'Upper bound of an unadjusted 95% interval; method specified in analysis_rules.csv.'),
    'p_for_significance': ('number', 'The stated nominal or Holm-adjusted p used to reproduce stars/bold.'),
    'significance': ('text', '***, **, *, ., or blank; see analysis_rules.csv.'),
    'significant_05': ('integer, 0/1', '1 reproduces PDF bold where p_for_significance<.05; 0 otherwise.'),
}
for key, meaning in {
    'sum_sq': 'Type II effect or residual sum of squares.', 'df': 'ANOVA term degrees of freedom; model residual df in factorial_effects.csv.',
    'F': 'ANOVA or nested-model F statistic.', 'PR(>F)': 'Nominal ANOVA F-test p value.',
    'partial_eta_sq': 'Term SS / (term SS + residual SS).', 'p': 'Nominal model contrast or nested F-test p.',
    'p_holm': 'Holm-adjusted model marginal contrast p (three within-analysis tests).',
    'p_t': 'Unadjusted two-sided paired-t p across five dataset differences.',
    'p_t_holm': 'Paired-t p adjusted over four planned or twelve labeler tests within analysis.',
    'p_exact': 'Unadjusted two-sided exact sign-flip p over 32 assignments.',
    'p_exact_holm': 'Exact sign-flip p adjusted over four planned tests within analysis.',
    'p_wilcoxon': 'Supplementary approximate Wilcoxon two-sided p.',
    'score_change': 'First mean minus second mean, macro-F1 units.',
    'percentage_points': '100 times macro-F1 score change.',
    'relative_percent': '100 times change divided by comparator mean.',
    'difference': 'Relabeled F1 minus matched generated F1, or left dataset mean minus right dataset mean.',
    'effect': 'Primary-model marginal mean difference.', 'first_mean': 'Model mean for first contrast level.',
    'second_mean': 'Model mean for second contrast level.', 'left_mean': 'First overall planned contrast mean.',
    'right_mean': 'Second overall planned contrast mean.', 'left': 'First dataset-specific planned contrast mean.',
    'right': 'Second dataset-specific planned contrast mean.', 'cohen_dz': 'Mean dataset difference / sample SD of dataset differences.',
    't': 'Two-sided paired t statistic with four df.', 'n_datasets': 'Five independent task summaries for paired inference.',
    'wins': 'Count delta>1e-10.', 'losses': 'Count delta<-1e-10.', 'ties': 'Count abs(delta)<=1e-10.',
    'median': 'Median of five dataset differences.', 'value_generated': 'Score of the one matched generated-target baseline.',
    'record_generated': 'Repository-relative run record of matched generated-target baseline.',
    'df_num': 'Nested-model numerator degrees of freedom.', 'df_den': 'Nested-model full-model residual degrees of freedom.',
    'n': 'Model observation count.', 'rank': 'Number of identifiable coefficients including intercept.',
    'columns': 'Design matrix column count.', 'df_model': 'Model rank minus one.', 'df_resid': 'Model n minus rank.',
    'r_squared': 'OLS fit R squared (different from per-run regression r2).',
    'adjusted_r_squared': 'OLS adjusted R squared.', 'shapiro_p': 'Shapiro-Wilk p for model residuals.',
}.items():
    dictionary[key] = ('number', meaning)
table('data_dictionary.csv', [dict(column=k, data_type=t, definition=d) for k, (t, d) in dictionary.items()],
      'Input column meanings and statistical result definitions.', 'Report and saved metric definitions.')

provenance = []
for path, item in sorted(prepared.items()):
    row = record_info[path]
    snap = record_snapshot[path]
    partition = snapshot['test_partitions'][f"{item['dataset']}/{item['seed']}"]
    norm_origin = normalization_origins.get((item['dataset'], item['seed']))
    pr = dict(record=path, record_sha256=item['record_sha256'], dataset=item['dataset'], seed=item['seed'],
              training_data='real_only' if row['mode'] == 'original' else row['mode'], row_role=item['row_role'],
              generator=item['generator'], has_y=item['has_y'], labeler=item['labeler'], scope=snap['scope'],
              test_table_sha256=item['test_table_sha256'], test_rows=partition['rows'],
              test_source_ids_sha256=partition['ids_sha256'], weight_present=int(snap['weight_present']),
              predictions_present=int(snap['predictions_present']), source_id_order_verified=int(snap['source_id_order_verified']),
              normalization_record=norm_origin, normalization_record_sha256=record_info[norm_origin]['record_sha256'] if norm_origin else None,
              normalization_split='test' if norm_origin else None, normalization_ddof=0 if norm_origin else None,
              sidecar=None, sidecar_sha256=None, predictions_sha256=None,
              frozen_snapshot_chicago=snapshot['checked_at_chicago'], source_rechecked_chicago=verification['checked_at_chicago'])
    if path in d2:
        dr, sidecar = d2[path]
        pr.update(sidecar=dr['sidecar'], sidecar_sha256=dr['sidecar_sha256'], predictions_sha256=sidecar['predictions_sha256'])
    provenance.append(pr)
table('provenance.csv', provenance, '406 distinct run identifiers, hashes, test partitions and regression sidecar provenance.',
      'Verified source_snapshot.json; frozen D2 sidecars; latest_source_verification.json.')
table('software_versions.csv', [dict(component=name, version=version, environment='/home/thuy/miniconda3/envs/env',
                                    evidence='Frozen source snapshot; current research-server versions agree')
                                for name, version in [('python', '3.10.21'), ('numpy', '1.26.4'), ('pandas', '2.2.3'),
                                                      ('scipy', '1.15.3'), ('statsmodels', '0.15.0'), ('patsy', '1.0.3')]],
      'Statistical analysis software versions. Optional R route not executed.', 'Frozen snapshot and read-only research-server version check.')


def stars(value):
    if value in ['', None]:
        return ''
    p = float(value)
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '.' if p < .10 else ''


def combine(filename, purpose, p_column=None, transform=None):
    rows = []
    for arm in ARMS:
        for original in read_csv(REPORT / arm / filename):
            row = dict(analysis=arm, **original)
            if transform:
                row = transform(row)
            if p_column:
                p = row[p_column]
                row.update(p_for_significance=p, significance=stars(p),
                           significant_05=int(p not in ['', None] and float(p) < .05))
            rows.append(row)
    table(filename, rows, purpose, f'Existing synthetic/{filename} and mix/{filename}; original numerical values preserved.')


anova = []
for arm in ARMS:
    for model in R_FORMULAS:
        for original in read_csv(REPORT / arm / f'{model}_anova.csv'):
            term = original['term']
            readable = term
            for factor in ['dataset', 'generator', 'has_y', 'labeler', 'seed']:
                readable = readable.replace(f'C({factor}, Sum)', factor)
            readable = readable.replace(':', ' x ')
            p = original['PR(>F)']
            anova.append(dict(analysis=arm, model=model, term=term, term_display=readable, **{k:v for k,v in original.items() if k != 'term'},
                              p_for_significance=p, significance=stars(p), significant_05=int(p != '' and float(p) < .05)))
table('anova_results.csv', anova, 'All three models per analysis, unrounded Type II ANOVA statistics and PDF significance notation.',
      'Existing factorial_anova.csv, seed_block_anova.csv, dataset_interaction_anova.csv in each analysis.')
combine('model_summary.csv', 'Model sizes, rank, R squared and residual Shapiro diagnostics.', transform=lambda r: dict(analysis=r.pop('analysis'), model=r.pop('name'), **r))
combine('joint_tests.csv', 'Supporting joint seed-interaction and dataset-interaction F tests.', 'p')
combine('estimated_means.csv', 'Primary-model cell means and unadjusted 95% model intervals.')
combine('factorial_effects.csv', 'Primary-model marginal contrasts and Holm p across three tests.', 'p_holm')
combine('planned_contrasts.csv', 'Four five-dataset planned contrasts per analysis, intervals and p values.', 'p_t_holm')
combine('labeler_effects.csv', 'Twelve matched generated-baseline comparisons per analysis.', 'p_t_holm')
combine('dataset_effects.csv', 'The five task-level differences behind each planned comparison and plot.')
combine('regression_summary.csv', 'All regression table means for News and California Housing, separately.')

weights_a = read_csv(REPORT / 'synthetic/contrast_weights.csv')
assert weights_a == read_csv(REPORT / 'mix/contrast_weights.csv')
table('contrast_weights.csv', weights_a, 'Observed construction-cell weights for four planned contrasts; apply separately within each dataset and analysis after averaging seeds.',
      'Identical contrast_weights.csv in both analyses.')

# Keep the baseline join compact and explicit, without duplicate metric aliases.
matched = []
for arm in ARMS:
    for row in read_csv(REPORT / arm / 'matched_relabeling_scores.csv'):
        matched.append(dict(analysis=arm, dataset=row['dataset'], seed=row['seed'], generator=row['generator'],
                            has_y=row['has_y'], labeler=row['labeler'], macro_F1=row['macro_F1'],
                            value_generated=row['value_generated'], difference=row['difference'],
                            record=row['record'], record_generated=row['record_generated']))
table('matched_relabeling_scores.csv', matched, '240 relabeled classification scores joined to their single matched generated-target comparator.',
      'Existing matched_relabeling_scores.csv in both analyses.')

assert len(tables) == 19
file_index = [dict(read_order=i + 1, file=t['filename'], rows=len(t['rows']), columns=len(t['headers']),
                   purpose=t['purpose'], original_evidence=t['origin']) for i, t in enumerate(tables)]
file_index += [dict(read_order=20, file='file_index.csv', rows=21, columns=6,
                   purpose='Bundle contents and what each file reproduces.', original_evidence='This bundle.'),
               dict(read_order=21, file='manifest.csv', rows=20, columns=4,
                    purpose='SHA256 and dimensions for all other delivered CSVs; verify export integrity.', original_evidence='Final exported CSV bytes.')]
table('file_index.csv', file_index, 'Bundle contents and reading order.', 'This bundle.')
assert len(tables) == 20
STAGE.mkdir(parents=True, exist_ok=True)
(STAGE / 'tables.json').write_text(json.dumps(tables, ensure_ascii=False, allow_nan=False), encoding='utf-8')
print(json.dumps(dict(input_files=2, rows_per_input=210, distinct_runs=len(prepared),
                      regression_sidecars=len(d2), prepared_csvs=len(tables), final_csvs_including_manifest=21)))
