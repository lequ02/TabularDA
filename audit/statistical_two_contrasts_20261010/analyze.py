"""Refit frozen results for two research contrasts. Run only on the server."""
import hashlib
import itertools
import json
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats
import scipy
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO = Path('/home/thuy/Research/minh_data_synth/TabularDA')
assert HERE.is_relative_to(REPO / '.cache'), 'Statistical execution must be remote'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
CLASS = ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28']
REG = ['news', 'california_housing']
LABELS = ['rf', 'xgb', 'dnn']
CONTRASTS = ['relabel_minus_generated', 'has_y1_minus_has_y0']
snapshot = json.loads((HERE / 'source_snapshot.json').read_text())
d2_verification = json.loads((HERE / 'd2_verification.json').read_text())
raw = pd.read_csv(HERE / 'scores.csv')
d2 = pd.read_csv(HERE / 'regression_d2_scores.csv')
assert len(snapshot['records']) == raw.record.nunique() == 406 and not snapshot['missing']
assert len(d2) == len(d2_verification['sidecars']) == 116
expected = {r['path']: r['sha256'] for r in snapshot['records']}
expected.update({r['sidecar']: r['sidecar_sha256'] for r in d2_verification['sidecars']})
assert len(expected) == 522
for name, digest in expected.items():
    assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest, name
verified_values = 0
for name, frame in raw.groupby('record'):
    record = json.loads((REPO / name).read_text())
    assert frame.record_sha256.eq(expected[name]).all()
    for row in frame.itertuples():
        np.testing.assert_allclose(row.value, record['test_scores'][row.metric], rtol=1e-12, atol=1e-12)
        verified_values += 1
    if record['dataset'] in REG:
        norm = record['target_normalization']
        assert norm['split'] == 'test' and norm['ddof'] == 0 and norm['rows'] > 0
        assert norm['target_table_sha256'] and norm['definition']
        np.testing.assert_allclose(record['test_scores']['nmae_sigma'],
                                   record['test_scores']['mae'] / norm['sigma_y'], rtol=1e-12)
verification = dict(checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                    checked_files=522, verified_source_values=verified_values, mismatches=[])
(OUT / 'source_verification.json').write_text(json.dumps(verification, indent=2))

def save(frame, name):
    frame.to_csv(OUT / name, index=False)

def contrasts(frame, datasets, arm, metric):
    """Keep native scores; orient differences so positive always means better."""
    selected = frame[frame.dataset.isin(datasets) & frame['mode'].eq(arm)].copy()
    primary = selected[selected.labeler.isin(LABELS)].copy()
    primary['has_y'] = primary.source.map({'full': 1, 'xonly': 0})
    baseline = selected[selected.labeler.eq('generated')].copy()
    keys = ['dataset', 'seed', 'generator', 'has_y', 'labeler']
    expected_cells = set(itertools.product(datasets, [42, 43], ['ctgan', 'tvae'], [0, 1], LABELS))
    assert set(primary[keys].itertuples(index=False, name=None)) == expected_cells
    assert len(primary) == len(expected_cells)
    assert len(baseline) == len(datasets) * 4
    assert not baseline.duplicated(['dataset', 'seed', 'generator']).any()
    matched = primary.merge(baseline[['dataset', 'seed', 'generator', 'value', 'record']],
                            on=['dataset', 'seed', 'generator'], validate='many_to_one',
                            suffixes=('', '_generated'))
    assert matched.value_generated.notna().all()
    matched.to_csv(OUT / f'{arm}_{metric}_matched_scores.csv', index=False)
    orientation = -1 if metric == 'nmae_sigma' else 1
    rows = []
    seed_rows = []
    for dataset in datasets:
        task = primary[primary.dataset.eq(dataset)]
        generated = baseline[baseline.dataset.eq(dataset)]
        for contrast in CONTRASTS:
            left = task if contrast == CONTRASTS[0] else task[task.has_y.eq(1)]
            right = generated if contrast == CONTRASTS[0] else task[task.has_y.eq(0)]
            for seed in [42, 43]:
                first = left[left.seed.eq(seed)].value.mean()
                second = right[right.seed.eq(seed)].value.mean()
                seed_rows.append(dict(training=arm, metric=metric, dataset=dataset, seed=seed,
                                      contrast=contrast, left_mean=first, right_mean=second,
                                      native_difference=first-second, benefit=orientation*(first-second)))
            first, second = left.value.mean(), right.value.mean()
            rows.append(dict(training=arm, metric=metric, dataset=dataset, contrast=contrast,
                             left_mean=first, right_mean=second, native_difference=first-second,
                             benefit=orientation*(first-second), orientation=orientation))
    return primary, pd.DataFrame(rows), pd.DataFrame(seed_rows)

def infer(values):
    x = np.asarray(values, dtype=float)
    assert len(x) == 5 and np.isfinite(x).all()
    assert np.all(x != 0) and len(np.unique(np.abs(x))) == 5
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert se > 0
    t = mean / se
    raw_p = 2 * stats.t.sf(abs(t), 4)
    np.testing.assert_allclose(raw_p, stats.ttest_1samp(x, 0).pvalue, atol=1e-12)
    signs = np.asarray(list(itertools.product([-1, 1], repeat=5)))
    null_means = (signs * x).mean(axis=1)
    exact_p = np.mean(np.abs(null_means) >= abs(mean) - 1e-12)
    ranks = stats.rankdata(np.abs(x))
    rank_observed = abs(np.sum(np.sign(x) * ranks))
    rank_p = np.mean(np.abs(signs @ ranks) >= rank_observed - 1e-12)
    np.testing.assert_allclose(rank_p, stats.wilcoxon(x, method='exact').pvalue, atol=1e-12)
    result = dict(n_datasets=5, df=4, mean=mean, se=se, t=t, p_raw=raw_p,
                  p_signflip_exact=exact_p, p_wilcoxon_exact=rank_p,
                  wins=int((x > 0).sum()), losses=int((x < 0).sum()))
    for family_size in [1, 2, 4]:
        margin = stats.t.ppf(1 - .05 / (2 * family_size), 4) * se
        result[f'ci{family_size}_low'] = mean - margin
        result[f'ci{family_size}_high'] = mean + margin
    return result

anova_rows, model_rows, f_checks, joint_rows = [], [], [], []
G, H, L, S, D = [f'C({factor}, Sum)' for factor in ['generator', 'has_y', 'labeler', 'seed', 'dataset']]

def fit_anova(frame, outcome, arm, scope, rhs, rank_expected):
    data = frame.copy()
    data[outcome] = data.value
    formula = f'{outcome} ~ {rhs}'
    fit = smf.ols(formula, data).fit()
    design = dmatrix(rhs, data, return_type='dataframe')
    rank = int(np.linalg.matrix_rank(design))
    assert rank == design.shape[1] == rank_expected
    a = anova_lm(fit, typ=2).rename_axis('term').reset_index()
    for row in a.itertuples(index=False):
        if row.term == 'Residual':
            continue
        keep = np.ones(design.shape[1], dtype=bool)
        keep[design.design_info.term_name_slices[row.term]] = False
        x, y = design.to_numpy()[:, keep], data.value.to_numpy()
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        independent_f = ((residual @ residual - fit.ssr) / row.df) / (fit.ssr / fit.df_resid)
        np.testing.assert_allclose(independent_f, row.F, rtol=2e-7, atol=1e-8)
        f_checks.append(dict(training=arm, outcome=outcome, scope=scope,
                             term=row.term, anova_F=row.F, restricted_OLS_F=independent_f))
    a['training'], a['outcome'], a['scope'] = arm, outcome, scope
    anova_rows.append(a)
    model_rows.append(dict(training=arm, outcome=outcome, scope=scope, formula=formula,
                           n=int(fit.nobs), rank=rank, residual_df=int(fit.df_resid),
                           inference='exploratory nominal; independent homoskedastic residual assumption'))
    return fit

class_tests, all_task_effects, all_seed_effects = [], [], []
for arm in ['synthetic', 'mix']:
    primary, effects, seed_effects = contrasts(raw[raw.metric.eq('f1_macro')], CLASS, arm, 'f1_macro')
    all_task_effects.append(effects)
    all_seed_effects.append(seed_effects)
    for contrast in CONTRASTS:
        rows = effects[effects.contrast.eq(contrast)]
        result = dict(training=arm, contrast=contrast, left_mean=rows.left_mean.mean(),
                      right_mean=rows.right_mean.mean(), **infer(rows.benefit))
        class_tests.append(result)
    factorial = fit_anova(primary, 'macro_F1', arm, 'factorial', f'{D} + {G}*{H}*{L}*{S}', 28)
    block = fit_anova(primary, 'macro_F1', arm, 'seed_block', f'{D} + {S} + {G}*{H}*{L}', 17)
    task_fit = fit_anova(primary, 'macro_F1', arm, 'dataset_interaction',
                         f'{D}*({H}*{L}) + {G}*{H}*{L} + {S}', 37)
    for name, expanded in [('seed_interactions', factorial), ('dataset_interactions', task_fit)]:
        row = anova_lm(block, expanded).iloc[1]
        joint_rows.append(dict(training=arm, test=name, df_num=row.df_diff,
                               df_den=row.df_resid, F=row.F, p_nominal=row['Pr(>F)']))
    for metric in ['r2', 'nmae_sigma', 'd2_absolute_error']:
        frame = d2 if metric == 'd2_absolute_error' else raw[raw.metric.eq(metric)]
        reg_primary, effects, seed_effects = contrasts(frame, REG, arm, metric)
        all_task_effects.append(effects)
        all_seed_effects.append(seed_effects)
        if metric == 'nmae_sigma':
            continue
        outcome = 'R2' if metric == 'r2' else 'D2'
        fit_anova(reg_primary, outcome, arm, 'pooled', f'{D} + {G}*{H}*{L}*{S}', 25)
        for dataset in REG:
            fit_anova(reg_primary[reg_primary.dataset.eq(dataset)], outcome, arm, dataset,
                       f'{S} + {G}*{H}*{L}', 13)

tests = pd.DataFrame(class_tests)
for column in ['p_raw', 'p_signflip_exact', 'p_wilcoxon_exact']:
    tests[column + '_holm4'] = multipletests(tests[column], method='holm')[1]
    tests[column + '_holm2'] = np.nan
    for arm in ['synthetic', 'mix']:
        mask = tests.training.eq(arm)
        assert mask.sum() == 2
        tests.loc[mask, column + '_holm2'] = multipletests(tests.loc[mask, column], method='holm')[1]
# Independently implement Holm, including monotonicity of adjusted p-values.
def holm(values):
    p = np.asarray(values)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted
for column in ['p_raw', 'p_signflip_exact', 'p_wilcoxon_exact']:
    np.testing.assert_allclose(tests[column + '_holm4'], holm(tests[column]), atol=1e-12)
    for arm in ['synthetic', 'mix']:
        rows = tests[tests.training.eq(arm)]
        np.testing.assert_allclose(rows[column + '_holm2'], holm(rows[column]), atol=1e-12)
save(tests, 'classification_contrasts.csv')
save(pd.concat(all_task_effects, ignore_index=True), 'task_contrasts.csv')
save(pd.concat(all_seed_effects, ignore_index=True), 'seed_contrasts.csv')
save(pd.concat(anova_rows, ignore_index=True), 'exploratory_anova.csv')
save(pd.DataFrame(model_rows), 'model_summary.csv')
save(pd.DataFrame(f_checks), 'independent_F_checks.csv')
save(pd.DataFrame(joint_rows), 'classification_joint_checks.csv')
assert len(model_rows) == 18 and len(f_checks) == 202
metadata = dict(
    checked_at_chicago=verification['checked_at_chicago'], selected_records=406,
    d2_sidecars=116, datasets=CLASS + REG, labelers=LABELS, generators=['ctgan', 'tvae'],
    seeds=[42, 43], contrasts=CONTRASTS, classification_metric='f1_macro',
    regression_metrics=['r2', 'nmae_sigma', 'd2_absolute_error'],
    classification_inference='Exploratory two-sided paired t across five task means (df=4), with exact sensitivity checks',
    primary_family='Two classification contrasts within each training regime; Holm2',
    joint_claim_family='All four classification contrasts across both training regimes; Holm4',
    intervals='ci1: pointwise 95%; ci2/ci4: Bonferroni simultaneous 95% for stated families, not Holm intervals',
    regression_inference='Native-unit descriptive contrasts separately for News and Housing; nominal factorial ANOVAs exploratory',
    scope_change='Two questions selected after previous analyses; not an untouched preregistered confirmatory test',
    limitations=['Five selected classification tasks; MNIST12/28 share a task family and split',
                 'Two fixed experiment seeds and shared heldout samples; task-generalization assumptions are uncertain',
                 'MNIST12 TVAE synthetic tables retain documented holdout-feature collisions',
                 'R2, D2, NMAE and macro F1 are distinct outcomes; no seven-task pooled score'],
    source_hashes={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                   for name in ['scores.csv', 'regression_d2_scores.csv', 'source_snapshot.json', 'd2_verification.json', 'analyze.py']},
    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     scipy=scipy.__version__, statsmodels=statsmodels.__version__),
    verified_source_values=verified_values, independent_F_checks=len(f_checks),
    numerical_thread_limit=1, experiment_changes=False)
(OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))
print(tests[['training', 'contrast', 'mean', 'p_raw', 'p_raw_holm2', 'p_raw_holm4', 'p_signflip_exact']].to_string(index=False))
print(json.dumps(dict(source_files=522, source_values=verified_values, models=18, independent_F_checks=202)))
