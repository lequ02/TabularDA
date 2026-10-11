"""Exploratory regression ANOVAs; execute only on the research server."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_regression_20261010')
INPUT = ROOT.parent / 'statistical_328_explicit_20261010/results'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
G, H, L, S, D = [f'C({x}, Sum)' for x in ['generator', 'has_y', 'labeler', 'seed', 'dataset']]
LABELS = ['rf', 'xgb', 'dnn']
DATASETS = ['news', 'california_housing']
CONTRASTS = [('tvae_minus_ctgan', 'generator', 'tvae', 'ctgan'),
             ('has_y1_minus_has_y0', 'has_y', 1, 0),
             ('seed43_minus_seed42', 'seed', 43, 42),
             ('rf_minus_dnn', 'labeler', 'rf', 'dnn'),
             ('xgb_minus_dnn', 'labeler', 'xgb', 'dnn')]

for arm in ['synthetic', 'mix']:
    dest = OUT / arm
    dest.mkdir(exist_ok=True)
    inputs = {'R2': pd.read_csv(INPUT / arm / 'regression_scores.csv').query('metric == "r2"'),
              'D2': pd.read_csv(INPUT / arm / 'regression_d2_scores.csv')}
    checks, models, effects, baseline_effects = [], [], [], []
    for outcome, frame in inputs.items():
        primary = frame[frame.labeler.isin(LABELS)].copy()
        primary['has_y'] = primary.has_y.astype(int)
        primary[outcome] = primary.value
        assert len(primary) == 48
        expected = set(itertools.product(DATASETS, [42, 43], ['ctgan', 'tvae'], [0, 1], LABELS))
        keys = ['dataset', 'seed', 'generator', 'has_y', 'labeler']
        assert set(primary[keys].itertuples(index=False, name=None)) == expected
        assert not primary.duplicated(keys).any()
        primary.to_csv(dest / f'{outcome}_scores.csv', index=False)
        baselines = frame[frame.labeler.eq('generated')]
        assert len(baselines) == 8 and baselines.has_y.eq(1).all()
        matched = primary.merge(baselines[['dataset', 'seed', 'generator', 'value', 'record']],
                                on=['dataset', 'seed', 'generator'], validate='many_to_one',
                                suffixes=('', '_generated'))
        matched['difference'] = matched.value - matched.value_generated
        assert len(matched) == 48 and matched.value_generated.notna().all()
        matched.to_csv(dest / f'{outcome}_matched_baselines.csv', index=False)
        for dataset in DATASETS:
            task = primary[primary.dataset.eq(dataset)]
            generated = baselines[baselines.dataset.eq(dataset)].value.mean()
            for name, subset in [('relabeled', task), ('has_y0', task[task.has_y.eq(0)]),
                                 ('has_y1', task[task.has_y.eq(1)])]:
                baseline_effects.append(dict(outcome=outcome, dataset=dataset, comparison=name,
                    generated_mean=generated, relabeled_mean=subset.value.mean(),
                    difference=subset.value.mean() - generated, inference='descriptive'))
        for scope in ['pooled'] + DATASETS:
            data = primary if scope == 'pooled' else primary[primary.dataset.eq(scope)]
            rhs = f'{D} + {G}*{H}*{L}*{S}' if scope == 'pooled' else f'{S} + {G}*{H}*{L}'
            formula = f'{outcome} ~ {rhs}'
            fit = smf.ols(formula, data).fit()
            design = dmatrix(rhs, data, return_type='dataframe')
            rank = int(np.linalg.matrix_rank(design))
            assert rank == design.shape[1] == (25 if scope == 'pooled' else 13)
            assert fit.df_resid == (23 if scope == 'pooled' else 11)
            np.testing.assert_allclose(design.to_numpy(), fit.model.exog, atol=1e-12)
            a = anova_lm(fit, typ=2).rename_axis('term').reset_index()
            a['partial_eta_sq'] = a.sum_sq / (a.sum_sq + fit.ssr)
            a.loc[a.term.eq('Residual'), 'partial_eta_sq'] = np.nan
            a.to_csv(dest / f'{outcome}_{scope}_anova.csv', index=False)
            (dest / f'{outcome}_{scope}_model.txt').write_text(fit.summary().as_text())
            for _, row in a[~a.term.eq('Residual')].iterrows():
                keep = np.ones(design.shape[1], dtype=bool)
                keep[design.design_info.term_name_slices[row.term]] = False
                x, y = design.to_numpy()[:, keep], data[outcome].to_numpy()
                residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
                independent = ((residual @ residual - fit.ssr) / row['df']) / (fit.ssr / fit.df_resid)
                np.testing.assert_allclose(independent, row.F, rtol=2e-7, atol=1e-8)
                checks.append(dict(outcome=outcome, scope=scope, term=row.term,
                                   anova_F=row.F, restricted_OLS_F=independent))
            models.append(dict(outcome=outcome, scope=scope, formula=formula, n=int(fit.nobs),
                               rank=rank, columns=design.shape[1], df_resid=fit.df_resid,
                               model_r_squared=fit.rsquared))
            grid = data[keys].copy()
            xgrid = np.asarray(build_design_matrices([design.design_info], grid)[0])
            for name, factor, first, second in CONTRASTS:
                vector = xgrid[grid[factor].eq(first)].mean(axis=0) - xgrid[grid[factor].eq(second)].mean(axis=0)
                test = fit.t_test(vector[None, :])
                low, high = test.conf_int()[0]
                observed = data[data[factor].eq(first)].value.mean() - data[data[factor].eq(second)].value.mean()
                np.testing.assert_allclose(test.effect.item(), observed, atol=1e-12)
                effects.append(dict(outcome=outcome, scope=scope, contrast=name, effect=observed,
                                    ci_low=low, ci_high=high, p=float(test.pvalue), df=fit.df_resid))
    pd.DataFrame(models).to_csv(dest / 'regression_model_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(dest / 'regression_independent_F_checks.csv', index=False)
    pd.DataFrame(effects).to_csv(dest / 'regression_factor_effects.csv', index=False)
    pd.DataFrame(baseline_effects).to_csv(dest / 'regression_baseline_effects.csv', index=False)
    # Explain the existing planned family without changing its effects or tests.
    planned = pd.read_csv(INPUT / arm / 'planned_contrasts.csv')
    ds = pd.read_csv(INPUT / arm / 'dataset_effects.csv')
    uncertainty = []
    for _, row in planned.iterrows():
        delta = ds[ds.contrast.eq(row.contrast)].difference.to_numpy()
        uncertainty.append(dict(contrast=row.contrast, n_datasets=len(delta),
            sd=float(np.std(delta, ddof=1)), se=float(np.std(delta, ddof=1)/np.sqrt(len(delta))),
            minimum=float(delta.min()), maximum=float(delta.max()), raw_p=row.p_t,
            holm_p=row.p_t_holm, family_size=len(planned),
            first_holm_cutoff=.05/len(planned), smallest_family_p=planned.p_t.min()))
    pd.DataFrame(uncertainty).to_csv(dest / 'contrast_uncertainty.csv', index=False)
    print(arm, pd.DataFrame(models).to_string(index=False))
    print(arm, 'signed factor effects:', pd.DataFrame(effects).to_string(index=False))
    print(arm, 'generated-target comparisons:', pd.DataFrame(baseline_effects).to_string(index=False))
(OUT / 'regression_analysis_metadata.json').write_text(json.dumps(dict(
    analyses=['synthetic', 'mix'], outcomes=['R2', 'D2'],
    scope='RF/XGB/DNN relabeled runs; generated targets compared separately',
    pooled_equation='outcome ~ dataset + generator * has_y * labeler * seed',
    within_dataset_equation='outcome ~ seed + generator * has_y * labeler',
    inference='Exploratory nominal Type II ANOVA and unadjusted model intervals; two tasks only',
    reason_within_dataset_seed_additive='Full four-factor model has 24 parameters for 24 scores and zero residual df',
    score_inputs='Previously verified saved R2 records and D2 sidecars; no new experiments'), indent=2))
