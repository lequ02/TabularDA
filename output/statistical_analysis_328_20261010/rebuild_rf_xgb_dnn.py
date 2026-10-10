"""Factorial reanalysis of existing scalar scores; execute on the research server.

Dataset is a fixed between-unit factor; dataset/seed is a repeated-measures
unit. The seven observed target constructions are decomposed into approach,
hybrid labeler, target inclusion and labeler-by-inclusion contrasts. There is
one generated-target baseline, never a fabricated X-only baseline.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert, null_space
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests

from analyze_results import prepare, select_real, make_pairs, collapse_pairs, inference, CORE

LABELERS = ['rf', 'xgb', 'dnn']


CONDITIONS = [('generated', 'full')] + [(label, source) for label in LABELERS for source in ['full', 'xonly']]


def condition_bases():
    """An orthonormal partition of the seven observed construction means."""
    intercept = np.ones((1, 7)) / np.sqrt(7)
    approach = np.array([[-np.sqrt(6/7)] + [1/np.sqrt(42)] * 6])
    label = np.column_stack([np.zeros(2), np.repeat(helmert(3), 2, axis=1) / np.sqrt(2)])
    inclusion = np.array([[0] + [v/np.sqrt(6) for _ in LABELERS for v in [1, -1]]])
    interaction = np.column_stack([np.zeros(2), np.kron(helmert(3), np.array([[1, -1]])/np.sqrt(2))])
    bases = {'': intercept, 'approach': approach, 'labeler': label,
             'x_inclusion': inclusion, 'labeler:x_inclusion': interaction}
    stacked = np.vstack(list(bases.values()))
    np.testing.assert_allclose(stacked @ stacked.T, np.eye(7), atol=1e-12)
    assert stacked.shape == (7, 7)
    return bases


def complete_arrays(data, metric, modes):
    b = data[(data.metric == metric) & data.generator.isin(['ctgan', 'tvae']) & data['mode'].isin(modes)
             & data.labeler.isin(['generated'] + LABELERS)].copy()
    b['condition'] = [CONDITIONS.index((label, source)) for label, source in zip(b.labeler, b.source)]
    grid = list(itertools.product(['ctgan', 'tvae'], range(7), modes))
    expected = len(grid)
    keys = ['dataset', 'seed', 'generator', 'condition', 'mode']
    assert not b.duplicated(keys).any()
    counts = b.groupby(['dataset', 'seed']).size()
    complete = sorted(counts[counts == expected].index)
    assert complete
    arrays = []
    used = []
    for dataset, seed in complete:
        block = b[(b.dataset == dataset) & (b.seed == seed)].set_index(['generator', 'condition', 'mode'])
        assert set(block.index) == set(grid)
        arrays.append([float(block.loc[cell, 'value'])*100 for cell in grid])
        used.append(dict(dataset=dataset, seed=int(seed), rows=expected))
    return np.array(arrays), used, grid, counts


def split_plot(data, domain, metric, modes):
    y, units, grid, counts = complete_arrays(data, metric, modes)
    datasets = sorted({r['dataset'] for r in units})
    design = np.array([[float(r['dataset'] == name) for name in datasets] for r in units])
    n, d = design.shape
    residual_units = n-d
    assert residual_units > 0
    inv = np.linalg.inv(design.T @ design)
    labels = [''] + ['generator']
    gb = {'': np.ones((1, 2))/np.sqrt(2), 'generator': helmert(2)}
    cb = condition_bases()
    mb = {'': np.ones((1, len(modes)))/np.sqrt(len(modes))}
    if len(modes) > 1:
        mb['training_mode'] = helmert(len(modes))
    rows = []
    errors = []
    validations = []
    for g, c, m in itertools.product(gb, cb, mb):
        basis = np.kron(np.kron(gb[g], cb[c]), mb[m])
        term = ':'.join(t for t in [g, c, m] if t)
        z = y @ basis.T
        coefficients = inv @ design.T @ z
        residual = z-design @ coefficients
        ss_error = float(np.sum(residual**2))
        k = z.shape[1]
        df_error = residual_units*k
        assert ss_error > 0
        if k == 1:
            epsilon = 1.
        else:
            covariance = residual.T @ residual / residual_units
            epsilon = float(np.trace(covariance)**2 / (k*np.trace(covariance @ covariance)))
            assert 1/k-1e-9 <= epsilon <= 1+1e-9
        for between in [False, True]:
            if not term and not between:
                continue
            contrast = helmert(d) if between else np.ones((1, d))/d
            target = contrast @ coefficients
            variance = contrast @ inv @ contrast.T
            ss_effect = float(np.trace(target.T @ np.linalg.solve(variance, target)))
            df_effect = contrast.shape[0]*k
            f = (ss_effect/df_effect)/(ss_error/df_error)
            name = ('dataset' + (':' + term if term else '')) if between else term
            gg = epsilon if term else 1.
            rows.append(dict(domain=domain, metric=metric, term=name, sum_sq=ss_effect,
                             df_num=df_effect, df_den=df_error, F=f,
                             p_raw=float(stats.f.sf(f, df_effect, df_error)), epsilon_GG=gg,
                             df_num_GG=df_effect*gg, df_den_GG=df_error*gg,
                             p_GG=float(stats.f.sf(f, df_effect*gg, df_error*gg)),
                             partial_eta_sq=ss_effect/(ss_effect+ss_error), error_sum_sq=ss_error,
                             n_datasets=d, n_seed_units=n, residual_seed_df=residual_units,
                             subjects=';'.join(datasets)))
            restricted_design = design @ null_space(contrast)
            restricted_coefficients = np.linalg.lstsq(restricted_design, z, rcond=None)[0]
            restricted_error = z - restricted_design @ restricted_coefficients
            independent_ss = float(np.sum(restricted_error**2) - ss_error)
            np.testing.assert_allclose(independent_ss, ss_effect, rtol=1e-8, atol=1e-8)
            validations.append(dict(domain=domain, term=name, sum_sq=ss_effect,
                                    independent_restricted_OLS_sum_sq=independent_ss))
            # Independently check every one-dimensional within contrast with OLS.
            if k == 1:
                fit = OLS(z[:, 0], design).fit()
                independent = float(fit.f_test(contrast).fvalue)
                np.testing.assert_allclose(independent, f, rtol=1e-8, atol=1e-9)
                validations.append(dict(domain=domain, term=name, F=f, independent_OLS_F=independent))
        errors.append(dict(domain=domain, error_stratum=term or 'between_dataset',
                           error_sum_sq=ss_error, df_den=df_error, epsilon=epsilon))
    result = pd.DataFrame(rows)
    result['p_GG_holm'] = multipletests(result.p_GG, method='holm')[1]
    # The first projection is the unit's grand mean; reconstructing the response
    # from the full orthonormal treatment basis must retain every score.
    matrix = np.vstack([np.kron(np.kron(gb[g], cb[c]), mb[m]) for g,c,m in itertools.product(gb,cb,mb)])
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(len(grid)), atol=1e-12)
    np.testing.assert_allclose((y @ matrix.T) @ matrix, y, atol=1e-10)
    inventory = [dict(domain=domain, dataset=dataset, seed=int(seed), observed_rows=int(count),
                      required_rows=len(grid), included=(dataset,int(seed)) in {(r['dataset'],r['seed']) for r in units})
                 for (dataset,seed),count in counts.items()]
    meta = dict(domain=domain, metric=metric, datasets=datasets, units=units,
                rows=n*len(grid), methods_per_generator=7, generator_levels=['ctgan','tvae'],
                training_modes=modes, residual_seed_df=residual_units, anova_rows=len(result))
    return result, inventory, meta, validations, errors


def method_comparisons(data, domain, metric):
    pairs = make_pairs(data, metric, LABELERS)
    rows = []
    unit_rows = []
    # Compare every labeler to its own generator baseline on identical support.
    for generator in ['ctgan', 'tvae']:
        for labeler in LABELERS:
            b = pairs[(pairs.generator == generator) & (pairs.labeler == labeler)]
            for source, contrast in [('full','full_hybrid_minus_generated'),('xonly','xonly_hybrid_minus_generated')]:
                block = b[b.contrast == contrast]
                unit = block.groupby(['dataset','seed'],as_index=False)[['left_score','right_score','difference']].mean()
                means = unit.groupby('dataset')[['left_score','right_score','difference']].mean()
                assert len(means)
                rows.append(dict(domain=domain, generator=generator, labeler=labeler, source=source,
                                 baseline_mean=float(means.right_score.mean()), hybrid_mean=float(means.left_score.mean()),
                                 **inference(means.difference)))
                for dataset,r in means.iterrows():
                    unit_rows.append(dict(domain=domain,generator=generator,labeler=labeler,source=source,
                                          dataset=dataset,baseline=r.right_score,hybrid=r.left_score,gain=r.difference))
    return pd.DataFrame(rows), pd.DataFrame(unit_rows), pairs


def groups_and_sources(pairs, domain):
    rows=[]
    groups={'rf_xgb_dnn':LABELERS,**{k:[k] for k in LABELERS}}
    for generator in ['pooled','ctgan','tvae']:
        gp = pairs if generator == 'pooled' else pairs[pairs.generator == generator]
        for group, labels in groups.items():
            b = gp[gp.labeler.isin(labels)]
            for contrast,part in b.groupby('contrast'):
                values=collapse_pairs(part).difference
                rows.append(dict(domain=domain,generator=generator,labeler_group=group,contrast=contrast,**inference(values)))
            # Average both feature sources before comparing against the baseline.
            hybrid=b[b.contrast.isin(['full_hybrid_minus_generated','xonly_hybrid_minus_generated'])]
            means=hybrid.groupby(['dataset','seed','generator','mode','labeler']).difference.mean().reset_index()
            means=means.groupby(['dataset','seed','generator','mode']).difference.mean().reset_index()
            means=means.groupby(['dataset','seed','generator']).difference.mean().reset_index()
            means=means.groupby(['dataset','seed']).difference.mean().reset_index()
            means=means.groupby('dataset').difference.mean()
            rows.append(dict(domain=domain,generator=generator,labeler_group=group,contrast='hybrid_both_sources_minus_generated',**inference(means)))
    result=pd.DataFrame(rows)
    result['p_exact_holm_primary']=np.nan
    result['p_t_holm_primary']=np.nan
    mask=result.labeler_group.eq('rf_xgb_dnn') & result.generator.eq('pooled') & result.contrast.isin(['full_minus_xonly','full_hybrid_minus_generated','xonly_hybrid_minus_generated'])
    assert mask.sum()==3
    result.loc[mask,'p_exact_holm_primary']=multipletests(result.loc[mask,'p_exact'],method='holm')[1]
    result.loc[mask,'p_t_holm_primary']=multipletests(result.loc[mask,'p_t'],method='holm')[1]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    snapshot_path=Path(args.snapshot);snapshot=json.loads(snapshot_path.read_text());df=prepare(snapshot)
    df=df[df.labeler.isin(['generated','original'] + LABELERS)].copy()
    real=select_real(df,CORE);sim=df[df.domain=='simulated']
    analyses=[('real','f1_macro',real,['synthetic','mix']),('simulated','accuracy',sim,['synthetic'])]
    metadata=[]; inventories=[];validations=[];errors=[];all_methods=[];all_dataset=[];all_groups=[]
    for domain,metric,data,modes in analyses:
        table, inventory, meta, checks, strata=split_plot(data,domain,metric,modes)
        table.to_csv(out/f'anova_{domain}.csv',index=False)
        metadata.append(meta);inventories.extend(inventory);validations.extend(checks);errors.extend(strata)
        methods, datasets, pairs=method_comparisons(data,domain,metric)
        all_methods.append(methods);all_dataset.append(datasets);all_groups.append(groups_and_sources(pairs,domain))
        complete_keys = pd.DataFrame(meta['units'])[['dataset','seed']]
        balanced_data = data.merge(complete_keys,on=['dataset','seed'],validate='many_to_one')
        balanced_pairs = make_pairs(balanced_data,metric,LABELERS)
        all_groups.append(groups_and_sources(balanced_pairs,domain+'_anova_subset'))
        print(domain.upper()+' ANOVA\n'+table.to_string(index=False))
    pd.concat(all_methods).to_csv(out/'generator_labeler_comparisons.csv',index=False)
    pd.concat(all_dataset).to_csv(out/'generator_labeler_dataset_scores.csv',index=False)
    groups=pd.concat(all_groups)
    groups.to_csv(out/'matched_group_contrasts.csv',index=False)
    pd.DataFrame(inventories).to_csv(out/'anova_coverage.csv',index=False)
    pd.DataFrame(validations).to_csv(out/'independent_F_checks.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'anova_error_strata.csv',index=False)
    # Credit changes inference only in an explicitly separated sensitivity.
    credit=select_real(df,CORE+['credit'])
    table,inventory,meta,checks,strata=split_plot(credit,'real_with_credit','f1_macro',['synthetic','mix'])
    table.to_csv(out/'anova_real_credit_sensitivity.csv',index=False)
    meta['sensitivity']=True;metadata.append(meta)
    for domain,metric,data,modes in analyses:
        score=data[(data.metric==metric)&data.generator.isin(['ctgan','tvae'])].copy()
        score.to_csv(out/f'score_inventory_{domain}.csv',index=False)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}
    meta=dict(snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),snapshot_chicago=snapshot['created_chicago'],
              analysis='Type III mixed between/within ANOVA using orthogonal nested construction contrasts',
              between_factor='dataset',repeat_unit='dataset/experiment_seed',
              within_factors=['generator','approach','labeler within hybrid','x_inclusion within hybrid','training_mode (real only)'],
              x_inclusion_definition='Full (X,Y) versus X-only generator fitting; X is present in both. User-confirmed target-inclusion question.',
              approaches='One generated-target baseline versus six hybrids per generator/mode; RF/XGB/DNN times two sources.',
              labelers=LABELERS,scope_selection='User-requested restricted family; selected after inspecting earlier results. Exploratory, not prospectively preregistered.',
              assumptions='Independent experiment RNG repeats conditional on fixed datasets/holdouts; normal errors and common repeat covariance across dataset groups. GG adjusts sphericity. Real residual seed df is only 2. No test-row sampling uncertainty is estimated.',
              scope='Conditional on the tested datasets. Dataset-mean paired tests separately assess heterogeneity across tasks; seeds do not create additional datasets.',
              models=metadata,packages=snapshot['packages'],table_sha256=files,
              validation='Orthogonal bases and exact score reconstruction; independent OLS F checks for every one-dimensional within contrast and dataset interaction; no duplicate records or fabricated baseline cells.')
    (out/'factorial_metadata.json').write_text(json.dumps(meta,indent=2))
    print('GROUP COMPARISONS\n'+groups[(groups.generator=='pooled')&groups.labeler_group.eq('rf_xgb_dnn')].to_string(index=False))
    print('Complete. Tables: '+str(len(files)))


if __name__=='__main__':
    main()
