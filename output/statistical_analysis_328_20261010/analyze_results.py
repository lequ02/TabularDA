"""Analyze frozen completed scores remotely. No model fitting or experiment launch."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

LABELERS = ['gaussian', 'categorical', 'pca_gmm', 'rf', 'xgb', 'dnn']
STRONG = ['rf', 'xgb', 'dnn']
CORE = ['adult', 'census_kdd', 'covertype', 'mnist28']
CONTRASTS = ['full_minus_xonly', 'full_hybrid_minus_generated', 'xonly_hybrid_minus_generated']


def parse_method(method):
    if method is None or method == 'original':
        return 'real', 'real', 'original'
    if method in ['ctgan', 'tvae']:
        return method, 'full', 'generated'
    generator = 'tvae' if method.startswith('tvae_') else 'ctgan'
    label = method.removeprefix('tvae_')
    source = 'full' if label.startswith('compare_') else 'xonly'
    return generator, source, label.removeprefix('compare_')


def prepare(snapshot):
    rows = []
    for r in snapshot['real']:
        g, source, labeler = parse_method(r['method'])
        assert r['predictions_present'] and r['weight_present'], r['path']
        assert r['selected_epoch'] is not None
        if g != 'real':
            assert r['generator_epochs'] == 500 and r['generator_batch_size'] == 500, r['path']
            assert r['synthetic_rows'] == 100000 and r['generator_cuda'], r['path']
        if r['dataset'] == 'news':
            norm = r['target_normalization']
            assert norm['split'] == 'test' and norm['ddof'] == 0
            assert np.isclose(r['metrics']['nmae_sigma'], r['metrics']['mae'] / norm['sigma_y'])
        protocol = ('census_weighted' if r['namespace'].startswith('census_kdd_weighted') else 'corrected')
        if protocol == 'census_weighted':
            assert r['selection_metric'] == 'f1_macro'
        for metric, value in r['metrics'].items():
            assert np.isfinite(value), (r['path'], metric)
            rows.append(dict(domain='real', dataset=r['dataset'], seed=r['seed'], generator=g,
                             source=source, labeler=labeler, mode=r['mode'], metric=metric,
                             value=float(value), protocol=protocol, record=r['path'],
                             record_sha256=r['record_sha256'], test_hash=r['test_table_sha256']))
    for r in snapshot['simulated']:
        method = r['method']
        if method in ['original', 'ctgan', 'tvae']:
            g, source, labeler = parse_method(method)
        else:
            g, source, labeler = method.split('-', 2)
            source = {'joint': 'full', 'features': 'xonly'}[source]
        for metric in ['accuracy', 'macro_f1', 'h_star_agreement', 'l_syn', 'l_test', 'support_violation_rate']:
            value = float(r[metric]); assert np.isfinite(value)
            rows.append(dict(domain='simulated', dataset=r['dataset'], seed=int(r['seed']),
                             generator=g, source=source, labeler=labeler, mode='synthetic',
                             metric=metric, value=value, protocol='simulated_300_epochs',
                             record=method, record_sha256=snapshot['simulated_csv_sha256'], test_hash=''))
    df = pd.DataFrame(rows)
    keys = ['domain', 'dataset', 'seed', 'generator', 'source', 'labeler', 'mode', 'metric', 'protocol']
    assert not df.duplicated(keys).any()
    return df


def select_real(df, datasets, old_census=False):
    mask = (df.domain == 'real') & df.dataset.isin(datasets)
    census_protocol = 'corrected' if old_census else 'census_weighted'
    mask &= (df.dataset != 'census_kdd') | (df.protocol == census_protocol)
    return df.loc[mask].copy()


def make_pairs(df, metric, labelers, common_triplets=True):
    df = df[df.metric == metric].copy()
    orient = -1 if metric in ['nmae_sigma', 'support_violation_rate'] else 1
    scale = 100 if metric in ['f1_macro', 'macro_f1', 'accuracy', 'h_star_agreement', 'support_violation_rate'] else 1
    df['score'] = df.value * orient * scale
    join = ['domain', 'dataset', 'seed', 'generator', 'mode', 'protocol']
    hybrid = df[df.labeler.isin(labelers)]
    full = hybrid[hybrid.source == 'full']
    xonly = hybrid[hybrid.source == 'xonly']
    generated = df[df.labeler == 'generated']
    if common_triplets:
        common = full[join+['labeler']].merge(xonly[join+['labeler']],on=join+['labeler'],validate='one_to_one')
        common = common.merge(generated[join],on=join,validate='many_to_one')
        full = full.merge(common,on=join+['labeler'],validate='one_to_one')
        xonly = xonly.merge(common,on=join+['labeler'],validate='one_to_one')
    result = []
    for contrast, left, right, keys in [
        (CONTRASTS[0], full, xonly, join + ['labeler']),
        (CONTRASTS[1], full, generated, join),
        (CONTRASTS[2], xonly, generated, join),
    ]:
        paired = left.merge(right, on=keys, suffixes=('_left', '_right'), validate='one_to_one' if 'labeler' in keys else 'many_to_one')
        for _, r in paired.iterrows():
            label = r['labeler'] if 'labeler' in keys else r['labeler_left']
            result.append({**{k: r[k] for k in join}, 'labeler': label, 'metric': metric,
                           'contrast': contrast, 'difference': r.score_left - r.score_right,
                           'left_score': r.score_left, 'right_score': r.score_right,
                           'left_record': r.record_left, 'right_record': r.record_right,
                           'units': 'percentage_points' if scale == 100 else 'native_difference'})
    return pd.DataFrame(result)


def collapse_pairs(pairs):
    # Each labeler, seed, mode, generator and dataset gets equal weight in turn.
    dims = ['domain', 'dataset', 'contrast', 'metric']
    cells = pairs.groupby(dims + ['generator', 'mode', 'seed'], as_index=False).agg(difference=('difference', 'mean'), paired_labelers=('labeler', 'nunique'))
    seeds = cells.groupby(dims + ['generator', 'mode'], as_index=False).agg(difference=('difference', 'mean'), seeds=('seed', 'nunique'))
    modes = seeds.groupby(dims + ['generator'], as_index=False).agg(difference=('difference', 'mean'), modes=('mode', 'nunique'))
    datasets = modes.groupby(dims, as_index=False).agg(difference=('difference', 'mean'), generators=('generator', 'nunique'))
    counts = pairs.groupby(dims).size().reset_index(name='matched_cells')
    return datasets.merge(counts, on=dims)


def inference(values):
    x = np.asarray(values, dtype=float); n = len(x)
    result = dict(n_datasets=n, mean=float(x.mean()), median=float(np.median(x)),
                  wins=int((x > 1e-10).sum()), losses=int((x < -1e-10).sum()), ties=int((abs(x) <= 1e-10).sum()))
    if n < 2:
        return {**result, 'ci_low': None, 'ci_high': None, 't': None, 'p_t': None,
                'p_exact': None, 'p_wilcoxon': None, 'cohen_dz': None}
    sd = x.std(ddof=1)
    if sd == 0:
        result.update(ci_low=float(x.mean()), ci_high=float(x.mean()), t=None, p_t=None, cohen_dz=None)
    else:
        se = sd / math.sqrt(n); margin = stats.t.ppf(.975, n-1) * se
        t = x.mean() / se
        result.update(ci_low=float(x.mean()-margin), ci_high=float(x.mean()+margin),
                      t=float(t), p_t=float(2*stats.t.sf(abs(t), n-1)), cohen_dz=float(x.mean()/sd))
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    permutations = abs((signs*x).mean(axis=1))
    result['p_exact'] = float(np.mean(permutations >= abs(x.mean()) - 1e-12))
    result['p_wilcoxon'] = (1.0 if np.all(abs(x) < 1e-12) else float(stats.wilcoxon(x, method='approx', zero_method='wilcox').pvalue))
    return result


def run_contrasts(df, name, metric, groups, all_pairs, all_effects, common_triplets=True):
    summaries = []
    for group, labelers in groups.items():
        pairs = make_pairs(df, metric, labelers, common_triplets=common_triplets)
        if pairs.empty:
            continue
        effects = collapse_pairs(pairs)
        pairs['analysis'] = name; pairs['labeler_group'] = group
        effects['analysis'] = name; effects['labeler_group'] = group
        all_pairs.append(pairs); all_effects.append(effects)
        for contrast, block in effects.groupby('contrast', sort=False):
            summaries.append(dict(analysis=name, metric=metric, labeler_group=group, contrast=contrast,
                                  matched_cells=int(block.matched_cells.sum()), **inference(block.difference)))
    return summaries


def repeated_anova(df, factors, name):
    # Complete dataset-seed blocks only, followed by averaging seeds within dataset.
    levels = {k: sorted(df[k].unique()) for k in factors}
    expected = math.prod(len(levels[k]) for k in factors)
    assert not df.duplicated(['dataset', 'seed'] + factors).any()
    block_counts = df.groupby(['dataset', 'seed']).size()
    complete = set(block_counts[block_counts == expected].index)
    df = df[[tuple(x) in complete for x in df[['dataset', 'seed']].to_numpy()]]
    if df.empty:
        raise ValueError(f'No complete factorial blocks: {name}')
    blocks = [{'dataset': d, 'seed': int(s)} for d, s in sorted(complete)]
    averaged = df.groupby(['dataset'] + factors, as_index=False).value.mean()
    subjects = sorted(averaged.dataset.unique())
    if len(subjects) < 2:
        raise ValueError(f'Insufficient dataset subjects: {name}')
    fit = AnovaRM(averaged, 'value', 'dataset', within=factors).fit()
    grid = list(itertools.product(*[levels[f] for f in factors]))
    arrays = []
    for subject in subjects:
        b = averaged[averaged.dataset == subject].set_index(factors)
        arrays.append([float(b.loc[cell, 'value']) for cell in grid])
    y = np.asarray(arrays)
    results = []
    for term, r in fit.anova_table.iterrows():
        active = term.split(':'); basis = np.array([[1.]])
        for f in factors:
            k = len(levels[f]); part = helmert(k, full=False) if f in active else np.ones((1,k))/math.sqrt(k)
            basis = np.kron(basis, part)
        v = y @ basis.T; d = basis.shape[0]
        if d == 1:
            epsilon = 1.
        else:
            cov = np.cov(v, rowvar=False, ddof=1)
            epsilon = float(np.trace(cov)**2 / (d*np.trace(cov @ cov)))
            assert 1/d - 1e-9 <= epsilon <= 1 + 1e-9
        num, den = float(r['Num DF']), float(r['Den DF']); fvalue = float(r['F Value'])
        results.append(dict(analysis=name, term=term, F=fvalue, df_num=num, df_den=den,
                            p_uncorrected=float(r['Pr > F']), epsilon_GG=epsilon,
                            p_GG=float(stats.f.sf(fvalue, num*epsilon, den*epsilon)),
                            partial_eta_squared=float(fvalue*num/(fvalue*num+den)),
                            n_datasets=len(subjects), n_complete_seed_blocks=len(complete), subjects=';'.join(subjects)))
    marginal = []
    for factor in factors:
        for level, b in averaged.groupby(factor):
            marginal.append(dict(analysis=name, factor=factor, level=level, mean=float(b.value.mean())))
    return results, blocks, marginal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--snapshot', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    snapshot_path = Path(args.snapshot); snapshot = json.loads(snapshot_path.read_text()); df = prepare(snapshot)
    df.to_csv(out/'all_scores.csv', index=False)
    selected = select_real(df, CORE)
    sim = df[df.domain == 'simulated']
    groups = {'all_six': LABELERS, 'rf_xgb_dnn': STRONG, **{k: [k] for k in LABELERS}}
    pairs, effects, summary = [], [], []
    scenarios = [('real_core','f1_macro', selected), ('simulated','accuracy', sim)]
    for name, metric, sub in scenarios:
        summary.extend(run_contrasts(sub, name, metric, groups, pairs, effects))
        for generator in ['ctgan','tvae']:
            summary.extend(run_contrasts(sub[sub.generator == generator], name+'_'+generator, metric,
                                         {'all_six':LABELERS, 'rf_xgb_dnn':STRONG, 'dnn':['dnn']}, pairs, effects))
        for mode in sorted(sub['mode'].unique()):
            if mode != 'original':
                summary.extend(run_contrasts(sub[sub['mode'] == mode], name+'_'+mode, metric,
                                             {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']}, pairs, effects))
    summary.extend(run_contrasts(selected,'real_available_pairs','f1_macro',
                                 {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']},pairs,effects,common_triplets=False))
    sensitivities = [
        ('real_without_census', select_real(df, ['adult','covertype','mnist28'])),
        ('real_with_credit', select_real(df, CORE+['credit'])),
        ('real_mnist12_instead', select_real(df, ['adult','census_kdd','covertype','mnist12'])),
        ('real_unweighted_census', select_real(df, CORE, old_census=True)),
        ('real_seed42_only', selected[selected.seed == 42]),
    ]
    for name, sub in sensitivities:
        summary.extend(run_contrasts(sub, name, 'f1_macro', {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']}, pairs, effects))
    for metric in ['macro_f1','h_star_agreement','l_syn','l_test','support_violation_rate']:
        summary.extend(run_contrasts(sim, 'simulated_secondary', metric,
                                     {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']}, pairs, effects))
        if metric in ['l_syn','l_test','support_violation_rate']:
            for family, datasets in {'mixed':['gaussian','grid','ring'],'bn':['asia','alarm','child','insurance']}.items():
                if metric == 'support_violation_rate' and family == 'mixed':
                    continue
                summary.extend(run_contrasts(sim[sim.dataset.isin(datasets)],'simulated_'+family,metric,
                                             {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']},pairs,effects))
    news = select_real(df,['news'])
    for metric in ['r2','nmae_sigma']:
        summary.extend(run_contrasts(news, 'news', metric,
                                     {'all_regression':['pca_gmm','rf','xgb','dnn'],**{k:[k] for k in ['pca_gmm','rf','xgb','dnn']}}, pairs, effects))
    sums = pd.DataFrame(summary)
    sums['p_exact_holm'] = np.nan; sums['p_t_holm'] = np.nan
    for name in ['real_core','simulated']:
        mask = (sums.analysis == name) & (sums.labeler_group == 'all_six')
        assert mask.sum() == 3
        for col in ['p_exact','p_t']:
            sums.loc[mask,col+'_holm'] = multipletests(sums.loc[mask,col], method='holm')[1]
    sums['p_exact_exploratory_holm'] = np.nan
    sums['p_exact_focused_holm'] = np.nan
    sums['p_exact_global12_holm'] = np.nan
    global_mask=sums.analysis.isin(['real_core','simulated']) & sums.labeler_group.isin(['all_six','rf_xgb_dnn'])
    assert global_mask.sum()==12
    sums.loc[global_mask,'p_exact_global12_holm']=multipletests(sums.loc[global_mask,'p_exact'],method='holm')[1]
    for name in ['real_core','simulated']:
        mask = (sums.analysis == name) & (sums.labeler_group == 'rf_xgb_dnn')
        sums.loc[mask,'p_exact_focused_holm'] = multipletests(sums.loc[mask,'p_exact'],method='holm')[1]
        mask = (sums.analysis == name) & (sums.labeler_group.isin(LABELERS))
        sums.loc[mask,'p_exact_exploratory_holm'] = multipletests(sums.loc[mask,'p_exact'], method='holm')[1]
    sums.to_csv(out/'contrast_tests.csv',index=False)
    pairdf = pd.concat(pairs,ignore_index=True); effectdf = pd.concat(effects,ignore_index=True)
    for (name,metric,group,dataset),b in effectdf.groupby(['analysis','metric','labeler_group','dataset']):
        if name == 'real_available_pairs':
            continue
        d=b.set_index('contrast').difference
        if len(d)==3:
            assert np.isclose(d[CONTRASTS[0]],d[CONTRASTS[1]]-d[CONTRASTS[2]],atol=1e-10)
    pairdf.to_csv(out/'paired_differences.csv',index=False); effectdf.to_csv(out/'dataset_effects.csv',index=False)
    anovas, blocks, marginals = [], {}, []
    for name, sub, metric, factors in [
        ('real_hybrid_factorial',selected,'f1_macro',['generator','source','labeler','mode']),
        ('simulated_hybrid_factorial',sim,'accuracy',['generator','source','labeler']),
    ]:
        hybrid = sub[(sub.metric == metric)&sub.labeler.isin(LABELERS)&sub.generator.isin(['ctgan','tvae'])].copy()
        hybrid.value *= 100
        a,b,m = repeated_anova(hybrid,factors,name); anovas.extend(a);blocks[name]=b;marginals.extend(m)
    for name, sub, metric, factors in [
        ('real_construction_factorial',selected,'f1_macro',['generator','construction','mode']),
        ('simulated_construction_factorial',sim,'accuracy',['generator','construction']),
    ]:
        main = sub[(sub.metric==metric)&sub.generator.isin(['ctgan','tvae'])].copy()
        main['construction'] = np.where(main.labeler=='generated','generated',np.where(main.source=='full','hybrid_full','hybrid_xonly'))
        main = main[main.labeler.isin(LABELERS+['generated'])]
        keys = ['dataset','seed','generator','mode','construction']
        ag = main.groupby(keys,as_index=False).agg(value=('value','mean'), n=('labeler','nunique'))
        # A hybrid construction requires all six labelers; baseline occurs exactly once.
        ag = ag[(ag.construction=='generated')|(ag.n==6)].copy();ag.value*=100
        a,b,m = repeated_anova(ag,factors,name);anovas.extend(a);blocks[name]=b;marginals.extend(m)
    anova = pd.DataFrame(anovas); anova['p_GG_holm']=np.nan
    for name, index in anova.groupby('analysis').groups.items():
        anova.loc[index,'p_GG_holm']=multipletests(anova.loc[index,'p_GG'],method='holm')[1]
    anova.to_csv(out/'anova.csv',index=False);pd.DataFrame(marginals).to_csv(out/'factor_marginals.csv',index=False)
    # Labeler quality grouping was specified before inspecting scores: flexible
    # discriminative RF/XGB/DNN versus GaussianNB/CategoricalNB/PCA-GMM.
    group_anovas=[]
    for name,sub,metric,factors in [
        ('real_labeler_group_factorial',selected,'f1_macro',['generator','source','labeler_group','mode']),
        ('simulated_labeler_group_factorial',sim,'accuracy',['generator','source','labeler_group']),
    ]:
        b=sub[(sub.metric==metric)&sub.labeler.isin(LABELERS)].copy()
        b['labeler_group']=np.where(b.labeler.isin(STRONG),'rf_xgb_dnn','nb_pca_gmm')
        ag=b.groupby(['dataset','seed']+factors,as_index=False).agg(value=('value','mean'),n=('labeler','nunique'))
        ag=ag[ag.n==3].copy();ag.value*=100
        a,bb,m=repeated_anova(ag,factors,name);group_anovas.extend(a);blocks[name]=bb;marginals.extend(m)
    ga=pd.DataFrame(group_anovas);ga['p_GG_holm']=np.nan
    for name,index in ga.groupby('analysis').groups.items():
        ga.loc[index,'p_GG_holm']=multipletests(ga.loc[index,'p_GG'],method='holm')[1]
    ga.to_csv(out/'labeler_group_anova.csv',index=False)
    pd.DataFrame(marginals).to_csv(out/'factor_marginals.csv',index=False)
    # Three mixed oracles share their target mechanism: a conservative family sensitivity.
    family = effectdf[(effectdf.analysis=='simulated')].copy()
    family['family']=family.dataset.replace({'gaussian':'mixed_family','grid':'mixed_family','ring':'mixed_family'})
    fam = family.groupby(['labeler_group','contrast','metric','family'],as_index=False).difference.mean()
    family_tests=[]
    for keys,b in fam.groupby(['labeler_group','contrast','metric']):
        family_tests.append(dict(labeler_group=keys[0],contrast=keys[1],metric=keys[2],**inference(b.difference)))
    pd.DataFrame(family_tests).to_csv(out/'simulated_family_sensitivity.csv',index=False)
    # Equivalence margins are a declared sensitivity grid, not a retroactive
    # claim that the study was designed to establish equivalence at these margins.
    equivalence=[]
    for name in ['real_core','simulated']:
        for group in ['all_six','rf_xgb_dnn','dnn']:
            b=effectdf[(effectdf.analysis==name)&(effectdf.labeler_group==group)&(effectdf.contrast==CONTRASTS[0])]
            x=b.difference.to_numpy();n=len(x);mean=x.mean();se=x.std(ddof=1)/np.sqrt(n)
            ci=stats.t.ppf(.95,n-1)*se
            for margin in [.5,1.,2.]:
                p_lower=stats.t.sf((mean+margin)/se,n-1);p_upper=stats.t.cdf((mean-margin)/se,n-1)
                equivalence.append(dict(analysis=name,labeler_group=group,margin_pp=margin,mean=mean,
                                        ci90_low=mean-ci,ci90_high=mean+ci,p_TOST=max(p_lower,p_upper),n_datasets=n))
    pd.DataFrame(equivalence).to_csv(out/'equivalence_sensitivity.csv',index=False)
    # Generator and mixing contrasts use only complete factorial seed blocks.
    factor_effects=[]
    for name,sub,metric,fac in [('real_core',selected,'f1_macro','generator'),('simulated',sim,'accuracy','generator'),
                              ('real_core',selected,'f1_macro','mode'),('real_core',selected,'f1_macro','seed'),
                              ('simulated',sim,'accuracy','seed')]:
        blockname='real_hybrid_factorial' if name=='real_core' else 'simulated_hybrid_factorial'
        complete={(b['dataset'],b['seed']) for b in blocks[blockname]}
        sub=sub[[tuple(x) in complete for x in sub[['dataset','seed']].to_numpy()]]
        for group,labels in {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']}.items():
            b=sub[(sub.metric==metric)&sub.labeler.isin(labels)].copy();b.value*=100
            dims=['dataset','source','labeler']+[k for k in ['seed','mode','generator'] if k!=fac]
            table=b.pivot(index=dims,columns=fac,values='value').dropna()
            left,right=('tvae','ctgan') if fac=='generator' else ('mix','synthetic') if fac=='mode' else (43,42)
            table['difference']=table[left]-table[right]
            vals=table.groupby('dataset').difference.mean()
            factor_effects.append(dict(analysis=name,labeler_group=group,contrast=str(left)+'_minus_'+str(right),**inference(vals)))
    pd.DataFrame(factor_effects).to_csv(out/'other_factor_contrasts.csv',index=False)
    real_reference=[]
    for name,metric,sub in scenarios:
        for group,labels in {'all_six':LABELERS,'rf_xgb_dnn':STRONG,'dnn':['dnn']}.items():
            common=make_pairs(sub,metric,labels)
            reference=sub[(sub.metric==metric)&(sub.generator=='real')]
            keys=['domain','dataset','seed','protocol']
            ref=reference[keys+['value']].rename(columns={'value':'real_score'})
            assert not ref.duplicated(keys).any()
            for contrast, column in [(CONTRASTS[1],'left_score'),(CONTRASTS[2],'left_score')]:
                b=common[common.contrast==contrast].merge(ref,on=keys,validate='many_to_one')
                b['difference']=b[column]-b.real_score*100
                values=collapse_pairs(b).difference
                real_reference.append(dict(analysis=name,labeler_group=group,
                                           contrast=contrast.replace('_generated','_real_only'),**inference(values)))
    pd.DataFrame(real_reference).to_csv(out/'real_only_reference.csv',index=False)
    inventory=[]
    for r in snapshot['real']:
        reason = ('regression separate' if r['dataset']=='news' else 'rare positive holdout sensitivity' if r['dataset']=='credit' else
                  'no completed generator comparisons' if r['dataset']=='intrusion' else 'same MNIST family sensitivity' if r['dataset']=='mnist12' else
                  'different Census evaluation protocol sensitivity' if r['dataset']=='census_kdd' and r['namespace']=='corrected_v2' else 'core analysis')
        inventory.append(dict(record=r['path'],dataset=r['dataset'],seed=r['seed'],mode=r['mode'],method=r['method'],scope=reason))
    pd.DataFrame(inventory).to_csv(out/'record_inventory.csv',index=False)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')}
    metadata = dict(snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),snapshot_time=snapshot['created_chicago'],
                    packages=snapshot['packages'],real_records=len(snapshot['real']),simulated_rows=len(snapshot['simulated']),
                    source_checks='all real referenced predictions/checkpoints present; recorded production budgets and News normalization checked',
                    primary_real_datasets=CORE,anova_complete_blocks=blocks,table_sha256=hashes,
                    inference='Dataset means, exact two-sided sign-flip and paired-t sensitivity. Common complete triplets. Holm across three all-labeler contrasts and three exploratory focused RF/XGB/DNN contrasts separately per domain; individual labeler tests form an 18-test exploratory family.',
                    positive_difference='full minus xonly; hybrid minus generated. NMAE and support violation differences orient lower as better.')
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('PRIMARY\n'+sums[(sums.analysis.isin(['real_core','simulated']))&(sums.labeler_group.isin(['all_six','rf_xgb_dnn','dnn']))].to_string(index=False))
    print('ANOVA\n'+anova.to_string(index=False))
    print('DATASET_EFFECTS\n'+effectdf[(effectdf.analysis.isin(['real_core','simulated']))&(effectdf.labeler_group.isin(['all_six','rf_xgb_dnn','dnn']))].to_string(index=False))


if __name__ == '__main__':
    main()
