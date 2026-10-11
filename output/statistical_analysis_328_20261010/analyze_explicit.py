"""Explicit-factor real-data analysis. Execute only on the research server."""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from analyze_results import inference

ROOT=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_explicit_20261010')
FROZEN=ROOT.parent/'statistical_328_20261010';OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
raw=pd.read_csv(FROZEN/'scores.csv')
d2raw=pd.read_csv(ROOT/'regression_d2_scores.csv')
snapshot=json.loads((FROZEN/'snapshot.json').read_text())
assert snapshot['selected_completed']==406 and not snapshot['missing']
CLASS=['adult','census_kdd','covertype','mnist12','mnist28'];LABELS=['rf','xgb','dnn']
G='C(generator, Sum)';H='C(has_y, Sum)';L='C(labeler, Sum)';S='C(seed, Sum)';D='C(dataset, Sum)'
FORMULAS={
 'factorial':f'macro_F1 ~ {D} + {G}*{H}*{L}*{S}',
 'seed_block':f'macro_F1 ~ {D} + {S} + {G}*{H}*{L}',
 'dataset_interaction':f'macro_F1 ~ {D}*({H}*{L}) + {G}*{H}*{L} + {S}'}
assert raw.record.nunique()==406 and len(d2raw)==116

def normalize(frame):
    result=frame.copy()
    result['has_y']=result.source.map({'full':1,'xonly':0}).astype('Int64')
    return result.drop(columns=['source','mode'])
def save(frame,path):
    assert not {'condition','source','mode','group','approach_has_y'}.intersection(frame.columns)
    frame.to_csv(path,index=False)
def without_labeler(frame):return frame.drop(columns=['labeler'])
save(without_labeler(normalize(raw[raw.labeler.eq('original')])),OUT/'original_references.csv')
all_metadata={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

for arm in ['synthetic','mix']:
    dest=OUT/arm;dest.mkdir(exist_ok=True)
    selected=normalize(raw[raw.dataset.isin(CLASS)&raw.metric.eq('f1_macro')&raw['mode'].eq(arm)])
    primary=selected[selected.labeler.isin(LABELS)].copy();primary['macro_F1']=primary.value
    primary['has_y']=primary.has_y.astype(int)
    baselines=selected[selected.labeler.eq('generated')].copy()
    assert len(primary)==120 and len(baselines)==20
    actual=set(primary[['dataset','seed','generator','has_y','labeler']].itertuples(index=False,name=None))
    expected=set(itertools.product(CLASS,[42,43],['ctgan','tvae'],[0,1],LABELS))
    assert actual==expected and len(actual)==len(primary)
    assert not baselines.duplicated(['dataset','seed','generator']).any() and baselines.has_y.eq(1).all()
    save(primary,dest/'classification_scores.csv')
    save(without_labeler(baselines),dest/'generated_baselines.csv')
    fits={};designs={};model_rows=[];checks=[]
    for name,formula in FORMULAS.items():
        fit=smf.ols(formula,primary).fit();fits[name]=fit
        design=dmatrix(formula.split('~',1)[1],primary,return_type='dataframe');designs[name]=design
        assert list(design.columns)==fit.model.exog_names
        np.testing.assert_allclose(design.to_numpy(),fit.model.exog,atol=1e-12)
        rank=int(np.linalg.matrix_rank(design));assert rank==design.shape[1]
        a=anova_lm(fit,typ=2).rename_axis('term').reset_index()
        a['partial_eta_sq']=a.sum_sq/(a.sum_sq+fit.ssr)
        a.loc[a.term.eq('Residual'),'partial_eta_sq']=np.nan
        a.to_csv(dest/f'{name}_anova.csv',index=False)
        # Balanced sum coding makes distinct effect subspaces orthogonal.
        # Independently reproduce every ANOVA F by removing that effect's columns.
        for _,r in a[~a.term.eq('Residual')].iterrows():
            sl=design.design_info.term_name_slices[r.term]
            kept=np.ones(design.shape[1],dtype=bool);kept[sl]=False
            x=design.to_numpy()[:,kept];y=primary.macro_F1.to_numpy()
            residual=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
            ss=float(residual@residual-fit.ssr)
            independent=(ss/r['df'])/(fit.ssr/fit.df_resid)
            np.testing.assert_allclose(independent,r.F,rtol=2e-7,atol=1e-8)
            checks.append(dict(model=name,term=r.term,anova_F=r.F,restricted_OLS_F=independent))
        model_rows.append(dict(name=name,formula=formula,n=int(fit.nobs),rank=rank,columns=design.shape[1],
          df_model=fit.df_model,df_resid=fit.df_resid,r_squared=fit.rsquared,adjusted_r_squared=fit.rsquared_adj,
          shapiro_p=float(stats.shapiro(fit.resid).pvalue)))
        (dest/f'{name}_model.txt').write_text(fit.summary().as_text())
    assert [m['rank'] for m in model_rows]==[28,17,37]
    pd.DataFrame(model_rows).to_csv(dest/'model_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(dest/'independent_F_checks.csv',index=False)
    joint=[]
    for name,expanded in [('seed_interactions','factorial'),('dataset_interactions','dataset_interaction')]:
        comparison=anova_lm(fits['seed_block'],fits[expanded]).iloc[1]
        joint.append(dict(test=name,restricted_formula=FORMULAS['seed_block'],full_formula=FORMULAS[expanded],
          df_num=comparison.df_diff,df_den=comparison.df_resid,F=comparison.F,p=float(comparison['Pr(>F)'])))
    pd.DataFrame(joint).to_csv(dest/'joint_tests.csv',index=False)

    # Marginal predictions average only observed relabeled cells. No baseline
    # enters this factorial model, and no generated-target has_y=0 is invented.
    fit=fits['factorial'];design=designs['factorial']
    grid=pd.DataFrame(list(expected),columns=['dataset','seed','generator','has_y','labeler']).sort_values(['dataset','seed','generator','has_y','labeler'])
    grid_design=np.asarray(build_design_matrices([design.design_info],grid)[0])
    emm=[];vectors={}
    for generator,h,label in itertools.product(['ctgan','tvae'],[0,1],LABELS):
        mask=grid.generator.eq(generator)&grid.has_y.eq(h)&grid.labeler.eq(label)
        vector=grid_design[mask].mean(axis=0);estimate=float(vector@fit.params)
        test=fit.t_test(vector[None,:]);limits=test.conf_int()[0]
        observed=primary[primary.generator.eq(generator)&primary.has_y.eq(h)&primary.labeler.eq(label)].macro_F1.mean()
        np.testing.assert_allclose(estimate,observed,atol=1e-12)
        vectors[(generator,h,label)]=vector
        emm.append(dict(generator=generator,has_y=h,labeler=label,mean=estimate,ci_low=limits[0],ci_high=limits[1]))
    pd.DataFrame(emm).to_csv(dest/'estimated_means.csv',index=False)
    factorial_effects=[]
    for name,factor,first,second in [('tvae_minus_ctgan','generator','tvae','ctgan'),('has_y1_minus_has_y0','has_y',1,0),('seed43_minus_seed42','seed',43,42)]:
        left=grid_design[grid[factor].eq(first)].mean(axis=0);right=grid_design[grid[factor].eq(second)].mean(axis=0)
        test=fit.t_test((left-right)[None,:]);limits=test.conf_int()[0]
        factorial_effects.append(dict(contrast=name,first_mean=float(left@fit.params),second_mean=float(right@fit.params),
          effect=float(test.effect.item()),ci_low=limits[0],ci_high=limits[1],p=float(test.pvalue),df=fit.df_resid))
    factorial_effects=pd.DataFrame(factorial_effects)
    factorial_effects['p_holm']=multipletests(factorial_effects.p,method='holm')[1]
    factorial_effects.to_csv(dest/'factorial_effects.csv',index=False)

    # One exact matched generated baseline per dataset/seed/generator. It is
    # shared as a comparator, never duplicated as independent observations.
    paired=primary.merge(baselines[['dataset','seed','generator','value','record']],on=['dataset','seed','generator'],
      how='left',validate='many_to_one',suffixes=('','_generated'))
    assert len(paired)==120 and paired.value_generated.notna().all()
    paired['difference']=paired.macro_F1-paired.value_generated
    save(paired,dest/'matched_relabeling_scores.csv')
    ds_generated=baselines.groupby('dataset').value.mean().reindex(CLASS)
    ds_relabel=primary.groupby('dataset').macro_F1.mean().reindex(CLASS)
    ds_h={h:primary[primary.has_y.eq(h)].groupby('dataset').macro_F1.mean().reindex(CLASS) for h in [0,1]}
    contrast_means={
     'relabel_minus_generated':(ds_relabel,ds_generated),
     'has_y1_minus_has_y0':(ds_h[1],ds_h[0]),
     'has_y1_minus_generated':(ds_h[1],ds_generated),
     'has_y0_minus_generated':(ds_h[0],ds_generated)}
    planned=[];dataset_effects=[]
    for name,(left,right) in contrast_means.items():
        difference=left-right;effect=float(difference.mean())
        planned.append(dict(contrast=name,left_mean=float(left.mean()),right_mean=float(right.mean()),score_change=effect,
          percentage_points=100*effect,relative_percent=100*effect/right.mean(),**inference(difference)))
        for dataset in CLASS:dataset_effects.append(dict(contrast=name,dataset=dataset,left=left[dataset],right=right[dataset],difference=difference[dataset]))
    planned=pd.DataFrame(planned)
    planned['p_t_holm']=multipletests(planned.p_t,method='holm')[1]
    planned['p_exact_holm']=multipletests(planned.p_exact,method='holm')[1]
    planned.to_csv(dest/'planned_contrasts.csv',index=False)
    pd.DataFrame(dataset_effects).to_csv(dest/'dataset_effects.csv',index=False)
    np.testing.assert_allclose(planned.set_index('contrast').loc['has_y1_minus_has_y0','score_change'],
      factorial_effects.set_index('contrast').loc['has_y1_minus_has_y0','effect'],atol=1e-12)
    # Explicit contrast weights over the observed construction cells only.
    weights=[]
    for g,h,l in itertools.product(['ctgan','tvae'],[0,1],LABELS):
        weights.append(dict(generator=g,has_y=h,labeler=l,relabel_minus_generated=1/12,
          has_y1_minus_has_y0=(1 if h else -1)/6,has_y1_minus_generated=1/6 if h else 0,
          has_y0_minus_generated=1/6 if not h else 0))
    for g in ['ctgan','tvae']:weights.append(dict(generator=g,has_y=1,labeler='',relabel_minus_generated=-1/2,
      has_y1_minus_has_y0=0,has_y1_minus_generated=-1/2,has_y0_minus_generated=-1/2))
    pd.DataFrame(weights).to_csv(dest/'contrast_weights.csv',index=False)
    label_rows=[]
    for g,h,l in itertools.product(['ctgan','tvae'],[0,1],LABELS):
        left=primary[primary.generator.eq(g)&primary.has_y.eq(h)&primary.labeler.eq(l)].groupby('dataset').macro_F1.mean().reindex(CLASS)
        right=baselines[baselines.generator.eq(g)].groupby('dataset').value.mean().reindex(CLASS)
        label_rows.append(dict(generator=g,has_y=h,labeler=l,generated=float(right.mean()),relabeled=float(left.mean()),
          relative_percent=100*(left.mean()-right.mean())/right.mean(),**inference(left-right)))
    labels=pd.DataFrame(label_rows);labels['p_t_holm']=multipletests(labels.p_t,method='holm')[1]
    labels.to_csv(dest/'labeler_effects.csv',index=False)

    # Preserve regression score definitions and saved normalization/sidecars.
    reg=normalize(raw[raw.dataset.isin(['news','california_housing'])&raw.metric.isin(['r2','nmae_sigma'])&raw['mode'].eq(arm)])
    d2=normalize(d2raw[d2raw['mode'].eq(arm)]);d2['metric']='d2_absolute_error'
    save(reg,dest/'regression_scores.csv');save(d2,dest/'regression_d2_scores.csv')
    combined=pd.concat([reg,d2],ignore_index=True);reg_rows=[]
    for dataset,metric in itertools.product(['news','california_housing'],['r2','nmae_sigma','d2_absolute_error']):
        sub=combined[combined.dataset.eq(dataset)&combined.metric.eq(metric)]
        reference=raw[raw.dataset.eq(dataset)&raw.metric.eq(metric)&raw.labeler.eq('original')].value.mean() if metric!='d2_absolute_error' else d2raw[d2raw.dataset.eq(dataset)&d2raw.labeler.eq('original')].value.mean()
        row=dict(dataset=dataset,metric=metric,original=float(reference),generated=float(sub[sub.labeler.eq('generated')].value.mean()))
        for h in [0,1]:row[f'has_y{h}']=float(sub[sub.labeler.isin(LABELS)&sub.has_y.eq(h)].value.mean())
        reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(dest/'regression_summary.csv',index=False)

    effects=pd.DataFrame(dataset_effects);fig,axes=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
    for ax,name,title in zip(axes,list(contrast_means)[:2],['Relabeling - generated Y','has_y=1 - has_y=0']):
        b=effects[effects.contrast.eq(name)].set_index('dataset').loc[CLASS]
        ax.scatter(100*b.difference,np.arange(5),color='#237b8a',s=40);ax.axvline(0,color='#777777',lw=.8)
        ax.set_title(title);ax.set_xlabel('Macro F1 change (percentage points)')
        ax.set_yticks(np.arange(5),['Adult','Census KDD','Covertype','MNIST12','MNIST28']);ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(dest/'dataset_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.2))
    axes[0].scatter(fit.fittedvalues,fit.resid,s=13,alpha=.6);axes[0].axhline(0,color='#777777',lw=.8)
    axes[0].set(xlabel='Fitted macro F1',ylabel='Residual',title='Requested factorial model')
    stats.probplot(fit.resid,dist='norm',plot=axes[1]);fig.tight_layout();fig.savefig(dest/'model_diagnostics.png',dpi=180);plt.close(fig)
    metadata=dict(training_rows='100000 synthetic rows only' if arm=='synthetic' else 'all real training rows plus 100000 synthetic rows',
      factorial_rows=120,generated_baselines=20,dataset_levels=CLASS,generator_levels=['ctgan','tvae'],has_y_levels=[0,1],
      labeler_levels=LABELS,seed_levels=[42,43],primary_formula='macro_F1 ~ dataset + generator * has_y * labeler * seed',
      models=model_rows,joint_tests=joint,anova_type=2,coding='Categorical factors; sum-to-zero contrasts. Complete balanced observed grid.',
      families='ANOVA: nominal p; factorial marginal effects: Holm over 3; planned dataset contrasts: Holm over 4; method comparisons: Holm over 12. Separate families within each training analysis.',
      inference='ANOVA/model intervals assume independent homoskedastic residuals and are exploratory. Shared heldout rows and dataset-specific treatment effects limit those assumptions. Dataset paired inference uses 5 task means, not 120 independent tasks.',
      removed='No combined treatment factor or random-intercept mixed model. No repeated-measures appendix.',
      seed='42 and 43 are categorical fixed levels. Seed interactions compare these two runs, not a population of possible seeds.')
    (dest/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2));all_metadata[arm]=metadata
    print(arm,'ANOVA:',pd.read_csv(dest/'factorial_anova.csv')[['term','df','F','PR(>F)']].to_string(index=False))
    print(arm,'planned contrasts:',planned[['contrast','score_change','p_t_holm']].to_string(index=False))
    print(arm,'joint tests:',json.dumps(joint))
(OUT/'analysis_metadata.json').write_text(json.dumps(dict(snapshot=snapshot['checked_at_chicago'],selected_records=406,
  frozen_scores_sha256=hashlib.sha256((FROZEN/'scores.csv').read_bytes()).hexdigest(),analyses=all_metadata,
  analysis='Two independent explicit-factor analyses; original scores retained.'),indent=2))
