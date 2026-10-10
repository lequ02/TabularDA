"""328 project reanalysis of frozen real-data scores; execute remotely only."""
import hashlib
import itertools
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from patsy import build_design_matrices, dmatrix
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from analyze_results import inference
from rebuild_rf_xgb_dnn import split_plot

ROOT=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_20261010')
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'scores.csv')
snapshot=json.loads((ROOT/'snapshot.json').read_text())
CLASS=['adult','census_kdd','covertype','mnist12','mnist28']
LABELS=['rf','xgb','dnn']
primary=df[df.dataset.isin(CLASS)&df.metric.eq('f1_macro')&~df.labeler.eq('original')].copy()
assert len(primary)==280 and snapshot['selected_completed']==406 and not snapshot['missing']
primary['score']=primary.value
primary['condition']=np.where(primary.labeler.eq('generated'),'generated',primary.labeler+'_'+primary.source)
primary['group']=np.where(primary.labeler.eq('generated'),'generated','hybrid_'+primary.source)
primary.to_csv(OUT/'classification_scores.csv',index=False)
reference=df[df.labeler.eq('original')].copy()
reference.to_csv(OUT/'original_references.csv',index=False)
coverage=df.groupby(['dataset','seed']).record.nunique().reset_index(name='completed')
coverage['expected']=29;assert (coverage.completed==coverage.expected).all()
coverage.to_csv(OUT/'coverage.csv',index=False)

# Follow the template's full, blocked and dataset-interaction linear models.
formulas={
 'full_interaction':'score ~ C(dataset)*C(generator)*C(condition)*C(mode)',
 'blocked':'score ~ C(dataset) + C(generator)*C(condition)*C(mode)',
 'dataset_labeler_interaction':'score ~ C(dataset)*C(condition) + C(generator) + C(mode)'}
model_info=[];models={}
for name,formula in formulas.items():
    fit=smf.ols(formula,primary).fit();models[name]=fit
    a=anova_lm(fit,typ=2).reset_index().rename(columns={'index':'term'})
    a.to_csv(OUT/f'lm_{name}_anova.csv',index=False)
    model_info.append(dict(name=name,formula=formula,n=int(fit.nobs),df_model=float(fit.df_model),
        df_resid=float(fit.df_resid),r_squared=fit.rsquared,adjusted_r_squared=fit.rsquared_adj,
        rank=int(np.linalg.matrix_rank(fit.model.exog)),columns=fit.model.exog.shape[1],
        shapiro_p=stats.shapiro(fit.resid).pvalue,
        inference='Exploratory only: ordinary residuals do not represent independent datasets.'))
    (OUT/f'lm_{name}.txt').write_text(fit.summary().as_text())
pd.DataFrame(model_info).to_csv(OUT/'lm_model_summary.csv',index=False)

# Reuse the existing repeated-measures decomposition, preserving its checks.
anova,inventory,meta,checks,strata=split_plot(primary,'classification','f1_macro',['synthetic','mix'])
anova.to_csv(OUT/'repeated_anova.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'independent_F_checks.csv',index=False)
pd.DataFrame(strata).to_csv(OUT/'anova_error_strata.csv',index=False)

# Each original or averaged hybrid group appears once per dataset/seed/generator/mode.
grouped=primary.groupby(['dataset','seed','generator','mode','group'],as_index=False).score.mean()
grouped['group']=pd.Categorical(grouped.group,['generated','hybrid_xonly','hybrid_full'])
formula='score ~ C(group)*C(generator)*C(mode)'
with warnings.catch_warnings(record=True) as issued:
    warnings.simplefilter('always')
    mixed=smf.mixedlm(formula,grouped,groups=grouped['dataset']).fit(reml=True,method='lbfgs')
warning_text=[str(w.message) for w in issued]
for message in warning_text:print('MIXED MODEL WARNING:',message)
assert mixed.converged, 'Mixed model optimizer did not converge; its estimates cannot be reported as converged.'
(OUT/'mixed_model.txt').write_text(mixed.summary().as_text())
mixed_meta=dict(formula=formula,random_effect='dataset random intercept',reml=True,
    converged=bool(mixed.converged),n=120,dataset_groups=5,
    random_intercept_variance=float(mixed.cov_re.iloc[0,0]),residual_variance=float(mixed.scale),warnings=warning_text,
    inference='Normal Wald sensitivity; random-intercept covariance omits task-specific treatment slopes and shared holdout uncertainty. Dataset-level paired contrasts are reported alongside it.')
(OUT/'mixed_metadata.json').write_text(json.dumps(mixed_meta,indent=2))
emms={}
emm_rows=[]
fixed_design=dmatrix(formula.split('~',1)[1],grouped,return_type='dataframe')
assert list(fixed_design.columns)==mixed.model.exog_names
np.testing.assert_allclose(fixed_design.to_numpy(),mixed.model.exog)
for group in grouped.group.cat.categories:
    grid=pd.DataFrame([dict(group=group,generator=g,mode=m) for g,m in itertools.product(['ctgan','tvae'],['synthetic','mix'])])
    vector=np.asarray(build_design_matrices([fixed_design.design_info],grid)[0]).mean(axis=0)
    emms[group]=vector
    estimate=float(vector@mixed.fe_params)
    covariance=mixed.cov_params().iloc[:len(vector),:len(vector)].to_numpy()
    se=float(np.sqrt(vector@covariance@vector))
    emm_rows.append(dict(group=group,estimate=estimate,se=se,ci_low=estimate-1.96*se,ci_high=estimate+1.96*se))
pd.DataFrame(emm_rows).to_csv(OUT/'mixed_emmeans.csv',index=False)
contrasts={
 'new_vs_old':{'hybrid_full':.5,'hybrid_xonly':.5,'generated':-1},
 'has_y_vs_no_y':{'hybrid_full':1,'hybrid_xonly':-1},
 'full_relabel_vs_generated':{'hybrid_full':1,'generated':-1},
 'xonly_relabel_vs_generated':{'hybrid_xonly':1,'generated':-1}}
mixed_rows=[]
for name,weights in contrasts.items():
    vector=sum(weight*emms[group] for group,weight in weights.items())
    test=mixed.t_test(vector[None,:])
    estimate=float(test.effect.item());ci=test.conf_int()[0]
    mixed_rows.append(dict(contrast=name,effect=estimate,ci_low=ci[0],ci_high=ci[1],p_wald=float(test.pvalue)))
mixed_table=pd.DataFrame(mixed_rows)
mixed_table['p_wald_holm']=multipletests(mixed_table.p_wald,method='holm')[1]
mixed_table.to_csv(OUT/'mixed_contrasts.csv',index=False)

# Equal-weight dataset means are the independent units for paired inference.
grouped.to_csv(OUT/'group_scores.csv',index=False)
effect_rows=[];summary=[]
for mode in ['pooled','synthetic','mix']:
    sub=grouped if mode=='pooled' else grouped[grouped['mode']==mode]
    means=sub.groupby(['dataset','group'],observed=True).score.mean().unstack()
    for name,weights in contrasts.items():
        left = ((means.hybrid_full+means.hybrid_xonly)/2 if name=='new_vs_old' else means.hybrid_full if name!='xonly_relabel_vs_generated' else means.hybrid_xonly)
        right=means.hybrid_xonly if name=='has_y_vs_no_y' else means.generated
        differences=left-right
        info=inference(differences)
        info.update(mode=mode,contrast=name,left_mean=float(left.mean()),right_mean=float(right.mean()),
            relative_percent=float(100*(left.mean()-right.mean())/right.mean()),
            score_change=float(differences.mean()),percentage_points=float(100*differences.mean()),
            score_metric='f1_macro',classification_datasets=';'.join(CLASS))
        summary.append(info)
        for dataset in means.index:
            effect_rows.append(dict(mode=mode,contrast=name,dataset=dataset,left=left[dataset],right=right[dataset],difference=differences[dataset],
                                   relative_percent=100*differences[dataset]/right[dataset]))
tests=pd.DataFrame(summary)
for mode in tests['mode'].unique():
    index=tests['mode'].eq(mode)
    tests.loc[index,'p_t_holm']=multipletests(tests.loc[index,'p_t'],method='holm')[1]
    tests.loc[index,'p_exact_holm']=multipletests(tests.loc[index,'p_exact'],method='holm')[1]
tests.to_csv(OUT/'planned_contrasts.csv',index=False)
effects=pd.DataFrame(effect_rows);effects.to_csv(OUT/'dataset_effects.csv',index=False)

# Generator-specific labeler effects and original reference comparisons.
label_rows=[]
for mode,g,label,source in itertools.product(['synthetic','mix'],['ctgan','tvae'],LABELS,['full','xonly']):
    sub=primary[primary['mode'].eq(mode)&primary.generator.eq(g)]
    left=sub[sub.labeler.eq(label)&sub.source.eq(source)].groupby('dataset').score.mean()
    right=sub[sub.labeler.eq('generated')].groupby('dataset').score.mean()
    assert left.index.equals(right.index)
    label_rows.append(dict(mode=mode,generator=g,labeler=label,source=source,
        baseline=float(right.mean()),hybrid=float(left.mean()),relative_percent=100*(left.mean()-right.mean())/right.mean(),**inference(left-right)))
labels=pd.DataFrame(label_rows)
labels['p_t_holm']=multipletests(labels.p_t,method='holm')[1]
labels.to_csv(OUT/'labeler_effects.csv',index=False)
group_means=grouped.groupby(['dataset','mode','group'],observed=True).score.mean().reset_index()
real=reference[reference.dataset.isin(CLASS)&reference.metric.eq('f1_macro')].groupby('dataset').value.mean()
group_means['original_f1']=group_means.dataset.map(real)
group_means['difference_from_real']=group_means.score-group_means.original_f1
group_means.to_csv(OUT/'real_only_comparisons.csv',index=False)

# Regression scores stay in their own units; no five-task F1 pooling.
reg=df[df.dataset.isin(['news','california_housing'])&df.metric.isin(['r2','nmae_sigma'])]
reg.to_csv(OUT/'regression_scores.csv',index=False)
reg_rows=[]
for dataset,metric,mode,g in itertools.product(['news','california_housing'],['r2','nmae_sigma'],['synthetic','mix'],['ctgan','tvae']):
    sub=reg[reg.dataset.eq(dataset)&reg.metric.eq(metric)&reg['mode'].eq(mode)&reg.generator.eq(g)]
    baseline=float(sub[sub.labeler.eq('generated')].value.mean())
    for label,source in itertools.product(LABELS,['full','xonly']):
        hybrid=float(sub[sub.labeler.eq(label)&sub.source.eq(source)].value.mean())
        reg_rows.append(dict(dataset=dataset,metric=metric,mode=mode,generator=g,labeler=label,source=source,
            generated=baseline,hybrid=hybrid,difference=hybrid-baseline,
            improvement=hybrid-baseline if metric=='r2' else baseline-hybrid,
            relative_nmae_reduction_percent=100*(baseline-hybrid)/baseline if metric=='nmae_sigma' else np.nan))
pd.DataFrame(reg_rows).to_csv(OUT/'regression_effects.csv',index=False)

# Publication artifacts, retaining the template's boxplots and method means.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
order=['generated','hybrid_xonly','hybrid_full'];colors=['#768499','#2a8194','#a75b40']
for ax,mode in zip(axes,['synthetic','mix']):
    b=grouped[grouped['mode']==mode]
    ax.boxplot([b.loc[b.group==group,'score'] for group in order],tick_labels=['Generated Y','X-only + labeler','Full + labeler'])
    for i,group in enumerate(order):
        values=b.loc[b.group==group,'score'].to_numpy()
        ax.scatter(np.full(len(values),i+1)+np.linspace(-.12,.12,len(values)),values,color=colors[i],s=14,alpha=.65)
        ax.scatter(i+1,values.mean(),color='black',marker='D',s=30)
    ax.set_title('Synthetic-only' if mode=='synthetic' else 'Real + synthetic');ax.tick_params(axis='x',rotation=15)
axes[0].set_ylabel('Macro F1 (development-selected checkpoint)')
fig.tight_layout();fig.savefig(OUT/'construction_boxplots.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,contrast,title in zip(axes,['new_vs_old','has_y_vs_no_y'],['Relabeling minus generated Y','Full-table minus X-only fitting']):
    for offset,mode,color in [(-.12,'synthetic','#2a8194'),(.12,'mix','#a75b40')]:
        b=effects[(effects['mode']==mode)&effects.contrast.eq(contrast)].set_index('dataset').loc[CLASS]
        ax.scatter(100*b.difference,np.arange(5)+offset,label=mode,color=color,s=40)
    ax.axvline(0,color='#666666',lw=.8);ax.set_title(title);ax.set_xlabel('Macro F1 change (percentage points)')
    ax.set_yticks(np.arange(5),['Adult','Census KDD','Covertype','MNIST12','MNIST28']);ax.grid(axis='x',alpha=.2)
axes[1].legend();fig.tight_layout();fig.savefig(OUT/'dataset_effects.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
fit=models['dataset_labeler_interaction']
axes[0].scatter(fit.fittedvalues,fit.resid,s=12,alpha=.5);axes[0].axhline(0,color='#777777',lw=.8)
axes[0].set(xlabel='Fitted macro F1',ylabel='Residual',title='Dataset-interaction model')
stats.probplot(fit.resid,dist='norm',plot=axes[1]);fig.tight_layout();fig.savefig(OUT/'model_diagnostics.png',dpi=180);plt.close(fig)
metadata=dict(snapshot=snapshot['checked_at_chicago'],selected_records=406,classification_records=290,
    classification_factorial_rows=280,regression_records=116,template='final_328_project.Rmd',
    repeated_model=meta,mixed_model=mixed_meta,lm_models=model_info,
    independent_unit='dataset mean, with seeds, generators, labelers and modes equally weighted',
    primary_family='Four planned contrasts, Holm within pooled/synthetic/mix families; mode families are secondary. Exact sign-flip is a sensitivity under task-effect sign symmetry.',
    minimum_two_sided_sign_flip_p=.0625,shared_holdouts=snapshot['shared_seed_test_partitions'],
    validation='406 source hashes, predictions/checkpoints, source IDs, production settings, saved regression normalization, complete selected grid; reused builder; orthogonal reconstruction and independent OLS F checks.')
metadata['results_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.is_file()}
(OUT/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2))
print(tests[['mode','contrast','score_change','relative_percent','ci_low','ci_high','p_t_holm','p_exact_holm']].to_string(index=False))
print(json.dumps(mixed_meta,indent=2))
