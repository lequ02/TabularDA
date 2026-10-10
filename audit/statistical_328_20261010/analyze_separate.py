"""Independent analyses of frozen real-data training arms. Run remotely only."""
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
from patsy import dmatrix, build_design_matrices
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from analyze_results import inference
from rebuild_rf_xgb_dnn import split_plot

ROOT=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_separate_20261010')
FROZEN=ROOT.parent/'statistical_328_20261010'
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(FROZEN/'scores.csv')
snapshot=json.loads((FROZEN/'snapshot.json').read_text())
assert snapshot['selected_completed']==406 and not snapshot['missing']
CLASS=['adult','census_kdd','covertype','mnist12','mnist28']
CONDITIONS=['generated']+[f'{label}_has_y{h}' for label in ['rf','xgb','dnn'] for h in [0,1]]
CONTRASTS={
 'relabel_minus_generated':np.array([-1]+[1/6]*6),
 'has_y1_minus_has_y0':np.array([0,-1/3,1/3,-1/3,1/3,-1/3,1/3]),
 'has_y1_minus_generated':np.array([-1,0,1/3,0,1/3,0,1/3]),
 'has_y0_minus_generated':np.array([-1,1/3,0,1/3,0,1/3,0])}
assert all(np.isclose(w.sum(),0) for w in CONTRASTS.values())
pd.DataFrame(CONTRASTS,index=CONDITIONS).rename_axis('condition').to_csv(OUT/'contrast_weights.csv')

def normalized(frame):
    result=frame.copy()
    result['has_y']=result.source.map({'full':1,'xonly':0}).astype('Int64')
    result['condition']=['original' if l=='original' else 'generated' if l=='generated' else f'{l}_has_y{int(h)}' for l,h in zip(result.labeler,result.has_y)]
    return result.drop(columns=['source','mode'])

def save(frame,path):
    assert not {'source','mode','group','approach_has_y'}.intersection(frame.columns)
    frame.to_csv(path,index=False)

save(normalized(df[df.labeler.eq('original')]),OUT/'original_references.csv')
all_meta={}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for arm in ['synthetic','mix']:
    dest=OUT/arm;dest.mkdir(exist_ok=True)
    legacy=df[df.dataset.isin(CLASS)&df.metric.eq('f1_macro')&df['mode'].eq(arm)].copy()
    primary=normalized(legacy)
    primary['macro_F1']=primary.value
    primary['condition']=pd.Categorical(primary.condition,CONDITIONS)
    assert len(primary)==140 and not primary.duplicated(['dataset','seed','generator','condition']).any()
    assert primary.groupby(['dataset','seed']).size().eq(14).all()
    save(primary,dest/'classification_scores.csv')
    fits={};model_info=[]
    formulas={
     'full_interaction':'macro_F1 ~ C(dataset)*C(generator)*C(condition)',
     'blocked':'macro_F1 ~ C(dataset) + C(generator)*C(condition)',
     'dataset_interaction':'macro_F1 ~ C(dataset)*C(condition) + C(generator)'}
    for name,formula in formulas.items():
        fit=smf.ols(formula,primary).fit();fits[name]=fit
        assert np.linalg.matrix_rank(fit.model.exog)==fit.model.exog.shape[1]
        anova_lm(fit,typ=2).rename_axis('term').reset_index().to_csv(dest/f'lm_{name}_anova.csv',index=False)
        model_info.append(dict(name=name,formula=formula,n=int(fit.nobs),df_model=fit.df_model,df_resid=fit.df_resid,
          rank=int(np.linalg.matrix_rank(fit.model.exog)),columns=fit.model.exog.shape[1],r_squared=fit.rsquared,
          adjusted_r_squared=fit.rsquared_adj,shapiro_p=float(stats.shapiro(fit.resid).pvalue)))
        (dest/f'lm_{name}.txt').write_text(fit.summary().as_text())
    pd.DataFrame(model_info).to_csv(dest/'lm_model_summary.csv',index=False)
    # Existing orthogonal repeated-measures implementation gets its historical
    # input schema internally; every published factor uses the requested name.
    repeated,inventory,meta,checks,strata=split_plot(legacy,'classification','f1_macro',[arm])
    for frame,column in [(repeated,'term'),(pd.DataFrame(checks),'term'),(pd.DataFrame(strata),'error_stratum')]:
        frame[column]=frame[column].str.replace('x_inclusion','has_y',regex=False)
        filename='repeated_anova.csv' if frame is repeated else 'independent_F_checks.csv' if column=='term' else 'anova_error_strata.csv'
        frame.to_csv(dest/filename,index=False)
    assert len(repeated)==19 and not repeated.term.str.contains('mode').any()
    meta.pop('training_modes')
    pd.DataFrame(inventory).to_csv(dest/'repeated_units.csv',index=False)

    formula='macro_F1 ~ C(generator)*C(condition)'
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter('always')
        mixed=smf.mixedlm(formula,primary,groups=primary.dataset).fit(reml=True,method='lbfgs')
    warning_text=[str(w.message) for w in issued]
    for message in warning_text:print(arm,'MIXED MODEL WARNING:',message)
    assert mixed.converged, 'Mixed model did not converge.'
    fixed=dmatrix(formula.split('~',1)[1],primary,return_type='dataframe')
    assert list(fixed.columns)==mixed.model.exog_names
    np.testing.assert_allclose(fixed.to_numpy(),mixed.model.exog)
    covariance=mixed.cov_params().iloc[:len(mixed.fe_params),:len(mixed.fe_params)].to_numpy()
    vectors={};emm_rows=[]
    for condition in CONDITIONS:
        grid=pd.DataFrame([dict(generator=g,condition=condition) for g in ['ctgan','tvae']])
        vector=np.asarray(build_design_matrices([fixed.design_info],grid)[0]).mean(axis=0)
        vectors[condition]=vector
        estimate=float(vector@mixed.fe_params);se=float(np.sqrt(vector@covariance@vector))
        emm_rows.append(dict(condition=condition,estimate=estimate,se=se,ci_low=estimate-1.96*se,ci_high=estimate+1.96*se))
    pd.DataFrame(emm_rows).to_csv(dest/'mixed_emmeans.csv',index=False)
    mixed_rows=[]
    for name,weights in CONTRASTS.items():
        vector=sum(w*vectors[c] for c,w in zip(CONDITIONS,weights))
        test=mixed.t_test(vector[None,:]);ci=test.conf_int()[0]
        mixed_rows.append(dict(contrast=name,effect=float(test.effect.item()),ci_low=ci[0],ci_high=ci[1],p_wald=float(test.pvalue)))
    mixed_table=pd.DataFrame(mixed_rows)
    mixed_table['p_wald_holm']=multipletests(mixed_table.p_wald,method='holm')[1]
    mixed_table.to_csv(dest/'mixed_contrasts.csv',index=False)
    (dest/'mixed_model.txt').write_text(mixed.summary().as_text())
    mixed_meta=dict(formula=formula+' + (1 | dataset)',converged=bool(mixed.converged),n=140,dataset_groups=5,
      fixed_columns=len(mixed.fe_params),reml=True,random_intercept_variance=float(mixed.cov_re.iloc[0,0]),
      residual_variance=float(mixed.scale),warnings=warning_text,
      inference='Normal Wald; random dataset intercept, independent conditional residuals. Does not model dataset-specific treatment slopes or shared heldout uncertainty.')
    (dest/'mixed_metadata.json').write_text(json.dumps(mixed_meta,indent=2))
    means=primary.groupby(['dataset','condition'],observed=True).macro_F1.mean().unstack().reindex(columns=CONDITIONS)
    rows=[];effect_rows=[]
    for name,weights in CONTRASTS.items():
        left=means.loc[:,np.array(CONDITIONS)[weights>0]]@weights[weights>0]
        right=means.loc[:,np.array(CONDITIONS)[weights<0]]@(-weights[weights<0])
        difference=left-right
        np.testing.assert_allclose(difference,means.to_numpy()@weights)
        estimate=float(difference.mean())
        np.testing.assert_allclose(estimate,mixed_table.set_index('contrast').loc[name,'effect'],atol=1e-9)
        rows.append(dict(contrast=name,left_mean=float(left.mean()),right_mean=float(right.mean()),
          score_change=estimate,percentage_points=100*estimate,relative_percent=100*estimate/right.mean(),**inference(difference)))
        for dataset in means.index:
            effect_rows.append(dict(contrast=name,dataset=dataset,left=left[dataset],right=right[dataset],difference=difference[dataset]))
    tests=pd.DataFrame(rows)
    tests['p_t_holm']=multipletests(tests.p_t,method='holm')[1]
    tests['p_exact_holm']=multipletests(tests.p_exact,method='holm')[1]
    tests.to_csv(dest/'planned_contrasts.csv',index=False)
    effects=pd.DataFrame(effect_rows);effects.to_csv(dest/'dataset_effects.csv',index=False)
    labels=[]
    for generator,labeler,h in itertools.product(['ctgan','tvae'],['rf','xgb','dnn'],[0,1]):
        sub=primary[primary.generator.eq(generator)]
        left=sub[sub.labeler.eq(labeler)&sub.has_y.eq(h)].groupby('dataset').macro_F1.mean()
        right=sub[sub.labeler.eq('generated')].groupby('dataset').macro_F1.mean()
        assert left.index.equals(right.index)
        labels.append(dict(generator=generator,labeler=labeler,has_y=h,generated=float(right.mean()),
          relabeled=float(left.mean()),relative_percent=100*(left.mean()-right.mean())/right.mean(),**inference(left-right)))
    labels=pd.DataFrame(labels);labels['p_t_holm']=multipletests(labels.p_t,method='holm')[1]
    labels.to_csv(dest/'labeler_effects.csv',index=False)

    regression=normalized(df[df.dataset.isin(['news','california_housing'])&df.metric.isin(['r2','nmae_sigma'])&df['mode'].eq(arm)])
    save(regression,dest/'regression_scores.csv')
    d2=normalized(pd.read_csv(ROOT/'regression_d2_scores.csv').query('mode == @arm'))
    d2['metric']='d2_absolute_error';save(d2,dest/'regression_d2_scores.csv')
    combined=pd.concat([regression,d2],ignore_index=True)
    reg_rows=[]
    for dataset,metric in itertools.product(['news','california_housing'],['r2','nmae_sigma','d2_absolute_error']):
        sub=combined[combined.dataset.eq(dataset)&combined.metric.eq(metric)]
        original=df[df.dataset.eq(dataset)&df.metric.eq(metric)&df.labeler.eq('original')].value.mean() if metric!='d2_absolute_error' else pd.read_csv(ROOT/'regression_d2_scores.csv').query('dataset == @dataset and labeler == "original"').value.mean()
        row=dict(dataset=dataset,metric=metric,original=float(original),generated=float(sub[sub.labeler.eq('generated')].value.mean()))
        for h in [0,1]:row[f'has_y{h}']=float(sub[~sub.labeler.eq('generated')&sub.has_y.eq(h)].value.mean())
        reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(dest/'regression_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
    for ax,contrast,title in zip(axes,list(CONTRASTS)[:2],['Relabeling - generated Y','has_y=1 - has_y=0']):
        b=effects[effects.contrast.eq(contrast)].set_index('dataset').loc[CLASS]
        ax.scatter(100*b.difference,np.arange(5),color='#237b8a',s=40)
        ax.axvline(0,color='#777777',lw=.8);ax.set_title(title);ax.set_xlabel('Macro F1 change (percentage points)')
        ax.set_yticks(np.arange(5),['Adult','Census KDD','Covertype','MNIST12','MNIST28']);ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(dest/'dataset_effects.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.2));fit=fits['dataset_interaction']
    axes[0].scatter(fit.fittedvalues,fit.resid,s=12,alpha=.5);axes[0].axhline(0,color='#777777',lw=.8)
    axes[0].set(xlabel='Fitted macro F1',ylabel='Residual')
    stats.probplot(fit.resid,dist='norm',plot=axes[1]);fig.tight_layout();fig.savefig(dest/'model_diagnostics.png',dpi=180);plt.close(fig)
    metadata=dict(training_rows='100000 synthetic rows only' if arm=='synthetic' else 'all real training rows plus 100000 synthetic rows',
      classification_rows=140,classification_datasets=CLASS,condition_levels=CONDITIONS,seed_levels=[42,43],
      ols_models=model_info,mixed_model=mixed_meta,repeated_model=meta,
      multiplicity='Holm across four planned contrasts, twelve method contrasts, and nineteen repeated ANOVA terms; each family is independent within this training analysis.',
      caveat='Seeds share test partitions; five dataset means are the primary independent units. Unadjusted confidence intervals; exact two-sided sign-flip minimum p=0.0625.')
    (dest/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2));all_meta[arm]=metadata
    print(arm,tests[['contrast','score_change','p_t_holm']].to_string(index=False))
    print(arm,mixed_table.to_string(index=False))
(OUT/'analysis_metadata.json').write_text(json.dumps(dict(snapshot=snapshot['checked_at_chicago'],selected_records=406,
  frozen_scores_sha256=hashlib.sha256((FROZEN/'scores.csv').read_bytes()).hexdigest(),analyses=all_meta,
  analysis='Two independent fits; no pooled effects or training factor.'),indent=2))
