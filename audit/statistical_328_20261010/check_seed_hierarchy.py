"""Compare a ten-observation hierarchical contrast fit with the balanced mean test."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('D:/SummerResearch')
HERE=Path(__file__).parent
names=[f'{arm}/{name}' for arm in ['synthetic','mix'] for name in
       ['matched_relabeling_scores.csv','planned_contrasts.csv']]
expected={name:hashlib.sha256((HERE/'explicit_results'/name).read_bytes()).hexdigest() for name in names}
source=r'''
import hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import statsmodels
import statsmodels.formula.api as smf
ROOT=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_explicit_20261010/results')
EXPECTED=EXPECTED_HASHES
for name,sha in EXPECTED.items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
answer={'equation':'delta_pp ~ C(seed, Sum) + (1 | dataset)',
 'delta_definition':'Within each dataset/seed, mean over generator/has_y/labeler of matched relabeled minus generated macro F1, multiplied by 100.',
 'estimation':'Random dataset intercept, fixed categorical seed block, restricted maximum likelihood.',
 'small_sample_inference':'Exact t with D-1 df for the grand mean in this balanced Gaussian model with positive dataset variance. REML SE independently checked against the analytic cluster-mean SE. No asymptotic Wald z test used.',
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},'analyses':{}}
for arm in ['synthetic','mix']:
 paired=pd.read_csv(ROOT/arm/'matched_relabeling_scores.csv')
 assert len(paired)==120
 data=paired.groupby(['dataset','seed'],as_index=False).difference.mean()
 data['delta_pp']=100*data.difference
 assert len(data)==10 and data.groupby('dataset').size().eq(2).all()
 assert data.groupby('dataset').seed.apply(set).apply(lambda x:x=={42,43}).all()
 fit=smf.mixedlm('delta_pp ~ C(seed, Sum)',data,groups=data.dataset).fit(reml=True,method='lbfgs')
 assert fit.converged
 matrix=data.pivot(index='dataset',columns='seed',values='delta_pp')
 means=matrix.mean(axis=1)
 ds=matrix[43]-matrix[42]
 residual_var=float(ds.var(ddof=1)/2)
 dataset_var=float(means.var(ddof=1)-residual_var/2)
 assert dataset_var>0
 np.testing.assert_allclose(fit.scale,residual_var,rtol=2e-4)
 np.testing.assert_allclose(fit.cov_re.iloc[0,0],dataset_var,rtol=2e-4)
 estimate=float(fit.fe_params['Intercept'])
 model_se=float(np.sqrt(fit.cov_params().loc['Intercept','Intercept']))
 exact_se=float(means.std(ddof=1)/np.sqrt(5))
 np.testing.assert_allclose(model_se,exact_se,rtol=2e-4)
 np.testing.assert_allclose(estimate,means.mean(),rtol=1e-12)
 df=4; raw_p=float(2*t.sf(abs(estimate/exact_se),df))
 original=pd.read_csv(ROOT/arm/'planned_contrasts.csv').query('contrast == "relabel_minus_generated"').iloc[0]
 np.testing.assert_allclose(estimate,100*original.score_change,atol=1e-10)
 np.testing.assert_allclose(raw_p,original.p_t,atol=1e-12)
 lower=estimate-t.ppf(.975,df)*exact_se;upper=estimate+t.ppf(.975,df)*exact_se
 icc=dataset_var/(dataset_var+residual_var)
 answer['analyses'][arm]={'observations':data.to_dict(orient='records'),
  'n_observations':10,'n_datasets':5,'n_seeds':2,'converged':bool(fit.converged),
  'mean_gain_pp':estimate,'REML_se_pp':model_se,'analytic_se_pp':exact_se,
  'seed43_minus_seed42_pp':float(ds.mean()),'dataset_sd_pp':float(np.sqrt(dataset_var)),
  'residual_sd_pp':float(np.sqrt(residual_var)),'ICC_after_fixed_seed_block':icc,
  'approximate_effective_n_for_common_mean':10/(1+icc),'small_sample_df':df,
  'raw_p_small_sample':raw_p,'unadjusted_95_ci_pp':[float(lower),float(upper)],
  'original_holm_p_unchanged_by_equivalent_mean_test':float(original.p_t_holm),
  'dataset_partial_pooling':{name:float(estimate+v.iloc[0]) for name,v in fit.random_effects.items()},
  'balanced_mean_test_equivalence_verified':True}
answer['limitations']=[
 'Dataset effects are assumed independent and Gaussian; MNIST12/28 share a task family, so that assumption remains a limitation.',
 'Seed is an additive fixed block with two levels; this does not estimate a population variance over all seeds.',
 'The within-dataset residual combines dataset-specific seed variation and other variability remaining in each paired contrast.',
 'The simple hierarchical contrast model checks the relabeling main contrast; it does not replace the explicit generator/has_y/labeler factorial analysis.',
 'Existing report files and experimental settings are unchanged.'
]
print(json.dumps(answer,indent=2))
'''.replace('EXPECTED_HASHES',repr(expected))
ssh=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
     '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
r=subprocess.run(ssh+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -'],
                 input=source,text=True,capture_output=True)
if r.stderr:print(r.stderr)
r.check_returncode()
result=json.loads(r.stdout)
(HERE/'seed_hierarchy_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps({arm:{k:v for k,v in analysis.items() if k not in ['observations','dataset_partial_pooling']}
                  for arm,analysis in result['analyses'].items()},indent=2))
