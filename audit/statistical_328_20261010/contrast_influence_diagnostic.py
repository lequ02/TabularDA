"""Read-only influence checks and conditional seed projections, run remotely."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
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
from statsmodels.stats.multitest import multipletests
ROOT=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_explicit_20261010/results')
EXPECTED=EXPECTED_HASHES
for name,sha in EXPECTED.items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
names=['relabel_minus_generated','has_y1_minus_has_y0','has_y1_minus_generated','has_y0_minus_generated']
def summaries(blocks):
 means=blocks.groupby('dataset')[names].mean()
 result=[]
 for name in names:
  x=means[name];n=len(x);se=float(x.std(ddof=1)/np.sqrt(n));mu=float(x.mean())
  result.append({'contrast':name,'n_datasets':n,'mean_pp':mu,'se_pp':se,
                 'raw_p':float(2*t.sf(abs(mu/se),n-1))})
 for row,p in zip(result,multipletests([r['raw_p'] for r in result],method='holm')[1]):row['holm_p']=float(p)
 return result
answer={'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                     'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
        'inputs_verified':len(EXPECTED),'analyses':{}}
for arm in ['synthetic','mix']:
 paired=pd.read_csv(ROOT/arm/'matched_relabeling_scores.csv')
 assert len(paired)==120
 paired['delta_pp']=100*paired.difference
 by_h=paired.groupby(['dataset','seed','generator','has_y']).delta_pp.mean().unstack('has_y')
 blocks=by_h.reset_index()[['dataset','seed','generator']]
 blocks[names[0]]=(by_h[0].to_numpy()+by_h[1].to_numpy())/2
 blocks[names[1]]=by_h[1].to_numpy()-by_h[0].to_numpy()
 blocks[names[2]]=by_h[1].to_numpy();blocks[names[3]]=by_h[0].to_numpy()
 assert len(blocks)==20 and blocks.groupby('dataset').size().eq(4).all()
 original=pd.read_csv(ROOT/arm/'planned_contrasts.csv').set_index('contrast')
 full=summaries(blocks)
 for r in full:
  saved=original.loc[r['contrast']]
  np.testing.assert_allclose(r['mean_pp'],100*saved.score_change,atol=1e-10)
  np.testing.assert_allclose(r['holm_p'],saved.p_t_holm,atol=1e-12)
 dataset_influence=[];block_influence=[]
 for dataset in sorted(blocks.dataset.unique()):
  subset=blocks[blocks.dataset.ne(dataset)]
  result=summaries(subset)
  dataset_influence.append({'omitted_dataset':dataset,'remaining_records':result})
 for i,b in blocks.iterrows():
  result=summaries(blocks.drop(index=i))
  block_influence.append({'omitted_dataset':b.dataset,'omitted_seed':int(b.seed),
                          'omitted_generator':b.generator,'remaining_records':result})
 main_datasets=blocks.groupby('dataset')[names[0]].mean()
 variance_components=[];seed_projection=[]
 # Exact balanced REML variance components with a fixed seed block, independently
 # verified for the main contrast in check_seed_hierarchy.py. Use them for the
 # three generated-target contrasts, whose dataset variances are positive.
 for name in [names[0],names[2],names[3]]:
  grid=blocks.groupby(['dataset','seed'])[name].mean().unstack('seed')
  means=grid.mean(axis=1);differences=grid[43]-grid[42]
  residual_variance=float(differences.var(ddof=1)/2)
  dataset_variance=float(means.var(ddof=1)-residual_variance/2)
  assert dataset_variance>0
  variance_components.append({'contrast':name,'mean_pp':float(means.mean()),
   'dataset_variance_pp2':dataset_variance,'residual_variance_pp2':residual_variance,
   'mean_variance_fraction_between_datasets_at_two_seeds':dataset_variance/(dataset_variance+residual_variance/2)})
 for seeds in [2,5,10,20,100,'infinity']:
  projected=[]
  for c in variance_components:
   r=0. if seeds=='infinity' else c['residual_variance_pp2']/seeds
   se=float(np.sqrt((c['dataset_variance_pp2']+r)/5))
   projected.append({'contrast':c['contrast'],'mean_pp':c['mean_pp'],'se_pp':se,
                      'raw_p':float(2*t.sf(abs(c['mean_pp']/se),4))})
  minimum=min(r['raw_p'] for r in projected)
  # Even if has_y becomes the smallest p and passes Holm's first step, the
  # smallest of the three generated-target comparisons must then pass .05/3.
  seed_projection.append({'seeds_per_dataset':seeds,'generated_target_contrasts':projected,
   'minimum_raw_p_of_three':minimum,'most_lenient_first_cutoff_for_these_three':.05/3,
   'none_of_three_can_pass_holm_with_any_fourth_p':bool(minimum>.05/3)})
 per_generator=[]
 for (dataset,generator),part in paired.groupby(['dataset','generator']):
  per_generator.append({'dataset':dataset,'generator':generator,
    'generated_F1':float(part.value_generated.mean()),'relabeled_F1':float(part.value.mean()),
    'mean_gain_pp':float(part.delta_pp.mean()),'negative_relabel_cells':int((part.delta_pp<0).sum()),
    'relabel_cells':len(part)})
 answer['analyses'][arm]={'full_contrasts':full,'dataset_seed_gains':paired.groupby(['dataset','seed']).delta_pp.mean().reset_index().to_dict(orient='records'),
  'dataset_mean_gains':main_datasets.to_dict(),'dataset_leave_one_out':dataset_influence,
  'matched_block_leave_one_out':block_influence,'by_dataset_generator':per_generator,
  'positive_relabel_cells':int((paired.delta_pp>0).sum()),'negative_relabel_cells':int((paired.delta_pp<0).sum()),
  'sum_gain_fraction_from_MNIST':float(main_datasets.loc[['mnist12','mnist28']].sum()/main_datasets.sum()),
  'without_MNIST_family':summaries(blocks[~blocks.dataset.isin(['mnist12','mnist28'])]),
  'variance_components':variance_components,'conditional_seed_projections':seed_projection}
answer['interpretation']={
 'influence_checks':'Diagnostic omission only; neither records nor reported inferential families are removed or changed.',
 'matched_block':'One dataset/seed/generator generated baseline and all six matched relabeled comparisons are omitted together, avoiding baseline reuse as independent evidence.',
 'seed_projections':'Conditional scenarios holding true means and estimated variance components fixed; two observed seeds cannot establish reliable forecasts. New experiments may change both.',
 'infinite_seed_limit':'Population-task mean retains dataset variation even when within-task run noise tends to zero. Fixed seed blocks condition on the studied seeds; seed-population variation is not estimated.',
 'data_caveats':'MNIST12/28 share a task family; documented MNIST12 TVAE holdout-feature collisions remain. Influence arithmetic cannot rule out experimental or provenance problems.'
}
print(json.dumps(answer,indent=2))
'''.replace('EXPECTED_HASHES',repr(expected))
ssh=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
 '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
r=subprocess.run(ssh+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -'],
 input=source,text=True,capture_output=True)
if r.stderr:print(r.stderr)
r.check_returncode();result=json.loads(r.stdout)
(HERE/'contrast_influence_diagnostic.json').write_text(json.dumps(result,indent=2))
for arm,a in result['analyses'].items():
 print(arm,'dataset means:',json.dumps(a['dataset_mean_gains']))
 print(arm,'leave-one-dataset-out:',json.dumps([{'omitted':r['omitted_dataset'],**r['remaining_records'][0]} for r in a['dataset_leave_one_out']]))
 block=[r['remaining_records'][0] for r in a['matched_block_leave_one_out']]
 print(arm,'leave-one-matched-block-out range:',json.dumps({k:[min(r[k] for r in block),max(r[k] for r in block)] for k in ['mean_pp','raw_p','holm_p']}))
 print(arm,'MNIST fraction:',a['sum_gain_fraction_from_MNIST'],'positive/negative cells:',a['positive_relabel_cells'],a['negative_relabel_cells'])
 print(arm,'without MNIST:',json.dumps(a['without_MNIST_family'][0]))
 print(arm,'generator breakdown:',json.dumps(a['by_dataset_generator']))
 print(arm,'seed projections:',json.dumps([{'seeds':r['seeds_per_dataset'],**r['generated_target_contrasts'][0],
   'no_generated_target_comparison_passes_holm':r['none_of_three_can_pass_holm_with_any_fourth_p']} for r in a['conditional_seed_projections']]))
