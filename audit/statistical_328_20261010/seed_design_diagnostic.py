"""Read-only seed diagnostics and prospective power calculations on the server."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('D:/SummerResearch')
HERE=Path(__file__).parent
expected={str(p.relative_to(HERE/'explicit_results')).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for arm in ['synthetic','mix'] for p in [HERE/'explicit_results'/arm/name for name in
          ['matched_relabeling_scores.csv','dataset_effects.csv','planned_contrasts.csv']]}
source=r'''
import hashlib,json,math,platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.power import TTestPower
from scipy.stats import t
ROOT=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_explicit_20261010/results')
EXPECTED=EXPECTED_HASHES
for name,sha in EXPECTED.items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
power=TTestPower()
answer={'environment':{'python':platform.python_version(),'numpy':np.__version__,
                      'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
        'inputs_verified':len(EXPECTED),'assumptions':'Two-sided one-sample t power over independent dataset means; alpha=.05/4 conservative Holm planning bound; plug-in SD from five observed tasks.',
        'analyses':{}}
for arm in ['synthetic','mix']:
 paired=pd.read_csv(ROOT/arm/'matched_relabeling_scores.csv')
 delta=paired.groupby(['dataset','seed']).difference.mean().unstack('seed')
 assert delta.shape==(5,2)
 means=delta.mean(axis=1)
 previous=pd.read_csv(ROOT/arm/'dataset_effects.csv').query('contrast == "relabel_minus_generated"').set_index('dataset').difference
 np.testing.assert_allclose(means.sort_index(),previous.sort_index(),atol=1e-12)
 observed_mean=float(means.mean()); observed_sd=float(means.std(ddof=1))
 differences=delta[43]-delta[42]
 rows=[]
 for target in [observed_mean,.10,.05,.03]:
  effect=target/observed_sd
  n80=int(math.ceil(power.solve_power(effect_size=effect,alpha=.0125,power=.8)))
  n90=int(math.ceil(power.solve_power(effect_size=effect,alpha=.0125,power=.9)))
  rows.append({'target_gain':target,'assumed_dataset_sd':observed_sd,'standardized_effect':effect,
               'datasets_for_80_percent_power':n80,'datasets_for_90_percent_power':n90,
               'power_at_5_datasets':float(power.power(effect_size=effect,nobs=5,alpha=.0125))})
 seed_rows=[]
 for seed in [42,43]:
  x=delta[seed]
  seed_rows.append({'seed':seed,'mean_gain':float(x.mean()),'between_dataset_sd':float(x.std(ddof=1)),
                    'dataset_gains':x.to_dict()})
 n_passing=next(n for n in range(2,501) if 2*t.sf(abs(observed_mean/(observed_sd/math.sqrt(n))),n-1)<.0125)
 answer['analyses'][arm]={'seed_results':seed_rows,'observed_mean':observed_mean,'observed_dataset_sd':observed_sd,
  'observed_se':observed_sd/math.sqrt(5),'mean_absolute_seed_difference':float(differences.abs().mean()),
  'seed_difference_by_dataset':differences.to_dict(),'planning_scenarios':rows,
  'datasets_to_cross_cutoff_if_mean_sd_stay_identical':n_passing}
print(json.dumps(answer,indent=2))
'''.replace('EXPECTED_HASHES',repr(expected))
ssh=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
     '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
r=subprocess.run(ssh+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -'],
                 input=source,text=True,capture_output=True)
if r.stderr:print(r.stderr)
r.check_returncode()
result=json.loads(r.stdout)
result['limitations']=[
 'Seeds repeat generator, labeler and downstream training randomness on fixed prepared partitions; they are not new train/test samples.',
 'Power scenarios assume independent prediction tasks, normal dataset-level differences and the same gain/SD as specified. Current SD is estimated from only five task means.',
 'MNIST12 and MNIST28 share the underlying prediction problem. Treating resolutions as independent domains is inadequate for planning dataset diversity.',
 'Two seeds are insufficient for reliable decomposition of seed and task variance; no precise forecast of the effect of adding seeds is provided.',
 'The number that crosses a p-value cutoff if mean/SD stay unchanged is not a power calculation or a guarantee. Prospective sample-size columns use noncentral t power.'
]
(HERE/'seed_design_diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
