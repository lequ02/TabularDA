"""Read-only remote rank check of the user's explicit factorial specification."""
import json
from pathlib import Path
import subprocess

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
ssh=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
 '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
source='''
import json
from pathlib import Path
import numpy as np
import pandas as pd
from patsy import dmatrix
root=Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/statistical_328_separate_20261010/results')
rows=[]
for arm in ['synthetic','mix']:
 d=pd.read_csv(root/arm/'classification_scores.csv')
 for scope,b in [('all_seven_constructions',d),('RF_XGB_DNN_only',d[d.labeler!='generated'])]:
  for name,formula in [
   ('seed_interactions','C(dataset) + C(generator)*C(has_y)*C(labeler)*C(seed)'),
   ('seed_block','C(dataset) + C(seed) + C(generator)*C(has_y)*C(labeler)')]:
   design=dmatrix(formula,b,return_type='dataframe')
   rank=int(np.linalg.matrix_rank(design.to_numpy()))
   rows.append(dict(training=arm,scope=scope,name=name,formula='macro_F1 ~ '+formula,
    rows=len(b),columns=design.shape[1],rank=rank,aliased_columns=design.shape[1]-rank,residual_df=len(b)-rank))
print(json.dumps(rows,indent=2))
'''
result=subprocess.run(ssh+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -'],input=source,text=True,capture_output=True)
if result.stderr:print(result.stderr)
result.check_returncode()
rows=json.loads(result.stdout)
(HERE/'explicit_formula_rank_check.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
