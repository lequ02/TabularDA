"""Read newly saved D2 sidecars and compute small descriptive tables remotely."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path('D:/SummerResearch');HERE=Path(__file__).parent
remote=r'''
import base64,hashlib,io,json,math,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
root=Path('/home/thuy/Research/minh_data_synth/TabularDA')
analysis=root/'.cache/statistical_328_20261010'
snapshot=json.loads((analysis/'snapshot.json').read_text())
rows=[];payloads={};checks=[]
for item in snapshot['records']:
    if item['dataset'] not in ('news','california_housing'):continue
    path=root/item['path'];record=json.loads(path.read_bytes())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    sidecar=path.with_name(path.name.replace('.run.json','.d2.json'))
    payload=sidecar.read_bytes();d=json.loads(payload)
    assert d['run_record']==item['path'] and d['run_record_sha256']==item['sha256']
    norm=d['absolute_error_normalization'];test=record['split_manifest']['files']['test']['raw']
    assert norm['split']=='test' and norm['rows']==test['rows'] and norm['target_table_sha256']==test['sha256']
    # Match the established float32-vs-CSV verification in compute_regression_d2.py.
    assert math.isclose(norm['prediction_mae'],record['test_scores']['mae'],rel_tol=1e-6,abs_tol=1e-6), item['path']
    value=d['test_scores']['d2_absolute_error']
    assert abs(value-(1-norm['prediction_mae']/norm['baseline_mae']))<1e-12
    assert hashlib.sha256(Path(d['predictions_path']).read_bytes()).hexdigest()==d['predictions_sha256']
    suffix=path.name.removeprefix(f"{item['dataset']}_seed{item['seed']}_").removesuffix('.run.json')
    if record['train_option']=='original':g,source,label='real','real','original'
    else:g,source,label,_=suffix.split('_')
    rows.append(dict(dataset=item['dataset'],seed=item['seed'],mode=record['train_option'],generator=g,
        source=source,labeler=label,value=value,record=item['path'],record_sha256=item['sha256'],sidecar=str(sidecar.relative_to(root)),
        sidecar_sha256=hashlib.sha256(payload).hexdigest(),median_y=norm['median_y'],baseline_mae=norm['baseline_mae']))
    payloads[str(sidecar.relative_to(root))]=payload
assert len(rows)==116
df=pd.DataFrame(rows);df.to_csv(analysis/'results/regression_d2_scores.csv',index=False)
summary=[];effects=[]
for dataset in ('news','california_housing'):
    original=df[df.dataset.eq(dataset)&df.labeler.eq('original')].value.mean()
    for mode in ('synthetic','mix'):
        sub=df[df.dataset.eq(dataset)&df['mode'].eq(mode)]
        generated=sub[sub.labeler.eq('generated')].value.mean()
        full=sub[sub.labeler.isin(['rf','xgb','dnn'])&sub.source.eq('full')].value.mean()
        xonly=sub[sub.labeler.isin(['rf','xgb','dnn'])&sub.source.eq('xonly')].value.mean()
        summary.append(dict(dataset=dataset,mode=mode,original=original,generated=generated,full=full,xonly=xonly))
        effects.append(dict(dataset=dataset,mode=mode,full_minus_generated=full-generated,xonly_minus_generated=xonly-generated,full_minus_xonly=full-xonly))
pd.DataFrame(summary).to_csv(analysis/'results/regression_d2_summary.csv',index=False)
pd.DataFrame(effects).to_csv(analysis/'results/regression_d2_effects.csv',index=False)
evidence=dict(checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),verified_sidecars=116,
    inputs='Original record hashes, selected metric arithmetic, test-target metadata and saved prediction hashes verified.',
    sidecars=rows)
(analysis/'results/d2_verification.json').write_text(json.dumps(evidence,indent=2))
b=io.BytesIO()
with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
    for name,payload in payloads.items():z.writestr(name,payload)
    for name in ('regression_d2_scores.csv','regression_d2_summary.csv','regression_d2_effects.csv','d2_verification.json'):
        z.write(analysis/'results'/name,'results/'+name)
print(base64.b64encode(b.getvalue()).decode())
'''
ssh=['ssh','-i',str(ROOT/'.cache/remote_intrusion/server_key'),'-o','IdentitiesOnly=yes',
     '-o','UserKnownHostsFile='+str(ROOT/'audit/ssh_known_hosts'),'-o','BatchMode=yes','thuy@10.24.10.133']
r=subprocess.run(ssh+['env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/thuy/miniconda3/envs/env/bin/python -'],input=remote,text=True,capture_output=True)
if r.stderr:print(r.stderr)
r.check_returncode()
archive=HERE/'d2_snapshot.zip';archive.write_bytes(base64.b64decode(r.stdout.strip(),validate=True))
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        target=HERE/name if name.startswith('results/') else HERE/'snapshot/records'/name
        assert target.resolve().is_relative_to(HERE.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
print('Retrieved 116 source-linked D2 sidecars and remotely computed descriptive tables.')
