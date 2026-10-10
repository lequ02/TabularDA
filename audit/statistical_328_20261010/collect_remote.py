"""Freeze the current report family remotely; no training or result changes."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import statsmodels

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
OUT = ROOT / '.cache/statistical_328_20261010'
OUT.mkdir(exist_ok=True)
assert not (OUT/'snapshot.json').exists(), 'A completed snapshot already exists; preserve it.'
SCOPES = {'adult':'corrected_v2', 'covertype':'corrected_v2',
          'census_kdd':'census_kdd_weighted_macro_f1_20261005',
          'mnist12':'mnist_head_fixed_20261009', 'mnist28':'mnist_head_fixed_20261009',
          'news':'news_log_v1', 'california_housing':'housing_no_faker_20261009'}
sys.path.insert(0, str(ROOT/'src'))
from modeling.constants import method_parts
spec = importlib.util.spec_from_file_location('mnist_plan', ROOT/'.cache/mnist_head_rerun_plan_20261009/run_plan.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
plan = json.loads(runner.PLAN_PATH.read_text())
mnist_jobs = {job['run_id']:job for job in plan['jobs']}
rows = []
records = []
missing = []
raw_files = {}
test_partitions = {}
for dataset, scope in SCOPES.items():
    for seed in (42,43):
        suffixes = ['real_original'] + [f'{g}_{fit}_{label}_{mode}'
            for g in ('ctgan','tvae') for fit, labels in (('full',('generated','rf','xgb','dnn')), ('xonly',('rf','xgb','dnn')))
            for label in labels for mode in ('synthetic','mix')]
        manifests = []
        for suffix in suffixes:
            run_id = f'{dataset}_seed{seed}_{suffix}'
            path = ROOT/'output'/scope/dataset/'acc'/(run_id+'.run.json')
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            payload = path.read_bytes()
            r = json.loads(payload)
            if dataset.startswith('mnist'):
                assert runner.verify(mnist_jobs[run_id]) == r
            assert r['dataset']==dataset and r['seed']==seed
            assert r['selected_dev_epoch'] is not None
            assert (r['batch_size'],r['learning_rate'],r['epoch_budget'])==(128,.001,100)
            assert Path(r['downstream_weight_path']).is_file()
            pred_path = Path(r['predictions_path'])
            assert pred_path.is_file()
            with pred_path.open() as stream:
                predictions = list(csv.DictReader(stream))
            manifest = r['split_manifest']
            assert [int(p['source_id']) for p in predictions]==manifest['splits']['test']
            assert all(v==0 for overlaps in manifest['train_overlap'].values() for v in overlaps.values())
            manifests.append(manifest)
            test = manifest['files']['test']['raw']
            test_partitions[f'{dataset}/{seed}'] = dict(hash=test['sha256'], rows=test['rows'],
                ids_sha256=hashlib.sha256(json.dumps(manifest['splits']['test']).encode()).hexdigest())
            if r['train_option']=='original':
                g, fit, label = 'real','real','original'
            else:
                g, fit, label = method_parts(r['augment_option'])
                provenance=r['generator_provenance']
                parameters=provenance['parameters']
                assert (parameters['epochs'],parameters['batch_size'],parameters['cuda'])==(500,500,True)
                assert provenance['seed']==seed and r['synthetic_quality']['rows']==100000
                assert provenance['training_rows']==manifest['files']['train']['raw']['rows']
                assert Path(r['generator_model_path']).is_file()
            scores=r['test_scores']
            assert all(np.isfinite(value) for value in scores.values())
            if dataset in ('news','california_housing'):
                norm=r['target_normalization']
                assert norm['split']=='test' and norm['ddof']==0 and norm['rows']==test['rows']
                assert norm['target_table_sha256']==test['sha256']
                assert abs(scores['nmae_sigma']-scores['mae']/norm['sigma_y'])<1e-12
            if dataset=='news':
                assert r['classifier']=='DNN_News_log_no_norm'
                assert r['target_transform']['name']=='log' and r['target_transform']['normalization']=='none'
            if dataset=='census_kdd':
                assert r['selection_metric']=='f1_macro'
                assert r['evaluation_protocol']['output_namespace']==scope
            else:
                assert r['selection_metric']=='loss'
            relative=path.relative_to(ROOT).as_posix()
            sha=hashlib.sha256(payload).hexdigest()
            raw_files[relative]=payload
            records.append(dict(path=relative,sha256=sha,dataset=dataset,seed=seed,scope=scope,
                                weight_present=True,predictions_present=True,source_id_order_verified=True))
            for metric,value in scores.items():
                rows.append(dict(domain='real',dataset=dataset,seed=seed,generator=g,source=fit,labeler=label,
                    mode=r['train_option'],metric=metric,value=value,protocol=scope,record=relative,
                    record_sha256=sha,test_hash=test['sha256']))
        assert all(m['files']==manifests[0]['files'] and m['splits']==manifests[0]['splits'] for m in manifests)
data=pd.DataFrame(rows)
assert not data.duplicated(['dataset','seed','generator','source','labeler','mode','metric']).any()
data.to_csv(OUT/'scores.csv',index=False)
snapshot=dict(checked_at_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    scopes=SCOPES,records=records,missing=missing,selected_expected=406,selected_completed=len(records),
    packages=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__),
    test_partitions=test_partitions,
    shared_seed_test_partitions={d:test_partitions[f'{d}/42']==test_partitions[f'{d}/43'] for d in SCOPES},
    worker_status={p.name:json.loads(p.read_text()) for p in (ROOT/'output/mnist_head_fixed_20261009').glob('worker-*.status.json')})
(OUT/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
# Aggregate this frozen, explicitly selected family using the existing builder.
frozen=OUT/'records';frozen.mkdir()
for name,payload in raw_files.items():
    (frozen/Path(name).name).write_bytes(payload)
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('result_builder',ROOT/'scripts/build_corrected_results.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
keys={(json.loads(p)['dataset'],json.loads(p)['train_option'],json.loads(p)['augment_option'],json.loads(p)['seed']) for p in raw_files.values()}
builder.expected_runs=lambda matrix,generators: keys
with io.StringIO() as log:
    from contextlib import redirect_stdout
    with redirect_stdout(log):
        builder.build(frozen,OUT/'aggregate',generators=('ctgan','tvae'))
    (OUT/'aggregation.log').write_text(log.getvalue())
with (OUT/'aggregate/per_run.csv').open() as stream:
    assert sum(1 for _ in csv.DictReader(stream))==len(records)
archive=io.BytesIO()
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,payload in raw_files.items():z.writestr('records/'+name,payload)
    for p in OUT.glob('*'):
        if p.is_file():z.write(p,p.name)
    for p in (OUT/'aggregate').glob('*'):
        if p.is_file():z.write(p,'aggregate/'+p.name)
print(base64.b64encode(archive.getvalue()).decode())
