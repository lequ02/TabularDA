#!/bin/bash
set -euo pipefail
queue_root=/home/thuy/Research/minh_data_synth/TabularDA/output/adult_devf1_audit_20261015_v1
cd "$queue_root"
for run_id in adult_seed42_ctgan_full_xgb_mix adult_seed43_ctgan_full_rf_synthetic adult_seed42_ctgan_xonly_xgb_mix
do
    printf 'START %s %s\n' "$(date -Is)" "$run_id"
    nice -n 19 taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/thuy/miniconda3/envs/env/bin/python -u recover_one.py "jobs/$run_id" > "jobs/$run_id/worker.log" 2>&1
    printf 'COMPLETE %s %s\n' "$(date -Is)" "$run_id"
done
/home/thuy/miniconda3/envs/env/bin/python - <<'PY'
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path('.')
selected=json.loads((root/'selected_candidates.json').read_text())
results=[]
for job in selected['jobs']:
    name=Path(job['record_relative_path']).name.removesuffix('.run.json')
    results.append(json.loads((root/'jobs'/name/'result.json').read_text()))
(root/'completed.json').write_text(json.dumps({'completed_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'selection_rule':selected['selection_rule'],'results':results},indent=2)+'\n')
print('ALL THREE CHECKPOINT EVALUATIONS COMPLETE')
PY
