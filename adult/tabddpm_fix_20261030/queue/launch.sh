#!/bin/bash
# Launch the Adult TabDDPM posterior-fix rerun: 2 generation workers + 4 pooled downstream workers.
set -euo pipefail
REPO=/home/thuy/Research/minh_data_synth/TabularDA
HERE=$REPO/.cache/tabddpm_adult_rerun_20261030
NS=tabddpm_adult_500ep_fix_20261030_v1
PY=/home/thuy/miniconda3/envs/env/bin/python
WLOG=$REPO/output/$NS/logs/workers
mkdir -p "$WLOG" "$HERE/state"
for s in tabddpm-adult-fix-gen42 tabddpm-adult-fix-gen43 tabddpm-adult-fix-eval{1,2,3,4}; do
  if tmux has-session -t "$s" 2>/dev/null; then echo "session $s exists" >&2; exit 1; fi
done
for seed in 42 43; do
  s=tabddpm-adult-fix-gen$seed
  tmux new-session -d -s "$s" "cd $HERE && CORRECTED_RUN_NAMESPACE=$NS $PY -u gen_worker.py $seed > $WLOG/$s.log 2>&1; echo \$? > $HERE/state/gen_seed$seed.exit; exec bash"
done
for i in 1 2 3 4; do
  s=tabddpm-adult-fix-eval$i
  tmux new-session -d -s "$s" "cd $HERE && CORRECTED_RUN_NAMESPACE=$NS nice -n 5 $PY -u eval_worker.py eval$i > $WLOG/$s.log 2>&1; echo \$? > $HERE/state/eval$i.exit; exec bash"
done
tmux ls | grep tabddpm-adult-fix
