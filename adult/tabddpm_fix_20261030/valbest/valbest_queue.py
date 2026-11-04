"""Pooled dev-F1 checkpoint replays for the Adult TabDDPM posterior-fix rerun (longest first, no retries)."""
import fcntl, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
BASE = Path(__file__).resolve().parent
PY = '/home/thuy/miniconda3/envs/env/bin/python'
INV = json.loads((BASE / 'inventory.json').read_text())
worker = sys.argv[1]
(BASE / 'claims').mkdir(exist_ok=True)
log = lambda m: print(f"{datetime.now(ZoneInfo('America/Chicago')).isoformat()} {worker} {m}", flush=True)

def ok():
    ram = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')) // 1024
    gpu = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True).stdout.split()[0])
    return ram >= 6144 and gpu >= 3584, ram, gpu

while True:
    with open(BASE / 'claim.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo = [i for i, item in enumerate(INV) if not (BASE / 'claims' / str(i)).exists()]
        if not todo:
            log('POOL DRAINED'); break
        good, ram, gpu = ok()
        last = BASE / 'last_launch'
        if not good or (last.exists() and time.time() - float(last.read_text()) < 20):
            index = None
        else:
            index = todo[0]
            (BASE / 'claims' / str(index)).write_text(worker)
            last.write_text(str(time.time()))
    if index is None:
        time.sleep(15); continue
    name = Path(INV[index]['record_relative_path']).name
    log(f'RUN {index} {name} replay_epochs={INV[index]["replay_epochs"]} ram={ram} gpu_free={gpu}')
    with open(BASE / 'logs' / f'{index:02d}.log', 'w') as out:
        code = subprocess.run(['nice', '-n', '10', PY, '-I', '-u', str(BASE / 'recover_one_adult_fix.py'), str(index)],
                              stdout=out, stderr=subprocess.STDOUT, cwd=BASE).returncode
    if code:
        log(f'FAILED {index} {name} exit={code}'); sys.exit(code)
    log(f'DONE {index} {name}')
