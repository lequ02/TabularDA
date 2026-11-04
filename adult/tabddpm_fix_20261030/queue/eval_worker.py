"""Shared downstream pool for the Adult TabDDPM rerun.

Each worker claims one ready arm at a time (atomic O_EXCL claim file), longest
expected runtime first, so the pool drains evenly and all workers finish
together. An arm is ready once its synthetic table's quality.json exists; the
generator writes it after the CSV. Launches are spaced and gated on available
RAM/GPU memory. Failures stop this worker with the real exception; no retries.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

snapshot = Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_covertype_rerun_20261011/repo')
sys.path[:0] = [str(snapshot / 'src'), str(snapshot / 'scripts')]
namespace = 'tabddpm_adult_500ep_fix_20261030_v1'
assert os.environ['CORRECTED_RUN_NAMESPACE'] == namespace
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'  # same as run_matrix
from run_corrected_matrix import methods_for, run_classifier  # noqa: E402
from modeling import constants  # noqa: E402
import torch  # noqa: E402

assert constants.RUN_NAMESPACE == namespace
if not torch.cuda.is_available():
    raise RuntimeError('Production experiments require remote CUDA')

STATE = Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_adult_rerun_20261030/state')
OUTPUT = snapshot / 'output' / namespace
DATA = snapshot / 'data' / namespace / 'adult'
MIN_RAM_MIB, MIN_GPU_MIB, LAUNCH_GAP_S, POLL_S = 6144, 1536, 60, 30
worker = sys.argv[1]
(STATE / 'claims').mkdir(parents=True, exist_ok=True)
(STATE / 'failed').mkdir(exist_ok=True)


def log(message):
    print(f"{datetime.now(ZoneInfo('America/Chicago')).isoformat()} {worker} {message}", flush=True)


def expected_minutes(method, mode):  # from the Oct 10-11 Adult TabDDPM stage logs
    if method == 'tabddpm':
        return 17 if mode == 'mix' else 8
    return 6 if mode == 'mix' else 4.5


ORDER = list(methods_for('adult', 'tabddpm'))
TASKS = sorted(((seed, method, mode) for seed in (42, 43) for method in ORDER for mode in ('synthetic', 'mix')),
               key=lambda t: (-expected_minutes(t[1], t[2]), ORDER.index(t[1]), t[0]))
assert len(TASKS) == 52


def arm(task):
    seed, method, mode = task
    return constants.run_name('adult', seed, mode, method)


def quality_path(task):
    seed, method, _ = task
    return (DATA / f'seed_{seed}' / constants.synthetic_name('adult', seed, method)).with_suffix('.quality.json')


def ready(task):
    quality = quality_path(task)
    return quality.is_file() and time.time() - quality.stat().st_mtime > 10


def generation_exit(seed):
    path = STATE / f'gen_seed{seed}.exit'
    return path.read_text().strip() if path.exists() else None


def resources_ok():
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    ram = int(meminfo['MemAvailable'].split()[0]) // 1024
    gpu = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True).stdout.split()[0])
    return ram >= MIN_RAM_MIB and gpu >= MIN_GPU_MIB, ram, gpu


def claim_next():
    """Under a global lock: pick the longest ready unclaimed arm, if resources and spacing allow."""
    with open(STATE / 'claim.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        unclaimed = [t for t in TASKS if not (STATE / 'claims' / arm(t)).exists()]
        if not unclaimed:
            return 'done', None
        for seed in (42, 43):
            code = generation_exit(seed)
            if code is not None and code != '0' and all(t[0] == seed for t in unclaimed):
                raise RuntimeError(f'Seed {seed} generation exited {code}; its arms cannot run')
            if code == '0':
                missing = [arm(t) for t in unclaimed if t[0] == seed and not quality_path(t).is_file()]
                if missing:
                    raise RuntimeError(f'Seed {seed} generation finished without tables for {missing}')
        candidates = [t for t in unclaimed if ready(t)]
        if not candidates:
            return 'wait', None
        last = STATE / 'last_launch'
        if last.exists() and time.time() - float(last.read_text()) < LAUNCH_GAP_S:
            return 'wait', None
        ok, ram, gpu = resources_ok()
        if not ok:
            log(f'waiting for resources: MemAvailable={ram} MiB, GPU free={gpu} MiB')
            return 'wait', None
        task = candidates[0]
        fd = os.open(STATE / 'claims' / arm(task), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps(dict(worker=worker, claimed_at=time.time(), ram_mib=ram, gpu_free_mib=gpu)).encode())
        os.close(fd)
        last.write_text(str(time.time()))
        return 'run', task


log(f'start; {len(TASKS)} arms in pool')
while True:
    status, task = claim_next()
    if status == 'done':
        log('POOL DRAINED; worker exiting')
        break
    if status == 'wait':
        time.sleep(POLL_S)
        continue
    log(f'RUN {arm(task)}')
    started = time.time()
    try:
        run_classifier(OUTPUT, 'adult', task[0], task[2], task[1], False, 'corrected')
    except BaseException:
        (STATE / 'failed' / arm(task)).write_text(traceback.format_exc())
        log(f'FAILED {arm(task)}')
        raise
    log(f'DONE {arm(task)} in {(time.time() - started) / 60:.1f} min')
