"""Resource-aware concurrent checkpoint recovery, preserving the active serial run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
PYTHON = '/home/thuy/miniconda3/envs/env/bin/python'
GIB = 1024**3
CONFIG = json.loads((BASE / 'parallel_config.json').read_text())
INVENTORY = json.loads((BASE / 'inventory.json').read_text())
ORDER = json.loads((BASE / 'queue_order.json').read_text())
HANDOVER = json.loads((BASE / 'parallel_handover.json').read_text())
lock = (BASE / 'parallel_scheduler.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
lock.write(str(os.getpid()) + '\n')
lock.flush()


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def write_json(name, value):
    path = BASE / name
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def job_dir(index):
    return BASE / 'jobs' / INVENTORY[index]['record_relative_path'].replace('/', '__').removesuffix('.run.json')


def read_result(index):
    result = json.loads((job_dir(index) / 'result.json').read_text())
    expected = INVENTORY[index]
    for key in ('record_relative_path', 'target_epoch', 'selection_metric', 'source_sha256'):
        if result[key] != expected[key]:
            raise ValueError(f'Completed result provenance mismatch: {index} {key}')
    if not result['original_checkpoint_tensors_exact'] or not result['test_ids_and_targets_match']:
        raise ValueError(f'Completed result verification failed: {index}')
    return result


def proc_stat(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return None
    fields = path.read_text().partition(') ')[2].split()
    return {'state': fields[0], 'start_ticks': int(fields[19]), 'exit_status': int(fields[49])}


def proc_rss(pid):
    path = Path(f'/proc/{pid}/status')
    try:
        status = path.read_text()
    except FileNotFoundError:
        # A worker can exit between polling it and sampling RAM; reserving its
        # entire estimated peak until the next poll is conservative.
        return 0
    values = [int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:')]
    return values[0] if values else 0


def resources():
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                       '--format=csv,noheader,nounits'], text=True).splitlines()
    used = {int(line.split(',')[0]): float(line.split(',')[1]) * 1024**2 for line in clients if line.strip()}
    return {'available_ram_bytes': available, 'free_gpu_bytes': float(gpu[0]) * 1024**2,
            'gpu_utilization_percent': int(gpu[1]), 'gpu_bytes_by_pid': used}


completed = set()
peaks = defaultdict(list)
for index in ORDER:
    if (job_dir(index) / 'result.json').exists():
        result = read_result(index)
        completed.add(index)
        peaks[(result['dataset'], result['mode'])].append(result['resource_usage']['own_peak_rss_kib'] * 1024)


def reserve_ram(index):
    item = INVENTORY[index]
    observed = peaks.get((item['dataset'], item['mode']))
    if observed is None:
        # A mode not yet observed uses the largest measured peak for its dataset.
        observed = [v for (dataset, _), values in peaks.items() if dataset == item['dataset'] for v in values]
    if not observed:
        raise ValueError(f'No measured memory evidence for {item["dataset"]}')
    return max(observed) * CONFIG['ram_peak_multiplier'] + CONFIG['ram_peak_padding_gib'] * GIB


adopted = HANDOVER['active_worker']
adopt_index = adopted['index']
controller = HANDOVER['legacy_controller']
controller_stat = proc_stat(controller['pid'])
if controller_stat is None or controller_stat['start_ticks'] != controller['start_ticks'] or controller_stat['state'] not in ('T', 't'):
    raise RuntimeError('Legacy controller is not the verified stopped controller')
worker_stat = proc_stat(adopted['pid'])
if worker_stat is None or worker_stat['start_ticks'] != adopted['start_ticks']:
    raise RuntimeError('Active worker identity changed during handover')
incomplete = {i for i in ORDER if job_dir(i).exists() and i not in completed}
if incomplete != ({adopt_index} - completed):
    raise RuntimeError(f'Unexpected incomplete jobs during handover: {incomplete}')
running = {adopt_index: {'pid': adopted['pid'], 'process': None, 'adopted': True,
                         'start_ticks': adopted['start_ticks'], 'log': adopted['log'],
                         'started_at_chicago': adopted['started_at_chicago']}}
if worker_stat['state'] != 'Z':
    os.sched_setaffinity(adopted['pid'], os.sched_getaffinity(0))
pending = [i for i in ORDER if i not in completed and i != adopt_index]
desired_workers = CONFIG['initial_workers']
rotation = 0
last_launch = 0.0
last_ramp = time.monotonic()
last_tables = 0.0
gpu_samples = deque(maxlen=max(1, CONFIG['ramp_interval_seconds'] // CONFIG['poll_seconds']))
failures = []
legacy_retired = False
print(f'{now()} PARALLEL HANDOVER completed={len(completed)} adopted_index={adopt_index} adopted_pid={adopted["pid"]}', flush=True)


def retire_legacy():
    global legacy_retired
    state = proc_stat(controller['pid'])
    if state is not None:
        if state['start_ticks'] != controller['start_ticks'] or state['state'] not in ('T', 't'):
            raise RuntimeError('Legacy controller identity/state changed before retirement')
        os.kill(controller['pid'], signal.SIGTERM)
        os.kill(controller['pid'], signal.SIGCONT)
    legacy_retired = True
    print(f'{now()} LEGACY CONTROLLER RETIRED pid={controller["pid"]}; completed worker preserved', flush=True)


def record_failure(index, code):
    failure = {'index': index, 'record': INVENTORY[index]['record_relative_path'],
               'returncode': code, 'log': running[index]['log'], 'failed_at_chicago': now(),
               'policy': 'stop new launches, preserve other active jobs, no automatic retry'}
    failures.append(failure)
    write_json('failure.json', {'failures': failures, **failures[0]})
    print(f'{now()} FAILED {json.dumps(failure)}', flush=True)


while pending or running:
    changed = False
    for index, worker in list(running.items()):
        if worker['adopted']:
            state = proc_stat(worker['pid'])
            if state is not None and state['start_ticks'] != worker['start_ticks']:
                raise RuntimeError('Adopted worker PID reused')
            if state is not None and state['state'] != 'Z':
                continue
            if state is None:
                raise RuntimeError('Adopted worker disappeared before its stopped parent reaped it')
            code = os.waitstatus_to_exitcode(state['exit_status'])
            retire_legacy()
        else:
            code = worker['process'].poll()
            if code is None:
                continue
        if code != 0:
            record_failure(index, code)
        else:
            result = read_result(index)
            completed.add(index)
            peaks[(result['dataset'], result['mode'])].append(result['resource_usage']['own_peak_rss_kib'] * 1024)
            print(f'{now()} DONE index={index} dataset={result["dataset"]} completed={len(completed)}/{len(INVENTORY)}', flush=True)
        del running[index]
        changed = True
    if changed:
        write_json('progress.json', {'completed': len(completed), 'planned': len(INVENTORY), 'updated_at_chicago': now()})
        if time.monotonic() - last_tables >= 60 or not running:
            subprocess.run([PYTHON, str(BASE / 'build_reports.py'), '--tables-only'], check=True)
            last_tables = time.monotonic()
    state = resources()
    gpu_samples.append(state['gpu_utilization_percent'])
    if time.monotonic() - last_ramp >= CONFIG['ramp_interval_seconds']:
        mean_gpu = sum(gpu_samples) / len(gpu_samples)
        if mean_gpu < CONFIG['target_gpu_utilization_percent'] and desired_workers < CONFIG['max_workers']:
            desired_workers += 1
            print(f'{now()} RAMP desired_workers={desired_workers} mean_gpu_utilization={mean_gpu:.1f}', flush=True)
        last_ramp = time.monotonic()
    launched = False
    if not failures and pending and len(running) < desired_workers and time.monotonic() - last_launch >= CONFIG['startup_spacing_seconds']:
        ram_floor = max(CONFIG['minimum_available_ram_gib'], 10 if adopt_index in running else 0) * GIB
        outstanding_ram = sum(max(0, reserve_ram(i) - proc_rss(w['pid'])) for i, w in running.items())
        outstanding_gpu = sum(max(0, CONFIG['reserved_gpu_gib_per_worker'] * GIB - state['gpu_bytes_by_pid'].get(w['pid'], 0))
                              for w in running.values())
        dataset_cycle = CONFIG['dataset_rotation']
        for step in range(len(dataset_cycle)):
            position = (rotation + step) % len(dataset_cycle)
            dataset = dataset_cycle[position]
            candidates = [i for i in pending if INVENTORY[i]['dataset'] == dataset]
            for index in candidates:
                if state['available_ram_bytes'] - outstanding_ram - reserve_ram(index) < ram_floor:
                    continue
                if state['free_gpu_bytes'] - outstanding_gpu - CONFIG['reserved_gpu_gib_per_worker'] * GIB < CONFIG['minimum_free_gpu_gib'] * GIB:
                    continue
                item = INVENTORY[index]
                name = item['record_relative_path'].replace('/', '__').removesuffix('.run.json')
                log = BASE / 'logs' / (name + '.log')
                command = [PYTHON, '-u', str(BASE / 'recover_one_parallel.py'), str(index)]
                with log.open('x') as stream:
                    process = subprocess.Popen(command, cwd=BASE, stdout=stream, stderr=subprocess.STDOUT)
                running[index] = {'pid': process.pid, 'process': process, 'adopted': False, 'log': str(log),
                                  'command': command, 'started_at_chicago': now()}
                pending.remove(index)
                last_launch = time.monotonic()
                rotation = (position + 1) % len(dataset_cycle)
                print(f'{now()} START index={index} pid={process.pid} dataset={dataset} mode={item["mode"]} '
                      f'epoch={item["target_epoch"]} reserve_ram_gib={reserve_ram(index)/GIB:.2f} active={len(running)}', flush=True)
                launched = True
                break
            if launched:
                break
    active = [{'index': i, 'item': INVENTORY[i], **{k: v for k, v in w.items() if k != 'process'},
               'ram_reservation_gib': reserve_ram(i) / GIB} for i, w in running.items()]
    write_json('active.json', {'scheduler': 'parallel', 'workers': active, 'desired_workers': desired_workers,
                              'maximum_workers': CONFIG['max_workers'], 'updated_at_chicago': now()})
    telemetry = {**state, 'active_workers': len(running), 'desired_workers': desired_workers,
                 'completed': len(completed), 'pending': len(pending), 'sampled_at_chicago': now()}
    with (BASE / 'parallel_telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(telemetry) + '\n')
    if failures and not running:
        break
    if pending and not running and not launched and not failures:
        print(f'{now()} WAITING FOR RESOURCE HEADROOM; no job has been started or retried', flush=True)
    if pending or running:
        time.sleep(CONFIG['poll_seconds'])

if failures:
    raise RuntimeError(f'Parallel queue stopped after failed jobs: {failures}')
if not legacy_retired or len(completed) != len(INVENTORY):
    raise RuntimeError('Completion checks failed')
subprocess.run([PYTHON, str(BASE / 'build_reports.py')], check=True)
bundle = BASE / 'completed_reports.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in (BASE / 'reports').iterdir():
        archive.write(path, path.relative_to(BASE))
    for path in (BASE / 'jobs').glob('*/result.json'):
        archive.write(path, path.relative_to(BASE))
    for name in ('inventory_summary.json', 'inventory.json', 'launch.json', 'README.md',
                 'recover_one.py', 'recover_one_parallel.py', 'recovery_queue.py', 'parallel_queue.py',
                 'parallel_config.json', 'parallel_handover.json', 'build_reports.py', 'launch.sh', 'launch_parallel.sh'):
        archive.write(BASE / name, name)
write_json('delivery.json', {'bundle': bundle.name, 'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()})
write_json('completed.json', {'completed_at_chicago': now(), 'jobs': len(completed), 'scheduler': 'parallel'})
print(f'{now()} ALL {len(completed)} EVALUATIONS AND BOTH REPORTS COMPLETE', flush=True)
