"""Freeze only our dispatcher; its current recovery continues without interruption."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime
from zoneinfo import ZoneInfo

base = Path(__file__).resolve().parent
controller_pid = 2862265
worker_pid = 3581067
expected = {
    'recover_one.py': '7a9a3d3287581626f94143f080116b637a46ae0039e0441421e3ca0dd08533b6',
    'recovery_queue.py': '89e1531630f7c37e7babe041ad27e8303d7c76aaba8583dc6f8cdc1a52c30a5d',
}
if (base / 'parallel_handover.json').exists():
    raise RuntimeError('A handover already exists; inspect it before doing anything')
for name, sha in expected.items():
    if hashlib.sha256((base / name).read_bytes()).hexdigest() != sha:
        raise ValueError(f'Deployed source changed: {name}')
active = json.loads((base / 'active.json').read_text())
index = active['index']
controller_command = Path(f'/proc/{controller_pid}/cmdline').read_bytes().split(b'\0')
worker_command = Path(f'/proc/{worker_pid}/cmdline').read_bytes().split(b'\0')
if b'recovery_queue.py' not in controller_command:
    raise RuntimeError('Controller PID does not belong to our original queue')
if str(base / 'recover_one.py').encode() not in worker_command or str(index).encode() not in worker_command:
    raise RuntimeError('Worker PID does not match the active recovery')
children = Path(f'/proc/{controller_pid}/task/{controller_pid}/children').read_text().split()
if children != [str(worker_pid)]:
    raise RuntimeError(f'Unexpected children of original queue: {children}')

def stat(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().partition(') ')[2].split()
    return {'state': fields[0], 'start_ticks': int(fields[19])}

controller_before = stat(controller_pid)
worker_before = stat(worker_pid)
snapshot = {
    'handover_at_chicago': datetime.now(ZoneInfo('America/Chicago')).isoformat(),
    'legacy_controller': {'pid': controller_pid, **controller_before,
                          'command': [x.decode() for x in controller_command if x]},
    'active_worker': {'pid': worker_pid, **worker_before, 'index': index,
                      'log': active['log'], 'started_at_chicago': active['started_at_chicago'],
                      'command': [x.decode() for x in worker_command if x]},
    'completed_before_handover': json.loads((base / 'progress.json').read_text()),
    'preserved_source_sha256': expected,
    'new_source_sha256': {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                          for name in ('parallel_queue.py', 'parallel_config.json', 'recover_one_parallel.py')},
    'other_gpu_experiment_pids': [2898355, 2969494, 2969463],
    'policy': 'SIGSTOP dispatcher only; current worker continues; new scheduler retires dispatcher after its worker exits',
}
(base / 'parallel_handover_before_signal.json').write_text(json.dumps(snapshot, indent=2) + '\n')
os.kill(controller_pid, signal.SIGSTOP)
for _ in range(100):
    after = stat(controller_pid)
    if after['start_ticks'] != controller_before['start_ticks']:
        raise RuntimeError('Controller PID changed during handover')
    if after['state'] in ('T', 't'):
        break
    time.sleep(.05)
else:
    raise RuntimeError('Dispatcher did not stop')
snapshot['legacy_controller']['state_after_handover'] = after['state']
snapshot['active_worker']['state_after_handover'] = stat(worker_pid)['state']
temporary = base / 'parallel_handover.json.tmp'
temporary.write_text(json.dumps(snapshot, indent=2) + '\n')
temporary.replace(base / 'parallel_handover.json')
print(json.dumps(snapshot, indent=2))
