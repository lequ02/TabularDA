"""Three persistent lanes with an explicit primary-before-supplementary barrier."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/thuy/Research/minh_data_synth/TabularDA')
OUTPUT = ROOT / 'output/mnist_head_fixed_20261009'
RUNNER = Path(__file__).with_name('run_plan.py')


def worker(lane):
    status_path = OUTPUT / f'worker-{lane}.status.json'
    status = dict(lane=lane, pid=os.getpid(), primary_complete=False,
                  labels_complete=False, supplementary_complete=False)

    def update(phase):
        status.update(phase=phase, updated_chicago=datetime.now(ZoneInfo('America/Chicago')).isoformat())
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)
        print(status['updated_chicago'], phase, flush=True)

    def execute(phase, arguments):
        update(phase)
        result = subprocess.run([sys.executable, '-u', str(RUNNER), *arguments], cwd=ROOT)
        if result.returncode:
            status['returncode'] = result.returncode
            update('failed')
            result.check_returncode()

    def wait_for(flag, lanes):
        while True:
            ready = True
            for peer in lanes:
                path = OUTPUT / f'worker-{peer}.status.json'
                if not path.exists():
                    ready = False
                    continue
                state = json.loads(path.read_text())
                if state['phase'] == 'failed':
                    raise RuntimeError(f'Worker {peer} failed; inspect its worker log')
                if not state[flag]:
                    os.kill(state['pid'], 0)  # A dead peer is an error, not a completed phase.
                    ready = False
            if ready:
                return
            time.sleep(15)

    with (OUTPUT / f'worker-{lane}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute('primary', ['train', '--phase', 'primary', '--lane', str(lane)])
        status['primary_complete'] = True
        update('waiting_for_primary')
        wait_for('primary_complete', (1, 2, 3))
        if lane == 1:
            execute('labels', ['labels'])
            status['labels_complete'] = True
            update('labels_complete')
        else:
            update('waiting_for_labels')
            wait_for('labels_complete', (1,))
            status['labels_complete'] = True
        execute('supplementary', ['train', '--phase', 'supplementary', '--lane', str(lane)])
        status['supplementary_complete'] = True
        update('supplementary_complete')
        if lane == 1:
            wait_for('supplementary_complete', (1, 2, 3))
            execute('verify_all', ['verify'])
        update('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or not ROOT.is_dir():
        parser.error('Run only on the research server')
    worker(args.lane)
