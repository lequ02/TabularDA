"""Collect generator fit wall-clock times from remote logs (run remotely with the env python).

CTGAN/TVAE: final tqdm elapsed of each 500/500 bar in corrected_v2 *_generate[.retry].log;
first bar = full (X,Y), second = X-only (order confirmed by checkpoint mtimes).
TabDDPM (no timed logs): full = full.pkl mtime - worker START; X-only = xonly.pkl mtime -
last full-source artifact mtime before it. Prints CSV to stdout.
"""
import csv, re, sys, datetime as dt
from pathlib import Path
root = Path('/home/thuy/Research/minh_data_synth/TabularDA')
w = csv.writer(sys.stdout)
w.writerow(['dataset', 'seed', 'generator', 'construction', 'fit_seconds', 'source'])
to_s = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(':'))))
logs = root / 'output/corrected_v2/logs'
for gen in ('ctgan', 'tvae'):
    for d in sorted(logs.glob('*/seed_*')):
        ds, seed = d.parent.name, int(d.name.split('_')[1])
        for name in (f'{ds}_seed{seed}_{gen}_generate.retry.log', f'{ds}_seed{seed}_{gen}_generate.log'):
            p = d / name
            if not p.exists():
                continue
            bars = re.findall(r'500/500 \[([0-9:]+)<00:00', p.read_text(errors='replace').replace('\r', '\n'))
            bars = [b for i, b in enumerate(bars) if i == 0 or b != bars[i - 1]]
            if len(bars) == 2:
                for cons, b in zip(('full', 'xonly'), bars):
                    w.writerow([ds, seed, gen, cons, to_s(b), str(p.relative_to(root))])
                break
nss = {'adult': 'tabddpm_adult_500ep_fix_20261030_v1', 'census_kdd': 'tabddpm_census_kdd_500ep_fix_20261011_v1',
       'covertype': 'tabddpm_covertype_500ep_fix_20261011_v1'}
for ns_dir in sorted((root / 'output').glob('tabddpm_*')):
    ns = ns_dir.name
    starts = {}
    for wl in (ns_dir / 'logs/workers').glob('*.log'):
        for line in open(wl, errors='replace'):
            m = re.match(r'(\S+) START .*--dataset (\S+) --seed (\d+) --generator TABDDPM', line)
            if m and '--prepare-only' not in line:
                starts[(m.group(2), int(m.group(3)))] = dt.datetime.fromisoformat(m.group(1)).timestamp()
    for full in sorted((root / 'sdv trained model' / ns).glob('*/seed_*/*_tabddpm_full.pkl')):
        ds, seed = full.parent.parent.name, int(full.parent.name.split('_')[1])
        if nss.get(ds, 'tabddpm_500ep_20261010_v1') != ns or (ds, seed) not in starts:
            continue
        xo = full.with_name(full.name.replace('_full.pkl', '_xonly.pkl'))
        tf, tx = full.stat().st_mtime, xo.stat().st_mtime
        files = list((root / 'data' / ns / ds / f'seed_{seed}').glob('*tabddpm*')) + list(full.parent.glob('*'))
        x_start = max(p.stat().st_mtime for p in files if tf <= p.stat().st_mtime < tx - 1 and 'xonly' not in p.name)
        w.writerow([ds, seed, 'tabddpm', 'full', round(tf - starts[(ds, seed)]), f'mtime:{ns}'])
        w.writerow([ds, seed, 'tabddpm', 'xonly', round(tx - x_start), f'mtime:{ns}'])
