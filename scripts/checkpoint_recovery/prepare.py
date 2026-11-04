import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RETAINED = Path('G:/SummerResearch_Retained_20261013_2221')
hashes = {}
with (ROOT / 'audit/cleanup_review_20261013/saved_run_validity_evidence.jsonl').open() as stream:
    for line in stream:
        item = json.loads(line)
        if item['kind'] == 'artifact' and item.get('sha256'):
            hashes[item['path']] = item['sha256']
(HERE / 'input_hashes.json').write_text(json.dumps(hashes))
with zipfile.ZipFile(HERE / 'runtime.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for source in (RETAINED / 'source/current/src').rglob('*.py'):
        archive.write(source, source.relative_to(RETAINED / 'source/current').as_posix())
for mode in ('synthetic', 'mix'):
    shutil.copyfile(RETAINED / f'comparisons/{mode}.csv', HERE / f'{mode}.csv')
shutil.copyfile(ROOT / 'scripts/plot_recent_comparison.py', HERE / 'plot_recipe.py')
inventory = json.loads((HERE / 'inventory.json').read_text())
# All original test evaluations are repeated. Recovering changed states requires
# training prefixes; unchanged states are evaluated first for quick coverage.
ordering = sorted(range(len(inventory)), key=lambda i: (not inventory[i]['same_checkpoint'],
                  ['adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28', 'news', 'california_housing'].index(inventory[i]['dataset']),
                  inventory[i]['replay_epochs']))
(HERE / 'queue_order.json').write_text(json.dumps(ordering))
pilots = []
for dataset in ('adult', 'census_kdd', 'covertype', 'mnist12', 'mnist28', 'news', 'california_housing'):
    group = [i for i in ordering if inventory[i]['dataset'] == dataset]
    pilots.append(min(group, key=lambda i: (inventory[i]['replay_epochs'] if not inventory[i]['same_checkpoint'] else 0)))
    changed = [i for i in group if not inventory[i]['same_checkpoint']]
    if changed:
        pilots.append(min(changed, key=lambda i: inventory[i]['replay_epochs']))
(HERE / 'pilot_order.json').write_text(json.dumps(list(dict.fromkeys(pilots))))
with zipfile.ZipFile(HERE / 'payload.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('inventory.json', 'inventory_summary.json', 'files_index.json', 'input_hashes.json',
                 'runtime.zip', 'recover_one.py', 'recovery_queue.py', 'queue_order.json', 'pilot_order.json',
                 'build_reports.py', 'synthetic.csv', 'mix.csv', 'plot_recipe.py'):
        archive.write(HERE / name, name)
print(json.dumps({'jobs': len(inventory), 'pilot_jobs': pilots,
                  'replay_jobs': sum(not x['same_checkpoint'] for x in inventory),
                  'payload_sha256': hashlib.sha256((HERE / 'payload.zip').read_bytes()).hexdigest()}, indent=2))
