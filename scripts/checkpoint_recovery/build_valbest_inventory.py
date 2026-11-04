"""Build a best-validation replay inventory for a freshly run output namespace.

Follows inventory.py (the October 15 historical inventory): each in-scope record selects the
earliest epoch with the maximum development metric in its saved history. Scope matches the
reported validation-best table: the real-only baseline plus generated targets and full/X-only
RF, XGB and DNN relabeling. Writes inventory.json and recover_one_fresh.py into --out, which
must be two levels below the repository root (for example output/valbest_<namespace>).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from modeling.constants import method_parts

# Adult uses binary F1, the multiclass tasks and weighted Census KDD macro F1, regression R2.
METRIC = {'adult': 'f1_binary', 'census_kdd': 'f1_macro', 'covertype': 'f1_macro',
          'mnist12': 'f1_macro', 'mnist28': 'f1_macro', 'news': 'r2', 'california_housing': 'r2'}
LABELERS = {'generated', 'rf', 'xgb', 'dnn'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def item_for(record_path, root=ROOT):
    record = json.loads(record_path.read_text())
    if record['augment_option'] is None or record['train_option'] == 'original':
        generator, method = 'real', None
    else:
        generator, _, labeler = method_parts(record['augment_option'])
        if labeler not in LABELERS:
            return None
        method = record['augment_option']
    metric = METRIC[record['dataset']]
    history_path = record_path.with_name(record_path.name.removesuffix('.run.json') + '.epochs.csv')
    with history_path.open(newline='') as stream:
        history = list(csv.DictReader(stream))
    target_epoch = int(max(history, key=lambda x: float(x['dev_' + metric]))['global_round'])
    original_epoch = int(record['selected_dev_epoch'])
    return {
        'record_relative_path': record_path.relative_to(root).as_posix(),
        'dataset': record['dataset'], 'seed': int(record['seed']),
        'mode': record['train_option'], 'generator': generator, 'method': method,
        'original_selection_metric': record['selection_metric'], 'original_epoch': original_epoch,
        'selection_metric': metric, 'target_epoch': target_epoch, 'history_epochs': len(history),
        'replay_epochs': max(target_epoch, original_epoch), 'same_checkpoint': target_epoch == original_epoch,
        'source_sha256': record['source_sha256'],
        'expected_sha256': {p.relative_to(root).as_posix(): digest(p) for p in (record_path, history_path)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent.parent != ROOT:
        parser.error('--out must be two levels below the repository root, e.g. output/valbest_<namespace>')
    if out.exists():
        parser.error('Choose a new --out directory')
    records = sorted((ROOT / 'output' / args.namespace).rglob('*.run.json'))
    inventory = [item for item in map(item_for, records) if item is not None]
    if not inventory:
        raise ValueError(f'No in-scope run records under output/{args.namespace}')
    out.mkdir()
    (out / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    shutil.copy2(Path(__file__).with_name('recover_one_fresh.py'), out / 'recover_one_fresh.py')
    print(json.dumps({'jobs': len(inventory), 'replays': sum(not x['same_checkpoint'] for x in inventory),
                      'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
