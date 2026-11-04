"""Stage the Adult TabDDPM posterior-fix rerun: copy verified splits, link baselines, write launch.json."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path('/home/thuy/Research/minh_data_synth/TabularDA')
SNAPSHOT = REPO / '.cache/tabddpm_covertype_rerun_20261011/repo'
HERE = REPO / '.cache/tabddpm_adult_rerun_20261030'
NS = 'tabddpm_adult_500ep_fix_20261030_v1'
OLD = 'tabddpm_500ep_20261010_v1'
SEEDS = (42, 43)
EXPECTED_SRC = 'ab84b26b134265af688bbaff3a5c23d14050e6e8261eb2ac0c6192a42f15f7dc'

assert os.environ['CORRECTED_RUN_NAMESPACE'] == NS
sys.path[:0] = [str(SNAPSHOT / 'src'), str(SNAPSHOT / 'scripts')]
from run_corrected_matrix import methods_for  # noqa: E402
from modeling import constants  # noqa: E402
from modeling.run_record import source_sha256  # noqa: E402

assert constants.RUN_NAMESPACE == NS


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


now = lambda: datetime.now(ZoneInfo('America/Chicago')).isoformat()
expected = json.loads((HERE / 'expected_inputs.json').read_text())
for folder in (REPO / 'data' / NS, REPO / 'output' / NS, REPO / 'sdv trained model' / NS):
    if folder.exists():
        raise FileExistsError(f'Namespace already exists; preserve it: {folder}')
if source_sha256() != EXPECTED_SRC:
    raise RuntimeError('Frozen snapshot source hash changed')
if 'log_sum_exp_by_classes(unnormed_logprobs, self.slices_for_classes)' not in (
        SNAPSHOT / 'src/synthesize_data/tabddpm_core/gaussian_multinomial_diffsuion.py').read_text():
    raise RuntimeError('Posterior fix missing from snapshot')

inputs = {}
for seed in SEEDS:
    src_dir = REPO / 'data' / OLD / 'adult' / f'seed_{seed}'
    dst_dir = REPO / 'data' / NS / 'adult' / f'seed_{seed}'
    manifest = json.loads((src_dir / 'split_manifest.json').read_text())
    assert (manifest['dataset'], manifest['seed'], manifest['split_random_state']) == ('adult', seed, 42)
    ids = {split: manifest['splits'][split] for split in ('train', 'dev', 'test')}
    assert sum(map(len, ids.values())) == len(set().union(*map(set, ids.values())))
    names = ['split_manifest.json'] + [constants.split_name('adult', seed, split, view)
                                       for split in ('train', 'dev', 'test') for view in ('raw', 'onehot')]
    dst_dir.mkdir(parents=True)
    for name in names:
        source = src_dir / name
        want = expected[str(source)]
        if sha(source) != want:
            raise ValueError(f'Source input differs from recorded hash: {source}')
        shutil.copy2(source, dst_dir / name)
        if sha(dst_dir / name) != want:
            raise ValueError(f'Copied input differs: {dst_dir / name}')
        inputs[str(dst_dir / name)] = dict(sha256=want, copied_from=str(source))
    for split in ('train', 'dev', 'test'):
        for view in ('raw', 'onehot'):
            details = manifest['files'][split][view]
            assert details['rows'] == len(ids[split])
            assert sha(dst_dir / constants.split_name('adult', seed, split, view)) == details['sha256']
    inputs[f'seed_{seed}_rows'] = {split: len(ids[split]) for split in ids}

reused = {}
acc = REPO / 'output' / NS / 'adult' / 'acc'
acc.mkdir(parents=True)
for seed in SEEDS:
    name = constants.run_name('adult', seed, 'original', None) + '.run.json'
    original = REPO / 'output' / OLD / 'adult' / 'acc' / name
    if sha(original) != expected[str(original)]:
        raise ValueError(f'Baseline record changed: {original}')
    record = json.loads(original.read_text())
    assert (record['dataset'], record['seed'], record['train_option'], record['augment_option']) == ('adult', seed, 'original', None)
    assert record['selected_dev_epoch'] is not None
    for artifact, digest in record['artifact_sha256'].items():
        if sha(artifact) != digest:
            raise ValueError(f'Baseline artifact changed: {artifact}')
    (acc / name).symlink_to(original)
    reused[str(acc / name)] = dict(target=str(original), sha256=expected[str(original)],
                                   original_source_sha256=record['source_sha256'],
                                   status='passes_current_validity_checks (cleanup_review_20261013)',
                                   note='Real-only training does not use the TabDDPM posterior module.')

old_arms = sorted(p.name.removesuffix('.run.json') for p in (REPO / 'output' / OLD / 'adult' / 'acc').glob('adult_seed*_tabddpm_*.run.json'))
planned = sorted(constants.run_name('adult', seed, mode, method) for seed in SEEDS
                 for method in methods_for('adult', 'tabddpm') for mode in ('synthetic', 'mix'))
assert planned == old_arms and len(planned) == 52, (len(planned), len(old_arms))

workers = {
    'tabddpm-adult-fix-gen42': 'generation seed 42 (prepare-only check, full fit/sample/6 labelers, X-only fit/sample/6 labelers)',
    'tabddpm-adult-fix-gen43': 'generation seed 43 (same)',
    **{f'tabddpm-adult-fix-eval{i}': 'shared downstream pool: ready-gated, longest-first, atomic claims' for i in range(1, 5)},
}
launch = dict(
    prepared_at_chicago=now(), namespace=NS, replaces_held_namespace=OLD,
    reason='52 Adult TabDDPM records held: generators trained before the Oct 11 categorical posterior normalization fix (audit/TAB_DDPM_PATCHES.md).',
    source_snapshot=str(SNAPSHOT), main_source_sha256=source_sha256(),
    runner_sha256=sha(SNAPSHOT / 'scripts/run_corrected_matrix.py'),
    worker_sha256={name: sha(HERE / name) for name in ('gen_worker.py', 'eval_worker.py', 'launch.sh', 'prepare.py')},
    python=sys.executable,
    versions=subprocess.run([sys.executable, '-c', 'import torch,sklearn,xgboost,pandas,numpy;print(torch.__version__,sklearn.__version__,xgboost.__version__,pandas.__version__,numpy.__version__)'],
                            capture_output=True, text=True, check=True).stdout.strip(),
    dataset='adult', generator='tabddpm', seeds=list(SEEDS), epochs=500, batch_size=500, num_timesteps=1000,
    synthetic_rows_per_source=100000,
    downstream=dict(batchsize=128, lr=0.001, global_round=100, patience=30, early_stop_crit='loss', protocol='corrected/standard'),
    planned_generator_fits=4, planned_synthetic_tables=26, planned_downstream=54, planned_new_downstream=planned,
    reused_original=reused, inputs=inputs, workers=workers,
    logs=dict(workers=f'output/{NS}/logs/workers/<session>.log', stages=f'output/{NS}/logs/adult/seed_<seed>/',
              queue_state=str(HERE / 'state')),
)
(REPO / 'output' / NS / 'launch.json').write_text(json.dumps(launch, indent=1))
print(json.dumps({k: launch[k] for k in ('namespace', 'main_source_sha256', 'versions', 'planned_downstream')}, indent=1))
print('PLANNED NEW', len(planned))
