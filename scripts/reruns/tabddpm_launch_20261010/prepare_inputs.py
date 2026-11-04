from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

root = Path('/home/thuy/Research/minh_data_synth/TabularDA')
namespace = 'tabddpm_500ep_20261010_v1'
sources = {'adult': 'corrected_v2', 'census_kdd': 'corrected_v2', 'covertype': 'corrected_v2',
           'mnist12': 'mnist_head_fixed_20261009', 'mnist28': 'mnist_head_fixed_20261009',
           'news': 'news_log_v1', 'california_housing': 'housing_no_faker_20261009'}

def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()

result = {'namespace': namespace, 'inputs': []}
for dataset, source_namespace in sources.items():
    for seed in (42, 43):
        source = root / 'data' / source_namespace / dataset / f'seed_{seed}'
        path = source / 'split_manifest.json'
        manifest = json.loads(path.read_text())
        assert (manifest['dataset'], manifest['seed'], manifest['split_random_state']) == (dataset, seed, 42)
        ids = [source_id for split in ('train', 'dev', 'test') for source_id in manifest['splits'][split]]
        assert len(ids) == len(set(ids))
        files = {}
        for split in ('train', 'dev', 'test'):
            for view in ('raw', 'onehot'):
                name = f'{dataset}_seed{seed}_real_{split}_{view}.csv'
                expected = manifest['files'][split][view]
                assert expected['rows'] == len(manifest['splits'][split])
                assert digest(source / name) == expected['sha256']
                files[name] = expected['sha256']
        files['split_manifest.json'] = digest(path)
        destination = root / 'data' / namespace / dataset / f'seed_{seed}'
        destination.mkdir(parents=True, exist_ok=False)
        for name, expected in files.items():
            shutil.copy2(source / name, destination / name)
            assert digest(destination / name) == expected
        result['inputs'].append({'dataset': dataset, 'seed': seed, 'source': str(source),
            'destination': str(destination), 'sha256': files,
            'rows': {split: len(manifest['splits'][split]) for split in ('train', 'dev', 'test')},
            'updates_at_500_epochs': 500 * ((len(manifest['splits']['train']) + 499) // 500)})
        print('VERIFIED', dataset, seed, flush=True)
for seed in (42, 43):
    def split_ids(dataset):
        return json.loads((root / 'data' / namespace / dataset / f'seed_{seed}/split_manifest.json').read_text())['splits']
    assert split_ids('mnist12') == split_ids('mnist28')
result['verified_at_chicago'] = datetime.now(ZoneInfo('America/Chicago')).isoformat()
(root / '.cache/tabddpm_launch_20261010/prepared_inputs.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'verified_splits': len(result['inputs']), 'namespace': namespace}), flush=True)
