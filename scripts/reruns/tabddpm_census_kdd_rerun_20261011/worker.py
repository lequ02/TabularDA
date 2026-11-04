from pathlib import Path
import os
import sys

snapshot = Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_covertype_rerun_20261011/repo')
sys.path[:0] = [str(snapshot / 'src'), str(snapshot / 'scripts')]
from run_corrected_matrix import run_experiment, run_classifier, methods_for

namespace = 'tabddpm_census_kdd_500ep_fix_20261011_v1'
assert os.environ['CORRECTED_RUN_NAMESPACE'] == namespace
output = snapshot / 'output' / namespace
for seed in (42, 43):
    print(f'WORKER Census KDD seed {seed}; corrected posterior; 500 epochs', flush=True)
    if seed == 43:
        run_classifier(output, 'census_kdd', seed, 'original', None, False, 'corrected')
    else:
        print('Reusing the validated seed-42 weighted real-only baseline; see launch.json', flush=True)
    run_experiment(seed=seed, datasets=['census_kdd'], generators=('tabddpm',),
                   generator_epochs=500, namespace=namespace, stage='generators')
    for method in methods_for('census_kdd', 'tabddpm'):
        for mode in ('synthetic', 'mix'):
            run_classifier(output, 'census_kdd', seed, mode, method, False, 'corrected')
print('QUEUE COMPLETE Census KDD seeds 42 and 43', flush=True)
