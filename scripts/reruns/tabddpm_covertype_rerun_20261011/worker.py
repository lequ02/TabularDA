from pathlib import Path
import os
import sys

snapshot = Path(__file__).resolve().parent / 'repo'
sys.path[:0] = [str(snapshot / 'src'), str(snapshot / 'scripts')]
from run_corrected_matrix import run_experiment, run_classifier, methods_for

seed = int(sys.argv[1])
assert seed in (42, 43)
namespace = 'tabddpm_covertype_500ep_fix_20261011_v1'
assert os.environ['CORRECTED_RUN_NAMESPACE'] == namespace
output = snapshot / 'output' / namespace
print(f'WORKER Covertype seed {seed}; corrected posterior; 500 epochs', flush=True)
if seed == 43:
    run_classifier(output, 'covertype', seed, 'original', None, False, 'corrected')
else:
    print('Reusing the validated seed-42 real-only baseline; see launch.json', flush=True)
run_experiment(seed=seed, datasets=['covertype'], generators=('tabddpm',),
               generator_epochs=500, namespace=namespace, stage='generators')
for method in methods_for('covertype', 'tabddpm'):
    for mode in ('synthetic', 'mix'):
        run_classifier(output, 'covertype', seed, mode, method, False, 'corrected')
print(f'QUEUE COMPLETE Covertype seed {seed}', flush=True)
