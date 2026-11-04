"""One seed's Adult TabDDPM generation (full then X-only) on the corrected frozen snapshot."""
import os
import sys
from pathlib import Path

snapshot = Path('/home/thuy/Research/minh_data_synth/TabularDA/.cache/tabddpm_covertype_rerun_20261011/repo')
sys.path[:0] = [str(snapshot / 'src'), str(snapshot / 'scripts')]
from run_corrected_matrix import run_experiment  # noqa: E402

namespace = 'tabddpm_adult_500ep_fix_20261030_v1'
assert os.environ['CORRECTED_RUN_NAMESPACE'] == namespace
seed = int(sys.argv[1])
assert seed in (42, 43)
print(f'WORKER Adult TabDDPM generation seed {seed}; corrected posterior; 500 epochs', flush=True)
run_experiment(seed=seed, datasets=['adult'], generators=('tabddpm',),
               generator_epochs=500, namespace=namespace, stage='generators')
print(f'GENERATION COMPLETE Adult seed {seed}', flush=True)
