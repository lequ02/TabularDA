"""Run every corrected configuration with seed 43."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ['CORRECTED_RUN_NAMESPACE'] = 'corrected_v2'
    cache = ROOT / '.cache' / 'corrected_v2' / 'seed_43'
    for name, folder in (('SCIKIT_LEARN_DATA', 'sklearn'),
                         ('XDG_CACHE_HOME', 'xdg'), ('MPLCONFIGDIR', 'matplotlib')):
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    from run_corrected_matrix import DATASETS, run_matrix
    run_matrix(DATASETS, (43,), 'all')


if __name__ == '__main__':
    main()
