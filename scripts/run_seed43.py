"""Run every corrected configuration with seed 43 with available GPUs visible."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, help='Restrict visibility to one GPU')
    args = parser.parse_args()

    os.environ['CORRECTED_RUN_NAMESPACE'] = 'corrected_v2'
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
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
