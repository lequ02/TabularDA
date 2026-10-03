"""Run remaining seed-42 datasets on the research server."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CORRECTED_RUN_NAMESPACE', 'corrected_v2')
os.environ.update(PYTHONUNBUFFERED='1',
                  MPLBACKEND='Agg', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  MKL_NUM_THREADS='2')


def run(*args, cwd=ROOT):
    print('Starting:', ' '.join(args), flush=True)
    subprocess.run([sys.executable, '-u', *args], cwd=cwd, check=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--resume-mnist28']:
        import pandas as pd
        sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/synthesize_data')]
        from create_synthetic_data.mnist28 import CreateSyntheticDataMnist28
        from synthesizer import load_synthesizer
        job = CreateSyntheticDataMnist28(seed=42)
        x, y, _, _ = job.read_train_data()
        load_synthesizer(job.paths['synthesizer_dir'] + job.paths['sdv_only_synthesizer'],
                         expected_data=pd.concat([x, y], axis=1), expected_seed=42)
        job.create_synthetic_data_sdv_gaussian()
        job.create_synthetic_data_sdv_categorical()
        job.create_synthetic_data_pca_gmm()
        job.create_synthetic_data_ensemble()
        job.create_synthetic_data_dnn()
        job.create_comparison_from_trained_model()
    else:
        datasets = sys.argv[1:] or ['intrusion']
        if any(dataset not in {'intrusion', 'mnist28', 'news'} for dataset in datasets):
            raise ValueError('Choose intrusion, mnist28, or news')
        for dataset in datasets:
            if dataset == 'mnist28':
                run(str(Path(__file__).resolve()), '--resume-mnist28')
                run('synthesize_data/main.py', '--dataset', dataset, '--seed', '42',
                    '--generator', 'TVAE', cwd=ROOT / 'src')
                run('scripts/run_corrected_matrix.py', '--dataset', dataset,
                    '--seed', '42', '--stage', 'classifiers', '--resume')
            else:
                run('scripts/run_corrected_matrix.py', '--dataset', dataset, '--seed', '42')
        print('Remaining seed-42 datasets completed.', flush=True)
