# Corrected modeling package

This package contains the corrected classification and regression entrypoints, training code, data loading, shared constants, run recording, and model definitions. The unified dispatcher routes the `news` dataset to regression and supported classification datasets to classification.

From the `src` directory, pass the dataset name and usual training options:

```powershell
python -m modeling --dataset-name adult --train-option original --test-option original --validation 0.2 --batchsize 128 --lr 0.001 --global-round 100
python -m modeling --dataset-name news --train-option original --test-option original --validation 0.2 --batchsize 128 --lr 0.001 --global-round 100
```

The direct entrypoints remain available as `python -m modeling.classification_main` and `python -m modeling.regression_main`. Run outputs are written under the repository-level `output/corrected_v2/`; corrected datasets are read from `data/corrected_v2/`.

The one-seed experiment matrix is listed in `audit/CORRECTED_RUN_SPEC.md` and launched from the repository root with `python scripts/run_corrected_matrix.py` on a GPU host.
