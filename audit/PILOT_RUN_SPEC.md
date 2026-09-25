# CTGAN pilot

Run `python scripts/run_corrected_pilot.py` from the repository root on a GPU host. `--dry-run` lists the commands; `--dataset adult` selects one dataset; `--stage generators` or `--stage classifiers` runs one stage. `--jobs 2` runs two datasets at a time on the same GPU while keeping each dataset's steps in order; the default is one job. Compare elapsed time before using `--jobs 3`. Each dataset writes to its own directories, and results are assembled after all workers finish. The launcher holds `output/pilot_ctgan_v1/.run.lock` until completion, so a second pilot launch fails before writing. If a process is killed and leaves the lock behind, remove it only after confirming the pilot has stopped.

The pilot covers Adult, MNIST28, and Covertype with seed 42. It fits full-table and X-only CTGAN for each dataset. It evaluates the CTGAN-generated target, RF/XGB/DNN labels on X-only samples, and RF/XGB/DNN relabeling of full-table samples. Each synthetic table gets synthetic-only and mixed downstream training, plus one real-only baseline: **6 generator fits, 21 synthetic tables, 45 downstream runs**. CTGAN uses 500 epochs, batch size 500, and 100,000 sampled rows. The downstream DNN uses batch size 128, learning rate 0.001, at most 100 epochs, and patience 30. Test evaluation uses the checkpoint selected by real-development loss.

Both the pilot and full matrix use these names, in separate `data/`, `sdv trained model/`, and `output/` directories:

```text
adult_seed42_real_train_onehot.csv
adult_seed42_ctgan_full.pkl
adult_seed42_ctgan_xonly.pkl
adult_seed42_ctgan_full_generated_100k.csv
adult_seed42_ctgan_xonly_rf_100k.csv
adult_seed42_ctgan_full_rf_100k.csv
adult_seed42_ctgan_xonly_rf_mix.weights.pth
adult_seed42_ctgan_xonly_rf_mix.run.json
```

`full_rf` means full-table CTGAN features with the generated target replaced by RF predictions. Matching `.quality.json`, `.provenance.json`, `.predictor.pkl` or `.predictor.pt`, `.dnn.json`, `.epochs.csv`, `.predictions.csv`, and plot files share the relevant stem. The checked pilot tables are `output/pilot_ctgan_v1/results/per_run.csv` and `summary.csv`.

Every downstream run saves train and real-development metrics per epoch and real-test metrics in its `.run.json` and `per_run.csv` row. **All three datasets:** loss, accuracy, balanced accuracy, and macro/micro/weighted precision, recall, and F1. **Adult additionally:** binary precision, recall, F1, PR-AUC (average precision), and ROC-AUC. The prediction CSV records source ID, true label, predicted label, and Adult's positive-class score. With one seed, between-seed variation is unavailable.
