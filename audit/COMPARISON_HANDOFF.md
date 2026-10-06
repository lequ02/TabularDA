# Comparison reports handoff

Workspace: `D:\SummerResearch`. Research host: `thuy@10.24.10.133`; repository `/home/thuy/Research/minh_data_synth/TabularDA`; Python `/home/thuy/miniconda3/envs/env/bin/python`. Follow `AGENTS.md`, preserve other chats' changes, and inspect live processes, tmux sessions, and logs before remote mutations. Never run experiments locally.

## News normalization is part of the saved result

Every News `*.run.json` must contain:

- `test_scores.mae`, `test_scores.r2`, and `test_scores.nmae_sigma`.
- `target_normalization.sigma_y`, `split` (`test`), `ddof` (`0`), `rows`, `target_table_sha256`, and `definition`.

`nmae_sigma = mae / sigma_y`. The denominator is the population standard deviation of the same real held-out test targets used for MAE, not training or synthetic targets. Both R² and NMAE are unscaled; lower NMAE is better. Seed 42 has 7,929 test rows and sigma_y = 9485.506480005333.

`src/modeling/run_record.py` computes and saves these fields for future News runs using the saved test predictions and source IDs. On October 5, 2026, all 37 completed News seed-42 records (original, synthetic, and mix) were enriched without retraining. Existing metrics, model provenance, and checkpoints were preserved. Deployment/backfill evidence and backup paths are in `audit/news_nmae_persistence_2026_10_05.json`.

The plot builder reads the saved NMAE and checks its normalization metadata and arithmetic. Do not recompute the denominator during plotting. `audit/news_target_normalization_2026_10_05.json` is historical audit evidence, not a reporting dependency. Missing required fields must surface as an error; do not substitute another split or normalization.

## Refresh workflow

1. Inspect relevant local changes and current remote jobs. Use `.cache/remote_intrusion/server_key` explicitly and `audit/ssh_known_hosts`; never expose credentials.
2. Run `python .cache/remote_intrusion/refresh_comparison.py` locally with authorized SSH access. It retrieves selected completed comparison records plus all News records and verifies archive hashes. Evidence: `audit/latest_comparison_snapshot.json`; archive: `.cache/remote_intrusion/latest_comparison.zip`.
3. Run `python scripts/plot_recent_comparison.py` locally for lightweight artifact preparation.
4. Run `python .cache/remote_intrusion/check_comparison.py`, inspect the PNG, and check the scoped diff. The checker verifies remote archive hashes, JSON equality, saved metric values, and NMAE arithmetic. Local line endings may differ without changing record content.

Deliverables: `output/comparisons/dataset_pipeline_comparison.png`, `.md`, and `.csv`. Source paths preserve namespaces: MNIST28 and News seed 42 use `corrected_v2_seed42_mnist28_news`; other displayed runs use `corrected_v2`.

## Preserve presentation and research settings

- No pilot runs. Each row pairs the same dataset/metric, seed 42 left and seed 43 right, with matching scales and visible missing-result placeholders.
- Adult/Credit/Census KDD: binary and macro F1. Covertype/Intrusion: macro F1. MNIST12/28: accuracy. News: separate R² and NMAE panels/tables.
- Colors distinguish metric and CTGAN/TVAE. Original-data points are metric-colored triangles; generated-target benchmarks are hollow; relabeled methods are filled. No "Local" legend labels.
- Order: original, CTGAN generated target, TVAE generated target, CTGAN full-feature RF/XGB/DNN, CTGAN X-only RF/XGB/DNN, TVAE full-feature DNN, TVAE X-only DNN.
- Paper Table 6: News R² lines CTGAN -0.43, TVAE -0.20, real 0.14; Intrusion macro F1 lines 52.8%, 51.1%, 86.2%. No paper NMAE is reported: do not invent NMAE reference lines or derive MAE from R². Binary-dataset paper references belong only to binary F1. Paper splits/classifier averages differ from these experiments.
- Preserve leakage/provenance checks and caveats. Seed repeats may share prepared splits. Comparison coverage does not establish completion of the full 816-run matrix.

No experiment settings, thresholds, training data, or model selection were changed to add NMAE.

## Mix comparison reports

The mix report uses the same panels, metrics, method order, styling, and provenance checks. Mix training concatenates all real training rows with 100,000 synthetic rows; the original-data row remains the real-only baseline. Synthetic-only outputs are preserved.

Refresh and rebuild with:

1. Inspect current remote jobs and logs as above.
2. `python .cache/remote_intrusion/refresh_comparison_mix.py`
3. `python scripts/plot_recent_comparison.py --train-option mix`
4. `python .cache/remote_intrusion/check_comparison_mix.py`, then inspect the PNG.

Mix deliverables: `output/comparisons/dataset_pipeline_comparison_mix.png`, `.md`, and `.csv`. Evidence: `audit/latest_comparison_mix_snapshot.json`; archive: `.cache/remote_intrusion/latest_comparison_mix.zip`. The refresh uses the dedicated research Python environment and preserves differing local originals under `.cache/remote_intrusion/mix_refresh_backup_20261005/`.

The October 5, 2026, 10:07 a.m. Chicago snapshot contains 166 fetched records, of which 122 distinct completed records supply the selected report configurations (110 mix and 12 original). The checker verified 199 reported metric values, snapshot hashes, source JSON equality, and saved News NMAE arithmetic. MNIST12, MNIST28, and News seed 43 have no completed comparison records; Intrusion seed 42 has none, and Intrusion seed 43 has only the original baseline. Missing configurations are visible placeholders. This coverage is not completion of the full experiment matrix.
