# Comparison reports handoff

Latest source snapshots: October 7, 2026, 11:41 p.m. Chicago. Both reports now contain 193 distinct completed records and 309 displayed metric values each, adding 39 synthetic-only and 40 mix records relative to the prior reports. All 618 values passed source/archive/hash checks; focused validation confirmed the production settings, namespace restrictions, and unchanged previously displayed records. Both plots were visually checked, and the scoped whitespace check passed with the existing CRLF line endings recognized. Coverage and new source paths are saved in `audit/comparison_refresh_2026_10_07.json`.

Weighted Census and MNIST12 now have all 15 selected configurations for both seeds in both reports. News seed 43 has nine selected configurations (original, both generated-target baselines, and full-feature RF/XGB/DNN for both generators); its six X-only configurations remain missing. MNIST28 seed 43 has only its original baseline. Housing has both original baselines and no selected synthetic/mix arms yet. Intrusion seed 42 remains empty; seed 43 still has only its original baseline. These selected comparison counts do not establish completion of the original 816-run matrix or the added Housing study.

Housing now has separate R² and NMAEσ panels/tables. Its two completed records were enriched from their existing real held-out predictions without retraining; test-table hashes, source IDs, target values, and MAE consistency were checked. The result writer now saves normalization for future Housing runs as well as News. Originals and the deployed writer are preserved remotely under `.cache/housing_nmae_20261007_234006/`; evidence is in `audit/housing_nmae_persistence_2026_10_07.json`. Housing uses `corrected_v2`, has 4,128 test rows and σ_y = 1.170145328085185, and has no supplied paper reference. Its baseline R²/NMAEσ values are 0.524633/0.540051 (seed 42) and 0.500283/0.561280 (seed 43).

The latest News seed-43 records include extreme negative R²: CTGAN generated-target synthetic −93.089 and CTGAN full-feature DNN mix −5216.564. Preserve them. For a paired R² range extending below −1, the builder uses a symmetric-log axis with a linear region of ±0.1, labeled in the figure and explained in the Markdown; score labels, CSVs, and tables remain unscaled. Both seeds use matching scales and all paper lines remain visible.

Live inspection found active `research-slow`, `research-fast-1`, and `research-fast-2` tmux workers. At 11:40 p.m. the queue had 90 completed tasks, three running, and 149 pending; task totals include preparation, generation, labeling, and evaluation, so they are not downstream completion counts. The running tasks were MNIST28 seed-43 CTGAN full generation and News seed-43 CTGAN/TVAE X-only generation. No jobs were interrupted, duplicated, or restarted. Previous reports, snapshots, and archives are preserved locally under `.cache/remote_intrusion/comparison_before_refresh_20261007_233801/`.

Previous refresh: October 6, 2026, 11:33 a.m. Chicago. Both 15-row reports were rebuilt and visually checked; source/snapshot checks verified 240 synthetic-only metric values from 154 distinct records and 238 mix values from 153 records. New weighted Census seed-42 entries are TVAE X-only XGB synthetic (49.7% binary / 73.0% macro F1) and TVAE X-only RF mix (55.2% / 75.9%). Census now had 11 displayed synthetic-only and 10 mix configurations; weighted seed 43 remained missing. Live inspection found no weighted Census runner or its tmux session, no completed.json, and stale status with failed configurations and a saved running field. That field did not establish a live job; inspected failed logs contained PyTorch import tracebacks. That refresh did not restart or change experiments.

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

Deliverables: `output/comparisons/dataset_pipeline_comparison.png`, `.md`, and `.csv`. Source paths preserve namespaces: MNIST28 and News seed 42 use `corrected_v2_seed42_mnist28_news`; Census KDD uses `census_kdd_weighted_macro_f1_20261005`; other displayed runs use `corrected_v2`. Both refresh helpers include the weighted namespace.

The user requested replacing the earlier Census KDD evaluation with the weighted rerun in both reports. Use only completed records from that namespace, including its real-only baseline; do not fill missing weighted configurations with earlier unweighted scores. This protocol uses BCEWithLogitsLoss with positive weight equal to actual training negatives / positives, and checkpoint selection by real-development macro F1. Both the loss and selection changed. The plot builder verifies the saved protocol and training-derived weight. The queue was still running on October 5 at 8:43 p.m. Chicago; seed 43 had no completed weighted records. Imported, verified full-budget results in this matrix namespace are valid; the separate pilot namespace is not a report source.

## Preserve presentation and research settings

- No pilot runs. Each row pairs the same dataset/metric, seed 42 left and seed 43 right, with matching scales and visible missing-result placeholders.
- Adult/Credit/Census KDD: binary and macro F1. Covertype/Intrusion: macro F1. MNIST12/28: accuracy. News and Housing: separate R² and NMAE panels/tables.
- Colors distinguish metric and CTGAN/TVAE. Original-data points are metric-colored triangles; generated-target benchmarks are hollow; relabeled methods are filled. No "Local" legend labels.
- Order: original, CTGAN generated target, TVAE generated target, CTGAN full-feature RF/XGB/DNN, CTGAN X-only RF/XGB/DNN, TVAE full-feature RF/XGB/DNN, TVAE X-only RF/XGB/DNN. Include all RF/XGB/DNN results in both reports; the user explicitly excludes NB and PCA/GMM. Each report has 15 configuration rows, with placeholders for missing runs.

The October 5, 2026, 9:01 p.m. Chicago refresh includes the four previously omitted TVAE RF/XGB rows. Synthetic-only uses 153 distinct completed records (238 displayed metric values); mix uses 152 (236 values). All displayed values passed source-record/snapshot checks, both plots were visually checked, and available TVAE RF/XGB entries were explicitly verified. Census still uses only the weighted namespace; seed 42 has ten completed displayed configurations for synthetic-only and nine for mix, while weighted seed 43 remains empty.
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

## Latest refresh: October 5, 2026, 7:31 p.m. Chicago

Both comparison reports were refreshed from verified remote snapshots: 195 fetched records for synthetic-only and 172 for mix. MNIST12 seed 43 now contributes seven selected synthetic-only configurations and six mix configurations. Its original baseline and TVAE configurations remain missing; CTGAN full-feature DNN mix is also missing. MNIST28 and News seed 43 still have no completed comparison records; Intrusion seed 42 has none, and seed 43 has only its original baseline. The reports preserve missing-result placeholders, matching dataset/seed panels, paper reference lines, and saved News normalization.

Source-record and snapshot checks passed for all 206 displayed synthetic-only metric values and 205 mix values. Both plots were visually inspected; split/provenance checks passed. This selected coverage does not establish completion of the full matrix.

At the October 5, 2026, 7:30 p.m. Chicago refresh, the snapshot contains 172 fetched records and the report uses 128 distinct completed records (116 mix and 12 original). Six MNIST12 seed-43 CTGAN mix records were added: generated targets, full-feature RF/XGB, and X-only RF/XGB/DNN. Their accuracies are 80.7%, 92.7%, 94.0%, 92.4%, 93.6%, and 93.8%, respectively. Existing source records are unchanged. The checker verified 205 metric values against snapshot hashes and source JSON, including saved News NMAE arithmetic; production generator and downstream settings were checked for all 128 report records. The updated PNG was visually reviewed. Other missing configurations remain marked. The previous report, snapshot, and archive are preserved under `.cache/remote_intrusion/mix_report_before_refresh_20261005_193014/`.
