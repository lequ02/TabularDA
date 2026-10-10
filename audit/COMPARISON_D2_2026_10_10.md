# D² absolute error in the comparison reports

Computed October 10, 2026, **1:10:03 a.m. Chicago**, using saved held-out predictions on the research server. No model was retrained and no checkpoint selection, prediction file, or original run record was changed.

Both synthetic-only and mix CSVs now include D² absolute error for News and Housing, seeds 42/43. This covers **120 displayed values from 116 distinct completed records**; the four real-only baselines occur in both reports. The reports use `news_log_v1` and `housing_no_faker_20261009`. Exploratory dropout trials are excluded. All **1,476 existing CSV cells**, including NMAEσ, remain unchanged; existing score verification covers 628 displayed metric values. This is a derived-metric update to the October 9, 9:42 p.m. report source snapshots, not a new completion claim or a refresh of other datasets.

The definition is:

`D²_absolute_error = 1 − MAE(y_test, prediction) / MAE(y_test, median(y_test))`

The remote computation uses `sklearn.metrics.d2_absolute_error_score`. Higher is better: 1 is perfect, 0 matches the constant test-median baseline, and negative values are worse than that baseline. R² uses a squared-error baseline, so its values need not equal D². The saved prediction targets were checked against the prepared real test targets and split source IDs. Saved MAE/R² were independently reproduced. Computation checks all requested inputs before writing any new files.

| Dataset | Test rows | Test median | Median-baseline MAE | Real-only D², seed 42 | Real-only D², seed 43 |
|---|---:|---:|---:|---:|---:|
| News | 7,929 | 1,400 | 2,439.027746 | −0.085693 | −0.003304 |
| Housing | 4,128 | 1.7925 | 0.897715 | 0.296059 | 0.268388 |

Each original `*.run.json` has a separate `*.d2.json` sidecar containing the metric, test median, median-baseline MAE, prediction MAE, row count, selected epoch, and target-table/prediction/original-record hashes. The report builder verifies the sidecar against the original record and reads its saved metric and normalization; plotting does not read test targets or recompute their normalization. Original run-record hashes stay valid for existing snapshots and archives.

The added CSV metric columns are `news_corrected_d2_absolute_error`, `news_corrected_seed43_d2_absolute_error`, `california_housing_corrected_d2_absolute_error`, and `california_housing_corrected_seed43_d2_absolute_error`. Four corresponding `*_d2_metadata` columns identify the sidecars. Values are unscaled with six decimal places. NMAEσ remains in the CSVs. Paper rows have blank D² cells because no paper D² references are supplied.

The figures and Markdown tables combine R²/D² for each regression dataset/seed. Green squares show R²; blue/purple downward triangles show CTGAN/TVAE D². Real-only scores retain triangular markers and generated-target baselines retain hollow markers. News paper lines refer only to R². Housing has no added missing-paper-reference annotation. Classification panels and completion placeholders remain unchanged.

The News pairs now use a symmetric-log axis, linear only within ±0.001 and logarithmic beyond that region, to spread the values clustered near zero. Both seeds use the same scale and limits. Axis ticks and score labels show the original signed values; no CSV values are transformed. Housing retains its linear axis because its scores are already separated across the positive range.

## Reproduce and refresh

Use [the remote computation script](../scripts/compute_regression_d2.py) with a JSON list of `{ "path": "output/...run.json", "sha256": "..." }` requests and `--requests`/`--evidence` arguments. Run it only with the research-server environment. The deployed computation and request list for this update are under remote `.cache/comparison_d2_20261010/`; `evidence.json` and `scores.zip` preserve the output. Synchronize verified sidecars alongside original records before rebuilding. New regression records require their own verified sidecars; missing or mismatched sidecars are errors.

Then run the existing `scripts/plot_recent_comparison.py` for synthetic-only and with `--train-option mix` for mix. Original artifacts and the builder are backed up at `.cache/remote_intrusion/comparison_before_d2_20261010_010957/`.

Evidence: [computation and source hashes](comparison_d2_computation_2026_10_10.json), [CSV preservation and 120-value checks](comparison_d2_csv_validation_2026_10_10.json), and [final artifact hashes and visual review](comparison_d2_artifacts_2026_10_10.json).
