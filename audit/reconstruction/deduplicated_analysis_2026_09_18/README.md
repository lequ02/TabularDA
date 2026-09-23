# Adult counted once: historical reanalysis

This folder is separate from the old analysis. `analyse.R` reads two frozen April 29 CSV snapshots and regenerates its own outputs. It does not modify the original files or refit the synthesis/prediction models. Input paths and SHA-256 hashes are recorded in `inputs/source_manifest.json`.

Run in PowerShell:

```powershell
& 'D:\R-4.5.1\bin\Rscript.exe' --vanilla 'D:\Rprojects\research_data_synthesis\deduplicated_analysis_2026_09_18\analyse.R'
```

Required R packages: ggplot2, lme4, lmerTest and emmeans. An optional first argument supplies another input directory containing the same two CSVs. Outputs are replaced only within this analysis folder.

## What changed

Adult, Census and Census_Kdd are averaged within each method and counted as one dataset, based on the reconstructed April code and data routing. The other tasks are Covertype, MNIST12 and MNIST28. This removes duplicate dataset weighting; it **does not remove leaked observations** or establish that Census_Kdd is an independent KDD evaluation.

Each dataset receives equal weight. Method-family means give each recorded configuration equal weight. The joint baseline pools CTGAN and TVAE equally. The main comparison includes every recorded labeling method; the historical exclusion of Gaussian methods is a sensitivity analysis. End-of-training scores provide another sensitivity check against reporting the maximum over test evaluations.

## Results

| Comparison | Mean macro-F1 gain | Relative gain | Dataset-level p | 95% interval for gain |
|---|---:|---:|---:|---:|
| All methods, test maximum | 0.0587 | 9.22% | 0.172 | -0.0457 to 0.1631 |
| Gaussian excluded, test maximum | 0.1176 | 18.49% | 0.071 | -0.0185 to 0.2537 |
| All methods, final epoch | 0.0654 | 10.58% | 0.188 | -0.0572 to 0.1880 |
| Gaussian excluded, final epoch | 0.1251 | 20.26% | 0.082 | -0.0295 to 0.2798 |

Tests use the four paired dataset differences, with two-sided t tests. These are exploratory comparisons: four tasks provide limited evidence, and the two MNIST tasks are related. No adjustment is applied to these sensitivity comparisons.

Matched target inclusion gives with-y minus without-y = **-0.0024**, p = **0.584**, 95% interval **[-0.0149, 0.0101]**. A nonsignificant result does not establish equivalence or justify a claim that excluding y has no cost.

For continuity, the script reproduces the historical mixed model and its three Sidak-adjusted contrasts, including the old third contrast without assigning it a new interpretation. That model reports a restricted-family gain p = 0.0011; adding labeling-method dependence gives p = 0.1191. Model sensitivity and the absence of independent seed replicates prevent treating the smaller p as conclusive evidence. See `outputs/statistical_results.txt` for estimates, diagnostics and R session details.

The original-data reference outperforms every recorded synthetic configuration on each of the four tasks. The gains above are against the pooled joint-synthesis baseline, not gains over original-data training or proof of superiority over both CTGAN and TVAE separately.

## Figures and tables

All figures are supplied as PNG and vector PDF in `outputs/figures`:

1. Dataset means: all methods versus the historical Gaussian exclusion.
2. All 22 synthesis configurations, including their scores.
3. Decoupled-minus-joint gains by dataset.
4. Observed, matched target-inclusion effects.
5. Dataset-level sensitivity intervals.
6. Historical mixed-model residuals.
7. Residual normality plot.
8. Original-data reference versus joint baselines and TVAE with RF/XGBoost targets.

CSV files expose collapsed method scores, dataset gains, sensitivity summaries, model contrasts and target-inclusion pairs. The revised title and abstract are in `revised_title_and_abstract.md`. As requested, the abstract amends the original poster wording and uses its historical Gaussian-excluded comparison and mixed-model target-inclusion contrast (p = 0.9991 after counting Adult once); the expanded analyses remain in this report.

## Trust limits

This analysis cannot calculate leakage-adjusted performance from summary scores. The audit found differing Covertype test versions, preprocessing fitted to test data, selection of maximum test scores, and incomplete run/split provenance. Removing Adult duplication changes statistical weighting, not the underlying predictions. Confirmatory results require generator fitting and target-predictor fitting on training data only, preprocessing learned from training data, model selection on validation data, and evaluation on one fixed untouched real test set with independent repeated seeds.
