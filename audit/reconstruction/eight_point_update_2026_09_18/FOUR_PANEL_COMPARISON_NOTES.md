# Four-panel comparison notes

Figure: outputs/figures/29_four_panel_comparison.png (vector PDF alongside). Reproduce with `Rscript plot_four_panel_comparison.R`.

## Labels and comparisons

CTGAN_e2e and TVAE_e2e generate features and targets jointly. CTGAN-RF, CTGAN-XGB, TVAE-RF and TVAE-XGB first generate features with the named generator, then assign targets with RF or XGBoost trained on original data. RF and XGBoost do not generate the entire synthetic dataset.

Bottom left: CTGAN-based pipelines. Bottom right: TVAE-based pipelines. Each joint box contains four task scores; each hybrid box contains eight task/configuration scores (four tasks times target included/omitted during feature-generator training).

Top right (Feature generator input): Features only omits the target during feature-generator training; Features + target includes it. Both final synthetic datasets contain targets assigned by RF/XGBoost. Each box contains eight observations: four tasks times two feature generators. Each point averages RF/XGBoost within one task, feature generator and input setting; CTGAN and TVAE remain separate. Joint pipelines are excluded because they always contain the target and would confound the comparison. Both groups still use targets to train the RF/XGBoost predictor.

Top left: e2e versus hybrid. Each box contains eight observations: four tasks times two feature generators. Each e2e point is its generator's joint score; each hybrid point averages RF/XGBoost and both input settings within that task and generator. CTGAN and TVAE remain separate in the plotted points and contribute equally to pooled means. The task breakdown is also in FOUR_PANEL_STATISTICS.md.

Adult aliases are averaged and counted once. The four tasks are Adult, Covertype, MNIST12 and MNIST28. Categorical, PCA-GMM and Gaussian methods are excluded. Points are not independent experiment seeds.

Purple: end-to-end; teal: hybrid. Red dashed lines and outlined circles show comparison-group means. Generator-specific RF/XGB circles repeat their pooled hybrid mean. The pooled panels' circles show their respective condition means. Red arrows show comparison minus reference; boxplot center lines show medians.

## Covertype assumption

Only joint TVAE Covertype maximum macro-F1 is changed from 0.2972320714 to 0.49, following the user's requested sensitivity. The CTGAN paper reports 0.433 for TVAE Covertype; 0.49 is not a paper value or measured replication. All other scores and recorded source files remain unchanged. Final-epoch scores are not plotted.

## Statistical summary

Two-sided paired t tests use four task differences, not eight independent observations. CTGAN and TVAE share the same tasks, so their plotted points are averaged within task for inference. This preserves the earlier statistical results while displaying all eight task/generator observations per group. Holm correction covers the four comparisons. TVAE and pooled e2e results use the assumed 0.49 baseline; the input comparison is unaffected by it.

| Panel | Reference mean | Comparison mean | Gain | Relative gain | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| End-to-end vs hybrid | 0.6602 | 0.8013 | 0.1411 | 21.37% | 0.0598 | 0.2392 |
| Feature generator input | 0.8028 | 0.7998 | -0.0030 | -0.37% | 0.1379 | 0.2930 |
| CTGAN-based | 0.5462 | 0.7900 | 0.2438 | 44.63% | 0.0977 | 0.2930 |
| TVAE-based | 0.7742 | 0.8126 | 0.0384 | 4.96% | 0.1313 | 0.2930 |

Confidence intervals and complete figure observations are in outputs/paper_comparison/four_panel_049_summary.csv and four_panel_049_figure_data.csv.

## Trust limits

Test-fitted preprocessing, test-maximum selection, inconsistent Covertype test versions, incomplete split/checkpoint provenance, related MNIST tasks, and absent independent seed runs remain unresolved. The assumed score is not a leakage correction. These comparisons do not establish leakage-free superiority.

Source: outputs/method_scores.csv; MD5: `9d7fb9530b5b9836e7db1f2b684f5b7a`.
