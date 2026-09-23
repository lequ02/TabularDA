# Four-panel comparison notes

Figure: outputs/figures/29_four_panel_comparison.png (vector PDF alongside). Reproduce with `Rscript plot_four_panel_comparison.R`.

## Labels and comparisons

CTGAN_e2e and TVAE_e2e generate features and targets jointly. CTGAN-RF, CTGAN-XGB, TVAE-RF and TVAE-XGB first generate features with the named generator, then assign targets with RF or XGBoost trained on original data. RF and XGBoost do not generate the entire synthetic dataset.

Top left: CTGAN-based pipelines. Top right: TVAE-based pipelines. Each joint box contains four task scores; each hybrid box contains eight task/configuration scores (four tasks times target included/omitted during feature-generator training).

Bottom left (Generator training inputs): X only means target omitted from generator training; X + y means target included in generator training. Only hybrid pipelines are compared. Both final synthetic datasets contain targets assigned by RF/XGBoost. Each point averages CTGAN/TVAE feature generators and RF/XGBoost target predictors within one task and inclusion setting. Each box has four task means. Joint pipelines are excluded here because they always contain y and would confound the comparison. Both groups still use targets to train the RF/XGBoost predictor.

Bottom right: e2e versus hybrid. Each point is a task mean. The e2e mean averages joint CTGAN and TVAE; the hybrid mean averages both feature generators, both target predictors and both inclusion settings. Each box has four task means, avoiding unequal configuration counts in this pooled comparison.

Adult aliases are averaged and counted once. The four tasks are Adult, Covertype, MNIST12 and MNIST28. Categorical, PCA-GMM and Gaussian methods are excluded. Points are not independent experiment seeds.

Purple: end-to-end; teal: hybrid. Red dashed lines and outlined circles show comparison-group means. Top-panel RF/XGB circles repeat their pooled hybrid mean. Bottom-panel circles show their respective condition means. Red arrows show comparison minus reference; boxplot center lines show medians.

## Covertype assumption

Only joint TVAE Covertype maximum macro-F1 is changed from 0.2972320714 to 0.49, following the user's requested sensitivity. The CTGAN paper reports 0.433 for TVAE Covertype; 0.49 is not a paper value or measured replication. All other scores and recorded source files remain unchanged. Final-epoch scores are not plotted.

## Statistical summary

Two-sided paired t tests use four task differences, not the number of plotted configurations. Holm correction covers the four comparisons in this figure. TVAE and pooled e2e results use the assumed 0.49 baseline; target-inclusion results are unaffected by it.

| Panel | Reference mean | Comparison mean | Gain | Relative gain | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| CTGAN-based | 0.5462 | 0.7900 | 0.2438 | 44.63% | 0.0977 | 0.2930 |
| TVAE-based | 0.7742 | 0.8126 | 0.0384 | 4.96% | 0.1313 | 0.2930 |
| Generator training inputs | 0.8028 | 0.7998 | -0.0030 | -0.37% | 0.1379 | 0.2930 |
| End-to-end vs hybrid | 0.6602 | 0.8013 | 0.1411 | 21.37% | 0.0598 | 0.2392 |

Confidence intervals and complete figure observations are in outputs/paper_comparison/four_panel_049_summary.csv and four_panel_049_figure_data.csv.

## Trust limits

Test-fitted preprocessing, test-maximum selection, inconsistent Covertype test versions, incomplete split/checkpoint provenance, related MNIST tasks, and absent independent seed runs remain unresolved. The assumed score is not a leakage correction. These comparisons do not establish leakage-free superiority.

Source: outputs/method_scores.csv; MD5: `9d7fb9530b5b9836e7db1f2b684f5b7a`.
