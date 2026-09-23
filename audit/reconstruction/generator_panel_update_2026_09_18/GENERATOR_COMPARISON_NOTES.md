# Generator-matched figure notes

Figure: outputs/figures/28_generator_matched_comparison.png (vector PDF alongside).

## Layout and observations

The left panel compares joint CTGAN with RF/XGBoost targets assigned to CTGAN-generated features. The right panel compares joint TVAE with RF/XGBoost targets assigned to TVAE-generated features. RF and XGBoost are target predictors, not feature generators. Categorical, PCA-GMM and Gaussian methods are excluded.

Four tasks are included: Adult, Covertype, MNIST12 and MNIST28. The three Adult aliases are averaged within configuration and counted as one task. Each joint baseline has four points. Each RF/XGBoost box has eight points: four tasks times two generator-training settings (target included versus omitted). These are task/configuration summaries, not independent random-seed runs.

Purple denotes joint synthesis; teal denotes decoupled labeling. Red dashed lines and outlined circles represent panel-specific approach means. Circles for RF and XGBoost repeat the pooled decoupled mean, rather than each labeler's individual mean. The red arrow shows the difference between approach means. Boxplot center lines are medians, not those means.

## User-specified Covertype sensitivity

Only the joint TVAE Covertype maximum macro-F1 changes from 0.2972320714 to **0.49**, as requested. Joint CTGAN Covertype and all RF/XGBoost scores remain recorded values. The final-epoch column is unchanged and is not plotted.

The 0.49 value is a user-specified hypothetical assumption. It is not the CTGAN paper's reported TVAE Covertype value (0.433), not a measured replication, and not a leakage adjustment. Earlier 0.433 analyses and original recorded results are preserved.

## Statistical summary

Configurations are averaged within each approach and task before averaging four task differences. Two-sided paired t tests use four differences (three degrees of freedom). Holm adjustment covers the two feature-generator comparisons. The TVAE statistics are mechanical sensitivity outputs from an assumed baseline, not results of a new experiment.

| Features | Joint mean | RF/XGB mean | Gain | Relative gain | 95% interval | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CTGAN-based | 0.5462 | 0.7900 | 0.2438 | 44.63% | [-0.0823, 0.5698] | 0.0977 | 0.1953 |
| TVAE-based | 0.7742 | 0.8126 | 0.0384 | 4.96% | [-0.0209, 0.0978] | 0.1313 | 0.1953 |

## Trust limits and reproduction

Known test-fitted preprocessing, test-maximum selection, inconsistent Covertype test versions, incomplete checkpoint/split provenance, related MNIST tasks, and absent independent seed runs remain unresolved. Separating generators and removing Adult duplication improves the comparison structure but does not remove leakage. This figure cannot establish leakage-free superiority.

Run `Rscript plot_generator_comparison.R` from this analysis folder to reproduce the figure, source data, task means, statistical summary and these notes. The full analyse.R workflow also generates them.

Input: outputs/method_scores.csv; MD5: `9d7fb9530b5b9836e7db1f2b684f5b7a`.

Figure-specific CSVs are in outputs/paper_comparison and start with generator_matched_049_.
