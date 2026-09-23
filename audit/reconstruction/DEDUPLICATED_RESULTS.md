# Counting Adult/Census/Census-KDD once

For each synthesis configuration, average its three Adult-like results, then give that combined dataset the same weight as Covertype, MNIST12 and MNIST28. This leaves 72 observations (18 configurations × 4 dataset labels), with the original Gaussian exclusion retained. The comparator is the equal-weight average of CTGAN and TVAE, not either baseline individually. Response: historical maximum test macro-F1.

| Result | Original six labels | Four labels after collapsing aliases |
|---|---:|---:|
| Proposed-method mean | 0.752627 | 0.753693 |
| CTGAN/TVAE pooled mean | 0.677468 | 0.636082 |
| Absolute difference | 0.075159 | 0.117611 |
| Relative gain | 11.09% | 18.49% |
| Original mixed model, adjusted p | 0.0032 | 0.0011 |

The mixed model uses the original combined approach/target-inclusion factor and random intercepts for dataset and feature-generator family, fitted by ML. Contrasts use Kenward–Roger degrees of freedom and the same three-comparison Sidak adjustment. Its adjusted 95% interval for the collapsed-data gain is approximately **[0.0409, 0.1944]** macro-F1.

The gain increases because the proposed methods perform slightly worse on Adult-like data; counting those results three times previously reduced the overall mean advantage.

## Sensitivity

- **Count dataset-level gains as the observations:** differences are Adult −0.009745, Covertype +0.152139, MNIST12 +0.153391, MNIST28 +0.174659. A two-sided paired t-test gives **p=0.0707**, with 95% interval **[−0.0185, 0.2537]**. Four tasks are a small sample and the two MNIST representations are related, so this is an exploratory check, not a definitive independence-corrected test.
- **Retain labeling-method random effects:** gain remains about 0.1176, but **p=0.1191**. This is a different modeling assumption; significance is not robust across these models.
- **Keep only the Adult-labeled runs instead of averaging aliases:** gain **0.11645 / 18.29%**, original-model adjusted **p=0.0012**, dataset-level **p=0.0769**. The deduplication choice changes little.
- **Include the excluded Gaussian configurations:** gain falls to **0.05868 / 9.22%**; original-model adjusted **p=0.6562** (singular fit) and dataset-level **p=0.1717**.

**Interpretation:** deduplication increases the descriptive advantage and strengthens significance under the original model. It does not establish a robust general superiority claim, resolve inconsistent Covertype test sets, repair test-fitted preprocessing or eliminate test-max selection. These are statistical recalculations of historical scores, not leakage-corrected experiments.

Source: `D:/Rprojects/research_data_synthesis/April29_macro_max_mfa.csv`. Reproduction: [complete R output](deduplicated_statistics.txt), [collapsed per-method scores](deduplicated_method_scores.csv), [dataset-level differences](deduplicated_dataset_differences.csv). Original inputs were not modified.
