# Full downstream matrix status — October 13, 2026

## Result

The current runner defines **eight datasets**: Adult, Census KDD, Credit, Covertype, MNIST12, MNIST28, News, and California Housing. Intrusion is absent; Housing is included, and News remains a separate regression dataset. There are six classification datasets and two regression datasets.

The matrix uses seeds 42 and 43, CTGAN/TVAE/TabDDPM, and synthetic-only/mix evaluation. It plans **1,168 downstream run records**: 384 per generator plus 16 shared real-only baselines. A live remote inventory at about **7:39 p.m. Chicago** found **1,070 records complete and 98 records absent**.

| Generator or baseline | Complete | Planned | Absent |
|---|---:|---:|---:|
| Real-only baselines | 16 | 16 | 0 |
| CTGAN | 364 | 384 | 20 |
| TVAE | 358 | 384 | 26 |
| TabDDPM | 332 | 384 | 52 |
| **Total** | **1,070** | **1,168** | **98** |

| Dataset | CTGAN | TVAE | TabDDPM |
|---|---:|---:|---:|
| Adult | 52/52 | 52/52 | 52/52 |
| Census KDD | 40/52 | 34/52 | 52/52 |
| Credit | 52/52 | 52/52 | 0/52 |
| Covertype | 52/52 | 52/52 | 52/52 |
| MNIST12 | 52/52 | 52/52 | 52/52 |
| MNIST28 | 52/52 | 52/52 | 52/52 |
| News | 36/36 | 36/36 | 36/36 |
| California Housing | 28/36 | 28/36 | 36/36 |

## Matrix definition

Each generator has 13 methods per classification dataset: one generated-target baseline, plus six labelers applied to both full-table relabeling and features-only relabeling. Each method has synthetic and mix modes for both seeds, giving 52 runs per classification dataset and generator.

Each generator has nine methods per regression dataset: one generated-target baseline, plus PCA/GMM, RF, XGB, and DNN applied to both relabeling constructions. With two modes and two seeds, that gives 36 runs per regression dataset and generator. Each of the eight datasets also has one shared original real-only baseline per seed.

Thus the plan is `3 × (6 × 52 + 2 × 36) + 16 = 1,168` downstream records. Generator fits and diagnostics are separate from this downstream count.

## Missing run records

- **CTGAN / Census KDD — 12:** seed 43, synthetic and mix, for the x-only Gaussian NB, Categorical NB, and PCA/GMM arms (`gaussian`, `categorical`, `pca_gmm`) and the full-table relabeling arms (`compare_gaussian`, `compare_categorical`, `compare_pca_gmm`).
- **TVAE / Census KDD — 18:** seed 42, synthetic and mix, for full-table relabeling with Gaussian NB, Categorical NB, and PCA/GMM. Seed 43 is missing those same three full-table arms and the corresponding x-only arms, each in both modes.
- **CTGAN / California Housing — 8:** seeds 42 and 43, synthetic and mix, for x-only PCA/GMM (`pca_gmm`) and full-table PCA/GMM relabeling (`compare_pca_gmm`).
- **TVAE / California Housing — 8:** seeds 42 and 43, synthetic and mix, for `tvae_pca_gmm` and `tvae_compare_pca_gmm`.
- **TabDDPM / Credit — 52:** both seeds and both modes are absent for the generated-target baseline (`tabddpm`), all six x-only labelers (`tabddpm_{gaussian,categorical,pca_gmm,rf,xgb,dnn}`), and all six full-table relabelers (`tabddpm_compare_{gaussian,categorical,pca_gmm,rf,xgb,dnn}`).

The inventory checked each expected record's dataset, seed, mode, and method; required selected development epoch; and presence of prediction, downstream-weight, synthetic-table, generator-provenance, and other referenced artifact files. All present records passed these checks. This was a record and artifact-presence audit; it does not determine whether an absent record is queued, running, or failed, and it does not count generator fits or diagnostics.

The separate selected comparison reports have **112/112 selected TabDDPM records**. That report subset is complete; it does not mean the broader 1,168-record matrix is complete.
