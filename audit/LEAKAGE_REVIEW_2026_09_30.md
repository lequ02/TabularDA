# Leakage review — September 30, 2026

**Verdict: the results do not support a blanket claim of leakage-free evaluation.**

The remote audit covered every completed corrected/pilot run available at the time of inspection: **313 downstream runs**, **152 referenced synthetic tables**, and **66 prepared real CSVs**. It also checked data provenance for **406 simulated benchmark records**. The remote run count remained 313 at the end of the main audit. The planned corrected matrix is not complete.

Remote root: `/home/thuy/Research/minh_data_synth/TabularDA` on `thuy@10.24.10.133`. The dedicated SSH identity worked. All remote checks were read-only: scripts were supplied through SSH standard input, diagnostics were saved locally, and no production samples, results, checkpoints, or training jobs were changed.

## Confirmed exception: MNIST12 TVAE synthetic/holdout feature overlap

All **13 TVAE synthetic tables for MNIST12 seed 42** contain feature vectors identical to images in the reserved real dev and test partitions. Both synthetic-only and mixed training consume these tables, so **26 downstream runs** fail the strict requirement that evaluation features have never appeared in downstream training.

| TVAE synthetic feature source | Target variants | Matching real dev rows / 6,033 | Matching real test rows / 9,873 |
| --- | --- | ---: | ---: |
| Full-table sample | Generated, GaussianNB, CategoricalNB, PCA-GMM, RF, XGB, DNN | 369 | 377 |
| X-only sample | CategoricalNB, PCA-GMM, RF, XGB, DNN | 348 | 386 |
| X-only GaussianNB sample | GaussianNB | 376 | 392 |

This represents **3.82–3.97% of the real test rows**. Counts are holdout rows with a matching synthetic feature vector; they include multiple source images sharing the same processed features. They are not counts of unique images copied from the original source. For example, the full-table TVAE sample has 208 synthetic rows matching test features, corresponding to 140 unique feature vectors and 377 real test rows.

The finding was first detected using normalized numeric feature fingerprints, then independently confirmed for every affected table by direct joins on all 144 pixel columns. Full feature-and-target row matches also occur. Feature overlap matters even when the generated label differs from the test label.

**Follow-up investigation supports train-only fitting and independently generated categorical collisions.** The actual saved full-table and X-only TVAE checkpoints encode all 144 pixels as discrete categorical variables, with 500 recorded training epochs. Four fresh 100,000-row CPU samples (two independent seeds per checkpoint) were generated in the original experiment environment with an audit hook denying every data CSV read throughout checkpoint loading and sampling. There were zero attempted CSV reads. Only after generation were holdout files opened for comparison. The four samples matched 395, 348, 376, and 371 real test rows, respectively, comparable to the existing results. Thus exact matches occur without supplying or reading the holdout CSVs during generation.

Both generator fit-table hashes reproduce from the 54,094 real training rows. Independently rebuilding the binary 12×12 representation from the cached 70,000 original MNIST images verified every saved train/dev/test pixel and target against its recorded source ID, with network access disabled. The real splits remain disjoint. Among the 140 unique shared test patterns in the saved full-table sample, the nearest real training pattern differs by one pixel for 128 patterns, two pixels for nine, three pixels for two, and four pixels for one. These are plausible small variations of training patterns in a compressed discrete image space; the nearest-neighbor observation supports that interpretation but does not identify the causal mechanism for each individual sample.

No holdout data entering TVAE fitting was found in the inspected fit path, source identities, or provenance. The fixed binarization/resizing operation uses no fitted holdout statistics. DNN relabeling selects a predictor checkpoint on the designated real dev set, as specified; test data is excluded from that selection. The overlap flag should therefore be read as a failure of the **strict unseen-feature guarantee**, rather than proof of source-record leakage. The original training files, checkpoints, and experiment scores were preserved; only in-memory diagnostic samples were generated. Full evidence is in [mnist_tvae_match_investigation_2026_09_30.jsonl](mnist_tvae_match_investigation_2026_09_30.jsonl), reproduced by `scripts/investigate_mnist_tvae_matches.py`.

**This proves input overlap, not unauthorized use of test data during fitting.** The real train/dev/test partitions are mutually disjoint, and the generator fit hashes reproduce from real training data. Binarization and resizing create a finite, compressed image space; a generator can independently produce a feature pattern present in the holdout. Consequently these results may be legitimate for an explicitly stated IID task that allows repeated feature values, but they cannot be described as evaluation exclusively on unseen feature vectors. The audit does not establish the direction or magnitude of any score bias.

The existing manifest checks only real train against real dev/test. Synthetic quality reports only synthetic against real train. Neither checks the actual augmented training set against the holdouts, so both can pass despite this exception.

The exact affected run names are in [leakage_affected_runs_2026_09_30.csv](leakage_affected_runs_2026_09_30.csv). Raw direct-comparison evidence is in [leakage_mnist_direct_confirmation_2026_09_30.jsonl](leakage_mnist_direct_confirmation_2026_09_30.jsonl).

## Completed corrected and pilot coverage

| Namespace | Dataset | Seed | Completed downstream records | Exact synthetic/holdout overlap |
| --- | --- | ---: | ---: | --- |
| corrected_v2 | Adult | 42 | 53 | None detected |
| corrected_v2 | Adult | 43 | 53 | None detected |
| corrected_v2 | Census KDD | 42 | 1 | No synthetic table referenced |
| corrected_v2 | Census KDD | 43 | 1 | No synthetic table referenced |
| corrected_v2 | Covertype | 42 | 53 | None detected |
| corrected_v2 | Credit | 42 | 53 | None detected |
| corrected_v2 | Credit | 43 | 1 | None detected; partial coverage |
| corrected_v2 | MNIST12 | 42 | 53 | All 26 TVAE downstream runs flagged |
| pilot_ctgan_v1 | Adult | 42 | 15 | None detected |
| pilot_ctgan_v1 | Covertype | 42 | 15 | None detected |
| pilot_ctgan_v1 | MNIST28 | 42 | 15 | None detected |

The following checks passed across the completed corrected/pilot records:

- Source IDs are unique within partitions, disjoint across train/dev/test, and within the recorded source range.
- All 66 real CSV hashes and row counts match their split manifests and source-ID counts.
- All 66 pairwise raw/one-hot comparisons have zero exact real train/dev/test feature overlap, including dev versus test.
- All 17 distinct referenced generator fit hashes reproduce from real train, including full-table and X-only fits. Recorded generator seed, row count, and fit columns agree with the split records.
- All 313 prediction files use the recorded test IDs in order, and their true labels match the real test targets under the train-fitted label encoder.
- All 313 selected downstream epochs agree with replaying the recorded real-dev loss history and the trainer's actual `1e-5` improvement threshold. Checkpoints are restored before test evaluation in the inspected remote implementation.
- Referenced downstream weights, generator models, and target-predictor artifacts exist; embedded split manifests match the on-disk manifests.
- The inspected remote tabular scaler and one-hot encoder fit on real train and only transform holdouts. DNN target-labeler normalization fits on real train and checkpoint selection uses reserved real dev.

**287 completed downstream runs have no leakage detected by these checks; 26 have the confirmed synthetic feature-overlap exception.** This is a bounded finding, not proof against every possible leakage mechanism.

Detailed evidence: [leakage_remote_2026_09_30.jsonl](leakage_remote_2026_09_30.jsonl) and [leakage_remote_provenance_2026_09_30.jsonl](leakage_remote_provenance_2026_09_30.jsonl). Inspected remote evaluation source is preserved in `leakage_remote_source_2026_09_30/`.

## Simulated benchmarks

All 406 simulated result records reference matching train/test hashes. All 434 distinct referenced synthetic/source files match their saved hashes. The 14 saved simulated train/test file pairs match their manifests. No data-hash mismatch was found.

Categorical Bayesian-network benchmarks contain repeated values across independently intended draws: Asia has 9,986–9,990 matching test rows out of 10,000; Alarm 4,819–4,883; Child 2,742–2,815; Insurance 1,751–1,770. Gaussian-mixture benchmarks have zero exact train/test row matches. Exact equality in a small discrete state space alone does not establish source-record leakage; the intended benchmark estimates performance on fresh IID draws from a known oracle.

The existing simulated production samples have no `*_sample.json` fit-provenance files. The older labeled records also omit an oracle hash. Their saved data can be verified, but generator training settings and the precise historical fit inputs cannot be fully certified retrospectively. The previous simulated audit also documented unsafe cache reuse, repaired in current code; these old results were not regenerated by this review. See [simulated_benchmark_audit.md](simulated_benchmark_audit.md).

Independent oracle regeneration reproduced **all 42 saved train/test/dev tables across all 14 dataset/seed groups**, using distinct random seeds `seed+1`, `seed+2`, and `seed+3`. Categorical states were preserved as strings; mixture numeric values matched to a tolerance of `1e-14` for CSV round-trip precision. This directly supports independent oracle sampling and establishes that the categorical train/test repeats do not require copied source records. Evidence is saved in [leakage_simulated_independence_2026_09_30.jsonl](leakage_simulated_independence_2026_09_30.jsonl).

## Historical results and limits

The earlier [AUDIT.md](AUDIT.md) already established actual historical splitter leakage, train/test row overlap, test-driven checkpoint selection in some legacy trainers, and evaluation-fitted preprocessing. Those saved historical splits and downstream results cannot be certified merely because current code has been corrected. The historical workbooks/reconstructed tables lack sufficient run-level lineage for a clean certification of every cell.

This review checks exact processed feature equality and inspected fit/evaluation paths. It does not exhaustively establish independence of related people, transactions, sessions, temporal groups, near-duplicate images, float32-equivalent transformed inputs, or features that indirectly encode the target. Those require dataset-specific generalization rules and source provenance. Most saved runs also reference historical source hashes that differ from the current remote source; a hash identifies a snapshot but does not by itself preserve its full contents for retrospective review. The follow-up investigation loaded only the user's two known MNIST12 TVAE checkpoints in the original training environment; their file hashes were unchanged afterward.

For the MNIST12 TVAE arms, retain the original scores and flag the overlap rather than silently rewriting them. Decide whether the scientific claim is IID performance with possible repeated values or performance on unseen feature groups. Any strict no-overlap rerun needs a protocol fixed before its new test scores; filtering generated data against the current test set and reusing that same test set would itself make training depend on the holdout.

Reusable read-only audit programs are `scripts/audit_run_leakage.py`, `scripts/audit_leakage_provenance.py`, `scripts/confirm_mnist_leakage_overlap.py`, and `scripts/audit_simulated_split_independence.py`. The local-only record audit reports missing prediction files because those CSVs reside on the server; the authoritative remote audit verifies them successfully.
