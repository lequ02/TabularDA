# Audit of the simulated CTGAN/TVAE replication

## Main finding: the paper's TVAE BN summary conflicts with its supplement

The main paper's Table 2 reports TVAE Bayesian-network averages of L_syn=-6.76 and L_test=-9.59. Supplement Table 3 lists:

| Dataset | Supplement TVAE L_syn | Ours, mean of seeds 42/43 | Supplement TVAE L_test | Ours, mean of seeds 42/43 |
|---|---:|---:|---:|---:|
| Asia | -2.31 | -2.298634 | -2.27 | -2.247758 |
| Alarm | -11.2 | -11.165835 | -10.7 | -10.809658 |
| Child | -12.3 | -12.421622 | -12.3 | -12.421243 |
| Insurance | -14.7 | -14.713492 | -14.2 | -14.020688 |
| Mean | **-10.1275** | **-10.149895** | **-9.8675** | **-9.874837** |

The supplementary averages are computed from rounded published numbers. Their discrepancy with main Table 2 is much larger than rounding permits. The apparent TVAE BN gap of 3.39 in L_syn is therefore misleading: compared with the supplementary dataset scores, our average differs by about 0.0224. L_test differs by about 0.0073. This establishes an internal reporting inconsistency; it does not establish which unpublished underlying result the authors intended.

Sources: [main paper, Table 2](https://papers.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf) and [official supplementary archive, Table 3](https://papers.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Supplemental.zip). The extracted supplement is saved as ctgan_2019_supplement.pdf in this folder. Its GM table also appears to mislabel the second neural-model row as TVAE; its numbers match the main table's CTGAN averages. That row identification is an inference.

## Actual saved-run checks

The Linux run at /home/thuy/Research/minh_data_synth/TabularDA/SDGym-research/data/simulated_paper contains 406 records: 29 methods x 7 datasets x 2 seeds. The two seeds are 42 and 43. Paper-model records specify 10,000 train/test/synthetic rows, 300 epochs, batch size 500, CUDA, ctgan 0.10.2, torch 2.5.1, and pgmpy 0.1.26. These are recorded settings; the generator did not save independent provenance sufficient to prove the epoch count of a reused sample.

- The means reconstructed from the raw CSV match the pasted summary to its displayed precision.
- All 406 saved L_syn scores recompute exactly from the current saved samples and oracles.
- All 42 paper-baseline L_test scores recompute within 1e-8.
- Sixteen accuracy/macro-F1 results were independently recomputed: seed 42, Grid and Insurance, both generators, generated-target/RF/XGB/DNN variants. Every score matched exactly.
- All 770 referenced synthetic/source hashes matched, as did split/oracle checks.
- The Alarm, Child and Insurance BIF probabilities agree with the historical pomegranate oracle metadata to approximately 4e-15 in log probability. Historical Asia metadata was not available in this workspace.
- Separate local checks verified GM log density against SciPy, BN joint probability against scalar CPD lookup, and BN refitted CPDs against direct empirical counts. No numerical likelihood bug was found.

The raw per-run results and audit diagnostics are saved in remote_simulated_methods_per_run.csv and remote_simulated_run_audit.json.

## Concrete bugs found and fixed

1. **CategoricalNB destroys rare binary indicators.** When fewer than 10% of rows contain 1, the training quantiles can all be 0. Both digitize(0,[0]) and digitize(1,[0]) then produce the same code. The fix keeps binary indicators binary, including in saved preprocessing artifacts. This affects labeled categorical variants; it cannot explain paper CTGAN/TVAE rows.
2. **RF labels depend on previous operations.** The labeler used random_state=None. A regression test changed 7 of 100 predictions merely by changing ambient RNG state. RF/XGB now receive the requested experiment seed through the existing Ensemble class.
3. **Generator cache accepts different settings or training values.** An existing sample could be reused when switching from 1 to 300 epochs, or after changing the training values, provided the schema and row count matched. Generation now records its training-table hash, seed, rows, epochs, device and sample hash, and checks them on reuse. Legacy samples without those records are rejected. The audited production run has no such sample metadata, so stale generation settings cannot be ruled out retroactively.
4. **Cached-score and summary identity checks were incomplete.** Cached likelihood now checks the requested method and split/oracle hashes. Labeled results also check their identity and oracle hash. Summary records must match their method/dataset/seed paths.

The non-DNN labeling path now uses training rows for its unused dev argument instead of exposing test rows. The DNN path still uses its independent oracle dev sample. No evidence of test-label leakage was found in the inspected labeler implementations.

The final production changes use direct checks and estimator parameters. No new exception-catching recovery paths, broad code/package fingerprinting or fallback modes were added. The task-specific server backup was removed at the user's request. Existing production samples and results were not overwritten.

## Remaining experimental differences

CTGAN 0.10.2 is a modern implementation rather than a frozen 2019 implementation. The historical CTGAN code here uses zero discriminator weight decay; the modern default is 1e-6. Continuous preprocessing now uses RDT, and discrete ordering is learned from training data. Both TVAE implementations use loss_factor=2; a different reconstruction-loss weight is not an explanation.

Our GM evaluator uses five seeded GMM initializations. Re-evaluating the same saved samples with one initialization lowered family-average L_test by 0.0278 for CTGAN and 0.0768 for TVAE. This is a measured evaluator contribution, but it is too small to explain TVAE's full GM difference from main Table 2. TVAE Grid L_test is -4.5203 here versus -11.26 in the supplement; Grid is the principal source of that difference. A controlled historical-versus-modern generator experiment would be needed to assign the remaining gap to a specific model or preprocessing change.

CTGAN's Insurance samples contain 53.10% and 52.57% oracle-impossible rows on seeds 42 and 43; TVAE's corresponding rates are 11.30% and 10.63%. The existing log(p+1e-8) evaluator correctly penalizes them. The CPD replacement warning occurs while refitting conditional tables and is expected; it is not evidence of accidental mutation of the original scoring oracle.

The labeled GM task has a deterministic linear boundary. RF/XGB/DNN relabelers learn that task using original training labels, explaining why their held-out accuracy can approach 100%. The paper has no matching prediction baseline for that extension.

## Validation

The five initial reproductions failed before the fixes. The final suite has 18 passing tests on both the local machine and the Linux server. A fresh CPU smoke run completed Grid and Asia with seven methods each, and a subsequent cache-reuse run completed successfully. Those one-epoch runs verify execution, not final model quality. Corrected full-quality results require a new root and rerunning the affected experiments; the old plotted results remain the audited pre-fix observations.
