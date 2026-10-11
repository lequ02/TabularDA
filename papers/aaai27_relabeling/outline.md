# When Should We Replace the Targets in Synthetic Tabular Data?

Paper plan, October 11, 2026 (Chicago). This is an outline with draft exhibits, not a finished manuscript.

## The argument

Treat target assignment as an experimental choice in synthetic-data evaluation. A generator supplies feature rows; either the generator or a predictor trained on real data supplies their targets. Ask whether changing the targets improves downstream prediction, whether fitting the generator with the target still matters after relabeling, and whether better prediction also means a better approximation to the original distribution.

The strongest comparison keeps each full-table generator's feature rows fixed and replaces only their targets. Make that comparison the center of the paper. A features-only generator is a second construction: it changes the feature distribution as well as the targets and cannot isolate the effect of target replacement.

The current evidence supports a bounded empirical paper. Five selected classification tasks have positive average relabeling gains, with very different magnitudes. Housing improves on both squared-error and absolute-error measures; News does not. The known-distribution benchmark shows cases where prediction improves while the density-refit score worsens. Do not frame this as a new generative architecture or a general guarantee of improvement.

Working title: **When Should We Replace the Targets in Synthetic Tabular Data?** A more descriptive alternative is **Target Assignment in Synthetic Tabular Data: Predictive Utility and Distributional Fidelity**.

## Scope and evidence

Use the current seven-task real-data selection: Adult, Census KDD, Covertype, MNIST12, MNIST28, News, and California Housing; CTGAN/TVAE; RF/XGB/DNN labelers; seeds 42/43. Keep synthetic-only and real-plus-synthetic training separate. This selection contains 406 distinct downstream records: 7 tasks x 2 seeds x (1 real-only + 14 synthetic-only + 14 mixed). The 14 synthetic constructions are 2 generated-target baselines plus 2 generators x 2 feature sources x 3 labelers. The same 14 real-only records appear in both training-regime displays. The selected grid is complete in the frozen source; this says nothing about completion of the original 816-run matrix.

Use the separate known-distribution study in `simulated_eval/README.md`: Gaussian, Grid, Ring, Asia, Alarm, Child, Insurance; two seeds; 27 methods per task/seed; 378 recorded evaluations. Main exhibits use RF/XGB/DNN to match the real-data comparison. Include all six simulated labelers in the supplement, including underperforming arms. Do not use the older RF-evaluated, deterministic-boundary simulated extension under `output/comparisons/simulated_methods/`: it is a different study.

Real scores come from the frozen two-contrast analysis checked remotely on October 10, 2026, at 11:35:57 p.m. Chicago. Simulated scores match the October 9 coverage-audit hash. This plan does not refresh experiments or claim current live-job completion.

## Section titles, contents, and placement

| Section | What belongs here | Main exhibit | Target content pages |
|---|---|---|---:|
| 1. Introduction | State the target-assignment question; distinguish prediction from joint-distribution fidelity; give three contributions with bounded scope. | Brief reference to the two empirical studies | 0.85 |
| 2. Related Work | Tabular generators; synthetic-data utility; teacher predictions/distillation. Explain what the fixed-feature comparison adds. | No survey table | 0.40 |
| 3. Target Assignment in Synthetic Tables | Define the three constructions and the two contrasts. Explain what hard relabeling does to a stochastic target. | Figure 1; Table 1 if space permits | 0.85 |
| 4. Experimental Protocol | Data, partitions, generator budgets, labelers, downstream networks, metrics, and how averages are formed. | Dataset details in supplement | 0.85 |
| 5. Results on Real Data | 5.1 Does replacing targets help? 5.2 Does joint generation help after relabeling? 5.3 Regression depends on the loss. | Table 2; Figure 2; Table 3 | 1.90 |
| 6. Prediction and Fidelity under Known Distributions | Present Bayes agreement, accuracy, and the two density scores together; explain the cases where they disagree. | Table 4; Figure 3 | 1.40 |
| 7. Limitations | Task selection, two seeds/shared holdouts, MNIST collisions, labeler/student controls, and density-refit assumptions. | No new headline result | 0.60 |
| 8. Conclusion | Answer the two questions at the scope supported by the experiments. | None | 0.15 |

The budget totals seven content pages, including floats. It is a writing target, not a claim that the current skeleton compiles to seven pages. Put extended tables, protocols, derivations, and exploratory ANOVAs in a separate supplement. Keep the central control comparison, negative regression result, and principal caveats in the main text.

### 1. Introduction

Candidate opening:

> A synthetic training table needs both features and targets. CTGAN and TVAE can generate them together, but the target can also be assigned by a predictor trained on the real training set. These choices need not produce the same downstream behavior. We compare them while separating two questions: whether a predictor provides more useful targets on a fixed set of generated features, and whether the generator should see the target when learning those features.

Then explain why predictive usefulness and faithful joint sampling differ. Name the real-data and known-distribution studies without cataloguing every method in the opening. Finish with three concrete contributions: the fixed-feature target comparison; the joint-versus-features-only comparison after relabeling; and the oracle study that separates boundary learning from stochastic-target fidelity. Calling these contributions does not establish novelty: complete the closest-prior-work review before claiming a new method.

### 2. Related Work

Keep three short paragraphs, each tied to a comparison the paper actually makes:

1. CTGAN/TVAE as the two evaluated generator families; TabDDPM as relevant modern context. A queued TabDDPM study is not a paper result.
2. Evaluations of synthetic data for sharing and augmentation; distinguish the synthetic-only and mixed uses.
3. Teacher predictions, pseudo-labeling, and distillation. Here predictors label generated covariates and we evaluate a fresh downstream learner. Hard class assignments differ from soft distillation targets. Establish the closest prior use of this exact construction before asserting originality.

Verified starting references are in `references.bib`; this is a starting bibliography, not a completed literature review.

### 3. Target Assignment in Synthetic Tables

Let D_tr be real training data, G_XY a generator fitted to (X,Y), G_X a separate generator fitted to X, and h a labeler fitted to D_tr. Define:

| Draft Table 1: construction | Generator fit input | Synthetic features | Synthetic targets | Comparison identifies |
|---|---|---|---|---|
| Joint / generated | (X,Y) | X from G_XY | Y from the same sample | Generated-target baseline |
| Joint / predicted | (X,Y) | Exactly the baseline X | h(X) | Target replacement on fixed features |
| Features-only / predicted | X | X from G_X | h(X) | A different generator fit after relabeling |

For a higher-is-better score S, the clean target contrast is S(joint/predicted) - S(joint/generated), matched by task, generator, seed, labeler, and training regime. The feature-source contrast is S(joint/predicted) - S(features-only/predicted), with the same matching factors. Reverse the benefit direction when interpreting NMAE, or present its signed native difference with an explicit lower-is-better label.

Hard class relabeling replaces a generated conditional with a point mass at h(X). Even if h equals the Bayes decision rule, this does not recreate a nondegenerate P(Y|X). In the mixed benchmark, the Bayes rule still has observed-target accuracy 0.90 because of independent 10% label noise. For any test classifier f, its error against Y equals 0.10 + 0.80 times its error against h*. Present this identity as an interpretation of that benchmark, not a general new theorem. Regression point predictions also omit conditional target variability; do not infer that this always changes utility in the same direction.

### 4. Experimental Protocol

Real data: partition source rows before fitted preprocessing. Describe feature-group splitting, fixed split state 42, rare-class protection, training-fitted transforms, and actual manifest counts. Seed 43 repeats model randomness rather than creating a new holdout. State 500 generator epochs, batch 500, 100,000 synthetic rows; downstream batch 128, learning rate .001, at most 100 epochs, patience 30, and development-loss selection, subject to the saved dataset-specific records. RF/XGB/DNN are labelers; downstream evaluators are dataset-specific neural networks. Mixed training concatenates all real training rows with all 100,000 synthetic rows, so its ratio varies by task.

The selected protocol has dataset-specific revisions: weighted Census KDD, corrected ten-class MNIST evaluation, log-target News with raw-scale reported errors, and the Housing rerun that retains coordinates without Faker substitution. Use their saved protocol names and records; do not imply a single unchanged historical implementation. Extract full hyperparameters, preprocessing and split sizes from saved provenance for the supplement. Avoid presenting nominal 72/8/20 proportions as actual counts.

Classification: use macro F1 consistently in the cross-task paper table, including MNIST. Existing overview plots use other metrics on some panels; do not relabel those scores as macro F1. Supply binary F1, accuracy, and class-specific summaries in the supplement. Regression: R2 and saved NMAE_sigma in the main table, D2 absolute-error in the supplement. NMAE_sigma is MAE divided by the population standard deviation of the same real test targets; read the saved normalization fields.

Known distributions: separate independent 10,000-row train/test draws and 10,000 synthetic rows; 300 generator epochs, batch 500; fresh standardized DNNs; the fixed six labelers and density estimators documented in the benchmark README. Accuracy uses observed targets, Bayes agreement uses h*, L_syn scores synthetic rows under the oracle, and L_test scores original test rows under a normalized density refitted to the training table. L_test is not a generator likelihood. BN scores apply log(p+1e-8) once to the full joint; track exact-zero support violations and keep the normalized-density KL identity separate from that adjusted score.

### 5. Results on Real Data

**5.1 Does replacing generated targets help?** Use Table 2 for absolute levels and Figure 2a for the fixed-feature differences. Present each task before an overall average. Generated-target baseline means average four task-specific generator/seed scores; each relabeled-source mean averages twelve generator/seed/labeler scores; real-only means average two seeds. These are descriptive method averages, not twelve independent datasets. Include the real-only baseline so improvement over a generator is not mistaken for improvement over real training.

**5.2 Does joint generation help after relabeling?** Figure 2b contrasts joint/predicted with features-only/predicted. The frozen exploratory analysis estimates mean differences of +0.357 and +0.138 macro-F1 percentage points for synthetic-only and mix. Its failure to detect a difference is not an equivalence test.

The existing two-contrast inferential summary averages BOTH relabeled feature sources before comparing with generated targets: +11.63 macro-F1 points for synthetic-only (raw p=.034977; Holm2=.069953; Holm4=.139907) and +6.77 for mix (raw p=.056598; Holm2=.113197; Holm4=.169795). Its exact sign-flip p=.0625 in both regimes. These tests use five task means, not hundreds of independent runs, and are exploratory because the questions were selected after earlier analyses. Do not attach these p-values to the fixed-feature-only differences in Figure 2a: that is a different contrast. The drafts add no new significance tests. If both training regimes supply headline inferential claims, report the four-test family as well as the within-regime family. Keep factorial ANOVAs as supporting exploratory results in the supplement.

**5.3 Regression depends on the loss.** Show News and Housing separately in Table 3. Housing improves in both R2 and NMAE; News relabeling can improve R2 while worsening NMAE and D2. Explain this as disagreement between observed squared-error and absolute-error outcomes, not proof of a mechanism not measured. News downstream utility remains weak in absolute terms. Do not pool regression and classification metrics or treat small increases from negative R2 as strong predictions.

### 6. Prediction and Fidelity under Known Distributions

Use Table 4 for all seven tasks and Figure 3 for the method-level disagreements. The current RF/XGB/DNN joint-relabeling means improve observed-target accuracy on all seven tasks. L_test worsens on Gaussian, Grid, Ring and Asia and improves on the other three (Alarm's mean is close to zero). On Gaussian, the mean accuracy gain is 7.29 points and Bayes agreement gain is 9.09 points, while L_test falls by 0.0195. On Asia, accuracy gains only 0.22 points while L_test falls by 0.7728. These are descriptive means across two generators, two seeds and three labelers; plot every task and method rather than highlighting only favorable examples.

Explain how concentrated hard targets can increase L_syn without restoring the original stochastic conditional. The density-refit family, regularization, and BN epsilon convention also affect L_test, so these measurements do not by themselves identify a universal utility-fidelity tradeoff. In the continuous mixed examples, the known 10% noise connects accuracy and Bayes agreement exactly in expectation. In BNs, posterior uncertainty comes from the original CPDs and expected Bayes accuracy is estimated on independent test features.

### 7. Limitations

The main paper needs four direct points:

- The tasks were selected, there are two model seeds, and MNIST12/28 are related representations. Shared holdouts and reused generator baselines do not create independent replication.
- MNIST12 TVAE tables have documented exact feature collisions with dev/test patterns. Train-only fitting was supported by the audit; collisions do not prove holdout ingestion. The results nevertheless lack a strict unseen-feature guarantee. Preserve this caveat and the affected-source provenance. A sensitivity analysis excluding these arms and one excluding both MNIST representations would help; neither is included or newly computed here.
- A teacher trained on real data may already predict well. Compare the teacher directly on the held-out test set, using a checkpoint chosen without test access, and compare with the real-only downstream network to determine what the synthetic stage adds. DNN labeler/downstream similarity can otherwise resemble distillation. Direct labeler-test metrics are not in these draft tables.
- Hard labels change conditional uncertainty; density refits are proxies; no privacy guarantee was evaluated. Broader generators, additional seeds, soft-label or posterior-sampling arms, and sensitivity to synthetic sample count remain possible follow-ups, not established results.

Housing's real-only baseline is much weaker than its relabeled arms. Discuss the existing baseline/protocol investigation in the supplement; do not silently tune it using test scores. The current dropout trials are separate and excluded from these sources.

### 8. Conclusion

Answer each question in two or three sentences. Target replacement can improve the evaluated learner, but its size depends on the task and metric. The current evidence does not establish an advantage or equivalence for including Y in generator fitting after relabeling. Improvements in prediction do not certify recovery of the original joint distribution.

## Draft tables and figures

The numerical drafts are in [results_tables.md](results_tables.md), with CSV and LaTeX versions under `tables/`. All values are actual frozen-source values or simple descriptive averages; no cells contain invented results.

| Exhibit | Contents and intended placement | Draft caption / reading rule |
|---|---|---|
| Table 1 | Three constructions; Section 3 | Target assignment separates joint generation, fixed-feature relabeling, and a separate features-only generator. Can be removed if Figure 1 supplies the same definitions. |
| Table 2 | Five classification tasks; real-only and three constructions; synthetic-only/mix; Section 5.1 | Macro F1 (%) averaged over the stated generator/seed/labeler cells. Joint relabeling keeps baseline features fixed. Report absolute levels, not best test-selected labelers. |
| Table 3 | News/Housing; R2 and NMAE; both regimes; Section 5.3 | R2 is higher-is-better; NMAE_sigma is lower-is-better. Targets are evaluated in their original scale. Averaging and baseline counts match Table 2. |
| Table 4 | Seven known distributions; generated-target accuracy and matched joint-relabeling changes in accuracy, Bayes agreement, L_syn and L_test; Section 6 | Changes are relative to the matching generator/seed baseline. Likelihood differences are within each dataset; no pooled density scale. Main labelers are RF/XGB/DNN. |
| Figure 1 | Vector pipeline with shared-feature branch; Section 3 | Only the joint/generated versus joint/predicted branch holds X fixed. All labelers train on real training data. |
| Figure 2 | Task-level fixed-feature target changes and joint-minus-features-only changes; Section 5 | Each small point is one generator/seed/labeler combination, with synthetic-only and mix separated; large symbols are task means. Shared baselines and task counts preclude interpreting these points as independent task replications. |
| Figure 3 | Seven separate panels: matched delta L_test against delta accuracy for joint relabeling; Section 6 | Positive vertical changes mean better prediction; negative horizontal changes mean worse density-refit score. Shape identifies the generator and color the labeler. Each point averages two seeds, with no fitted trend or causal claim. |

The storyboard PDF places the section map first, then the construction diagram, real-data results, regression, oracle results, and evidence gaps. It is a review document, not a PDF rendered from the AAAI style. `paper.tex` is the editable template-based skeleton; it needs PDFLaTeX to compile. Neither the local machine nor the research server had a LaTeX compiler during this preparation. The supplied style and bibliography-style files are copied unchanged and hash-checked against the ZIP.

## Supplement plan and remaining evidence

| Supplement | Include | Status |
|---|---|---|
| A. Data and preprocessing | Manifest row counts/source IDs, features/classes, target definitions, dataset sources/licenses, fixed MNIST transformations | Saved provenance exists; full dataset inventory still needs manuscript extraction |
| B. Model and training settings | Generator settings/hashes, labeler hyperparameters, downstream architectures/losses, development selection, resources/runtime | Saved records and source exist; consolidate without mixing protocol revisions |
| C. Full real results | Per-generator/seed/labeler tables, all classification metrics, D2 and saved regression normalization, real-only rows | Current frozen scores available |
| D. Full oracle results | All six labelers, both feature sources, both seeds, Bayes ceilings, support violation rates, precise density refits | Current 378-row table available |
| E. Statistical checks | Exact task contrasts, the existing Holm families, seed differences, exploratory ANOVAs | Existing analysis is exploratory; no new tests in this outline |
| F. Validity checks | Fit provenance, MNIST collision investigation, revised Census/MNIST/News/Housing protocols, sensitivity results when available | Caveats retained; proposed sensitivities have not been run |
| G. Interpretation | Mixed-noise error identity, point-mass conditional under hard relabeling, scope of KL identity | Mathematical observations, not evidence of empirical causality |

Before a submission draft, resolve closest prior work, define the primary estimand, inspect direct teacher performance, and prepare the MNIST/task-family sensitivities. More seeds and a completed diffusion comparison would broaden the evidence; they are not prerequisites for writing this bounded outline and are not authorized experiment launches here. Do not add soft-label, random-label or sample-count experiments simply to populate a checklist.

## Source ledger and author-kit constraints

Local source files:

- `output/statistical_two_contrasts_20261010/frozen_inputs/scores.csv`: 3,770 score rows from 406 selected records; the builder checks its stored SHA-256.
- `output/statistical_two_contrasts_20261010/task_contrasts.csv` and `classification_contrasts.csv`: existing contrasts and inferential summaries; p-values are copied, not re-estimated.
- `output/statistical_two_contrasts_20261010/analysis_metadata.json` and `HANDOFF.md`: source verification time, design, multiplicity and limits.
- `output/simulated_all_20260930/per_run.csv` and `config.json`: the current seven-distribution study, not the older simulated extension. Its hash matches `audit/completed_simulated_coverage_2026_10_09.json`.
- `simulated_eval/README.md`: oracle definitions, metrics, refitting families and budgets.
- `audit/LEAKAGE_REVIEW_2026_09_30.md` and `audit/COMPARISON_HANDOFF.md`: collisions, provenance and protocol revisions. Historical counts in these files are not current completion claims.
- `papers/AuthorKit27.zip`: supplied source of `aaai2027.sty`, `aaai2027.bst`, anonymous-submission preamble and checklist.

The official [AAAI-27 submission instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/) specify seven content pages and at most two additional pages reserved for references, anonymous review, and a separately uploaded reproducibility checklist. Use PDFLaTeX, US Letter, the unmodified provided style, and 9-10 point table text. Put table and figure captions below. Export figures externally as PDF/PNG; avoid forbidden layout/font/spacing changes. The supplied kit's suggested checklist inclusion does not override the track's separate-upload instruction.

Starting literature links: [CTGAN/TVAE](https://proceedings.neurips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html), [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html), [On the Usefulness of Synthetic Tabular Data Generation](https://arxiv.org/abs/2306.15636), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). These identify relevant starting points, not an exhaustive novelty search.

Rebuild draft exhibits with `python papers/aaai27_relabeling/build_exhibits.py`. This only reads small saved score tables and prepares artifacts locally; it performs no fitting, retraining or inferential analysis. Compile `paper.tex` from this directory with PDFLaTeX, BibTeX, then PDFLaTeX twice in a TeX installation with the required packages. See `validation.json` for hashes and focused checks.
