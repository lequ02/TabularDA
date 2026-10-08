# Research direction after the October 6 stoppage

Prepared October 6, 2026, about noon Chicago time. This is an analysis and proposed plan, not a changed experiment specification or authorization to restart, cancel, delete, deploy, or launch jobs. Existing results and research settings were preserved.

Recommendation: concentrate on supervised target replacement as a way to improve predictive utility. Treat features-only generation as an ablation until it demonstrates an advantage. Deprioritize Credit and Intrusion for additional production computation; preserve their results and reasons for exclusion. Add California Housing, complete the important matched comparisons, evaluate the teacher and a control without generation, and then add a small, faithful Tab-DDPM comparison. Start writing the methods, results interpretation, and limitations now. Broad superiority and novelty claims need additional evidence.

## Evidence inspected

- The October 6, 11:33 a.m. comparison CSVs/tables and COMPARISON_HANDOFF.md. There are 154 distinct synthetic-report source records and 153 mixed-report records, 296 distinct records in their union. The reported 478 checked metric values are not 478 independent experiments.
- Source mappings, downstream models, DNN labeler, weighted-pilot protocols, Credit/Census diagnoses, and the California Housing and Tab-DDPM implementation notes.
- Read-only server inspection around 11:53-11:57 a.m. Chicago. The root filesystem was 99% full with about 11 GB available. RAM availability was about 27 GB; the GPU was a GTX 1080 Ti with 11 GB memory. Python 3.10.21, torch 2.5.1+cu124, SDV 1.18.0, CTGAN 0.10.2, and scikit-learn 1.5.2 imported successfully; CUDA was available.
- A Credit weighted pilot had been launched at 11:51 a.m. and was actively training its original-data arm. No completed pilot record was present at inspection. Its early development precision was poor despite increased recall; these are not final test results. Do not duplicate or interrupt this job based on a proposal to deprioritize Credit.
- The Census weighted queue was stopped. Its status reported 38 of 106 planned records, three failures, and a stale running field. Logs establish `OSError: [Errno 28] No space left on device`; other late logs contain truncated PyTorch import traces. The inspection does not establish that disk exhaustion caused every historical stoppage, or that dependencies require reinstalling.
- Remote record inventory: corrected_v2 had 504 run JSON files; the MNIST28/News seed-42 namespace had 90; weighted Census had 38. These are file counts, not a fresh validation of every record's checkpoints and predictions. Earlier unweighted Census records cannot fill weighted-report gaps.
- The seven-dataset simulated production collection has 378 unique dataset/seed/method rows, seeds 42/43, 10,000 train/test/synthetic rows, and 300 generator epochs. The local and remote per_run.csv SHA256 agree: `0bc6d31a97584b71da43c5c4a14928dd8641d6bd4424f7306e0a73e2ba0c3db2`. The final server log includes the last Insurance results and the subsequent likelihood-rescoring note. Individual table densities and predictions were not recomputed in this review. The separate simulated_all_verified collection is a one-epoch seed-7 smoke run and is not production evidence.

## What the displayed results support

The following examples use seed 42 and one fixed illustrative labeler, DNN, with full-table relabeling. They do not select the best labeler per test result. Percentages are metric-specific.

| Dataset / metric | CTGAN generated target -> DNN target | TVAE generated target -> DNN target |
| --- | --- | --- |
| Adult / binary F1 | 59.7 -> 68.8 | 59.1 -> 68.2 |
| Covertype / macro F1 | 44.6 -> 72.9 | 43.3 -> 66.1 |
| MNIST12 / accuracy | 51.4 -> 91.7 | 92.9 -> 93.9 |
| MNIST28 / accuracy | 51.5 -> 90.9 | 94.3 -> 95.9 |
| Weighted Census KDD / binary F1 | 43.5 -> 54.2 | 47.0 -> missing |

Adult and Covertype also show DNN relabeling gains for both generators at seed 43. CTGAN MNIST12 repeats show large gains. TVAE digit gains are modest because its baseline is already strong. Several RF relabeling arms fail to improve TVAE, so the claim cannot encompass all labelers.

Mixed training changes the practical question: real training data are still available. Some gains persist, especially Covertype; others nearly disappear. For example, Adult seed-43 CTGAN full+DNN changes binary F1 from 67.618 to 67.692, and MNIST28 seed-42 TVAE full+DNN changes accuracy from 96.516 to 96.577. These are not compelling augmentation gains without uncertainty estimates. Mixing uses all real rows plus 100,000 synthetic rows, not a common 50/50 ratio.

Full-table and features-only DNN relabeling are often similar, and neither uniformly wins. Covertype seed-42 synthetic CTGAN favors full-table relabeling (72.9 macro F1 versus 68.9); MNIST28 favors features-only (92.4 accuracy versus 90.9). Do not claim that removing Y from generator fitting is the demonstrated improvement.

News contradicts a metric-independent superiority claim. Seed-42 CTGAN full+DNN increases R2 from 0.0169 to 0.0284 but worsens NMAE_sigma from 0.2751 to 0.3240. TVAE full+DNN increases R2 from -0.0764 to 0.0311 but worsens NMAE_sigma from 0.3088 to 0.3230. All values use the saved test normalization. The original neural baseline's R2 is -0.2457. Investigate development behavior and compare with training-derived constant predictors and the real-trained teacher; the observed test data must not determine a new primary metric or target transformation.

Covertype's real-only macro F1 is 47.9/53.0, while the DNN teacher has a substantially stronger development result (seed-42 macro F1 87.3). The teacher differs from the downstream model in architecture, class weighting, training duration, and batching. This is evidence that distillation/optimization could contribute, not proof of the cause of the downstream gain or a comparable teacher test result. Evaluate that teacher directly on the same test set using its existing development-selected checkpoint.

## The simulated evidence and the scientific claim

For the existing full-table DNN arm, mean accuracy across seeds 42/43 improves over the corresponding CTGAN and TVAE baselines on all seven simulated datasets: 14 dataset-generator means. This is descriptive, not a significance claim. Gaussian CTGAN improves from 77.72% to 89.33%, close to the expected Bayes accuracy of 90%; TVAE improves from 86.39% to 89.30%.

However, L_test decreases in eight of these fourteen comparisons. Grid TVAE, for example, improves accuracy from 89.01% to 89.62% while L_test decreases from approximately -1.604 to -1.989. Insurance CTGAN full+DNN still has an approximately 48.8% support violation rate despite 94.83% downstream accuracy. Target replacement can repair decision information while leaving distorted or impossible features.

Hard classification relabeling constructs Q_h(x,y) = Q_X(x) 1[y=h(x)]. If the true conditional target distribution is noisy, this removes conditional uncertainty even for a perfect Bayes teacher. Accuracy can improve while joint fidelity deteriorates. With an imperfect teacher or inadequate feature coverage, gains can fail. A universal theorem that the hybrid dominates joint generation is false without substantive assumptions. The reported L_test is a density-refit proxy; its BN epsilon convention is not a normalized generator likelihood or a direct KL identity.

A defensible working claim is: supervised target replacement often improves the predictive utility of generated tables when labeler quality and feature coverage suffice; utility gains need not imply better reproduction of the original joint distribution. The study's potential contribution is a controlled comparison of target replacement versus features-only fitting, with known-distribution evidence explaining benefits and limitations.

A focused mechanistic analysis can reuse saved simulated tables without new generator fits: compare generated targets and teacher targets against the oracle's Bayes decisions on exactly the same generated X, and compute their expected correctness under the oracle conditional probabilities. Distinguish these assigned-target diagnostics from the already reported downstream h_star_agreement. Pair them with existing support violations and L_test to show which part of the joint distribution changes.

## Dataset scope

Use Adult, Census KDD, Covertype, MNIST28, and California Housing as the proposed main set. Keep MNIST12 as a resolution sensitivity check, since it shares MNIST source images with MNIST28. Keep News as a disclosed regression limitation/sensitivity result. California Housing support is locally implemented and was validated in an isolated remote staging directory; no completed production results were found. Verify and selectively deploy the relevant source changes before running it.

Credit can leave the primary comparison on measurement grounds: the fixed holdout has only ten positive cases among 9,992 rows, and seed repeats share those same cases. One additional detected fraud changes recall by ten percentage points. Its CTGAN full RF/XGB gains are actually favorable, so excluding it should explicitly cite inadequate minority evaluation support, not unfavorable performance. Preserve the successful arms, all-negative arms, and the pending weighted-pilot outcome as a case study. Class weighting cannot create additional independent positive test observations. Revisiting Credit requires a separately specified split/protocol, not retroactive replacement of the current test set.

Intrusion can leave the next queue on cost and relevance grounds: the full KDD99 source is large, its corrected comparison is almost empty, and its only displayed real baseline is 20.9% macro F1. Covertype already supplies a substantial multiclass task. Size alone does not invalidate Intrusion; report the omitted scope and unfinished status. Do not relabel it as evidence that the hybrid failed.

California Housing is the first replacement to prioritize. If a new independent mixed-type classification dataset is needed for broader claims, UCI Default of Credit Card Clients is a bounded option (30,000 instances, 23 features); exclude its ID and define categorical roles and minority support before looking at method results. It is a different default-prediction dataset, not the current fraud Credit dataset. Do not automatically add it as another full factorial matrix.

## Minimal next experiment plan

1. Restore operational reliability. Inventory storage and identify safe relocation/cleanup candidates with the owner; preserve completed samples, checkpoints, manifests, predictions, and logs. Establish space for the next job's declared artifacts. Diagnose any reproducible import failure directly. Preserve the active Credit pilot and coordinate with other jobs. No deletion or restart was performed during this review.
2. Freeze the reduced scientific question and evaluation plan before new test evaluations. DNN is a candidate primary labeler selected from exploratory evidence; XGB is a useful robustness arm. Existing GaussianNB/CategoricalNB/PCA-GMM/RF results remain available in supplementary tables. Do not present this retrospective selection as preregistration. Keep the three constructions distinct; use paired, identical full-table feature samples for the decisive generated-target versus replaced-target comparison.
3. Complete missing core CTGAN/TVAE comparisons at seeds 42/43. Prioritize weighted Census's missing full-table TVAE comparisons and seed 43, missing digit repeats, and California Housing. Reuse only provenance-validated fits/tables; use supported classifier resume and verify records plus referenced artifacts. Preserve 500 epochs, batch size 500, and 100,000 synthetic rows. Do not rerun a completed 53-arm block merely because its queue stopped. Census needs its explicitly weighted protocol for all compared arms; keep earlier unweighted results separately identified.
4. Add two decisive controls: direct evaluation of the already fitted teacher, and the same downstream student trained on teacher-labeled resamples of real training features at the same synthetic-row budget. The latter tests whether new generated feature coverage contributes beyond target replacement and training volume. Include real-only student and the unchanged generated-target arm. Match student settings, selection, and declared update budget within these comparisons; do not silently replace existing models or loss rules. A simple augmentation/distillation reference can follow if needed by the selected paper claim.
5. Quantify uncertainty from saved predictions, pairing comparisons on the same held-out rows and accounting for repeated feature groups where relevant. Report dataset-wise effects and seed variability. Two model seeds on a common holdout do not establish split robustness, and 478 metric values are not independent replicates. After the key comparisons are complete, add three more model-seed repeats only for primary arms if feasible; independent holdouts answer a different question and require a separate prospective split protocol. New Housing results supply evidence not used to choose the method. Do not choose labelers, thresholds, metrics, or epochs from these new test scores.
6. Add a limited Tab-DDPM study on Adult, Covertype, and California Housing, seeds 42/43, with generated targets and replacement targets on identical sampled features, synthetic and mixed evaluation, and the frozen primary labeler. Assess declared production runtime/storage first. Report optimizer steps and sampling cost, not fictitious epoch equivalence. The existing unconditional adapter passed small checks but has no production utility results. The published classification method is class-conditioned; include that native baseline for any claim against original Tab-DDPM. An unconditional joint adapter is a matched-factorization experiment, not a reproduction of the published classification method. Features-only diffusion is optional unless the paper claims that construction is portable. Do not expand to the earlier 436 additional downstream evaluations by default.

The completion criterion is a coherent, honestly scoped study with matched core comparisons, meaningful controls, and disclosed uncertainty/failures, not completion of every historical queue. If Tab-DDPM gains are small or absent, publish the boundary of the method's usefulness rather than changing datasets or selection rules to force superiority.

## Prior work and writing

The generic strategy of labeling generated features with a teacher predates this project. GAN-assisted teacher-student compression studies tabular data. FAST-DAD explicitly compares a CTGAN-based synthetic-feature arm with teacher probability targets. DisTab uses teacher-labeled augmented tabular features for pretraining. Differences in teachers, hard versus soft targets, generator families, and evaluation do not by themselves establish a novel general principle.

- [Liu, Fusi, and Mackey, Teacher-Student Compression with Generative Adversarial Networks](https://arxiv.org/abs/1812.02271).
- [Fakoor et al., Fast, Accurate, and Simple Models for Tabular Data via Augmented Distillation, especially section 5.1](https://proceedings.nips.cc/paper_files/paper/2020/file/62d75fb2e3075506e8837d8f55021ab1-Paper.pdf).
- [Wang, Fu, and Ciliberto, Deep Tabular Learning via Distillation and Language Guidance](https://openreview.net/notes/edits/attachment?id=wQueUlkQwR&name=pdf).
- [Kotelnikov et al., TabDDPM, section 4 and experimental protocol](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf).
- [Official Tab-DDPM configuration](https://github.com/yandex-research/tab-ddpm/blob/main/CONFIG_DESCRIPTION.md).
- [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients).

Write the current paper as an evidence-led study of predictive utility and target fidelity. Use the simulated benchmark to explain mechanism, real data to establish practical behavior, and controls to separate generation from distillation. Preserve News tradeoffs and excluded-dataset disclosures. Published CTGAN reference lines use different splits/evaluators and cannot establish controlled superiority. If the claim extends to arbitrary downstream models, add a separately specified small second-evaluator robustness study; the current neural-only design supports claims about those evaluated neural students.
