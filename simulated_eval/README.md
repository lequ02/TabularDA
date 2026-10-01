# Known-distribution simulated benchmark

Standalone benchmark; it imports no code from SDGym or SDGym-research. Seven datasets share the same generator, relabeling, and downstream evaluation pipeline.

## Mixed Gaussian, Grid, and Ring

Each table contains continuous `feature_0`/`feature_1` (X0/X1), binary `feature_3` (C), and binary `target` (Y). Only the continuous distribution changes:

- **Gaussian:** X0 and X1 are independent standard normals.
- **Grid:** independently choose one of five centers {-4,-2,0,2,4} for each coordinate, add independent N(0, spread²) noise, then divide by sqrt(8 + spread²). This gives 25 equally weighted modes and independent coordinates.
- **Ring:** choose one of eight equally spaced unit-circle centers, add isotropic N(0, spread² I) noise, then divide by sqrt(1/2 + spread²). The coordinates share the selected center and are dependent. Ring deliberately tests dependence beyond the independent-coordinate scheme.

Both coordinates have population mean zero and variance one. Standardization uses the known population scale, never fitted sample statistics. Default spread is 0.20.

All three use the same conditional category and noisy target:

    C | X ~ Bernoulli(sigmoid(kappa * X0 * X1))
    h*(X,C) = 1[X1 + (1 + beta * (2C-1)) * X0 > 0]
    Y = h*(X,C) XOR E,  E ~ Bernoulli(noise), independent of X,C

Defaults: kappa=2, beta=0.75, noise=0.10. Every feature affects the target. The exact normalized joint density is

    p(x,c,y) = p_X(x) * p(c|x) * p(y|x,c).

`p_X` is the stated Gaussian or Gaussian mixture, not a fitted estimate. Computation uses log space without probability floors. The mixed measure is Lebesgue measure for X and counting measure for C/Y.

For noise < 1/2, h* is Bayes optimal and its expected accuracy is 1-noise. For any fixed classifier h on independent test data:

    P(h != Y) = noise + (1 - 2*noise) * P(h != h*).

Thus agreement with h* separates boundary error from label noise. Reflection symmetry balances the population targets in all three datasets. C is dependent on the feature pair but marginally independent of either coordinate alone. Ring does not make X0 and X1 independent.

## Bayesian networks

Asia, Alarm, Child, and Insurance use the packaged BIF graphs and conditional probability tables in `oracles/`. These retain the original BN distributions rather than adding the mixed benchmark's category or target rule.

| Dataset | Target node |
|---|---|
| Asia | dysp |
| Alarm | BP |
| Child | Disease |
| Insurance | Accident |

Every other node is an observed feature. States are encoded by their CPD order as integers; this encoding carries no ordinal meaning. Samples are drawn ancestrally in topological order. The exact joint probability is the product of all original CPDs. No extra label noise is added; uncertainty comes from those CPDs.

The oracle computes P(Y=y | all observed features) by enumerating target states and normalizing their joint probabilities. h* is its argmax, with ties assigned the lowest encoded state. The accuracy reference is the mean maximum posterior probability on the independent test features, an estimate of the population Bayes accuracy. Accuracy and h* agreement can differ even with no added noise.

BN CPDs may contain exact zeros. A synthetic row violating any of them has L_syn = negative infinity. We preserve that result and report the impossible-row fraction alongside it; plots do not invent a finite reference line for an infinite baseline.

## Methods

Each dataset/seed has 27 methods by default:

- `original`: original process samples, used as the training table.
- `ctgan`, `tvae`: generate the complete table and keep generated targets.
- `{generator}-joint-{labeler}`: keep the complete-table generator's features and replace its targets.
- `{generator}-features-{labeler}`: fit a separate generator on features only and assign targets afterward.

All six labelers train once on original training features and targets: Gaussian NB, Categorical NB, PCA-GMM, RF, XGB, and DNN. They assign hard labels, so they do not reproduce independent label noise. Relabelers within a source group receive exactly the same generated features.

Categorical NB quantile-bins continuous features into ten bins and leaves categorical states discrete. Other labelers one-hot encode categorical features using the oracle's known state schema. Gaussian NB operates on this encoded input. PCA-GMM standardizes it, projects to two PCA components, and fits four full-covariance mixture components per observed target class with empirical priors. RF uses 100 trees. XGB uses 100 depth-six trees, learning rate 0.1, and supports multiclass targets. DNN uses two 64-unit hidden layers, batch size 256, at most 100 iterations, and early stopping on an internal 10% training split.

## Metrics

Every training table, including `original`, gets the same five metrics:

- **L_syn:** mean log p(z) on that table, using the fixed exact original distribution.
- **L_test:** fit a normalized density q to that table, then calculate mean log q(z) on independent original test rows.
- **Accuracy:** a fresh DNN trained on that table predicts the original test targets.
- **Macro F1:** unweighted mean F1 over all oracle target classes, for the same predictions.
- **h* agreement:** those predictions compared with the oracle's Bayes decisions on the same test features.

All downstream classifiers are fresh standardized DNNs with the architecture above. RF is only a relabeler. Preprocessing, early stopping, and density fitting use the training table only. The original test set is reused across methods within each dataset/seed.

L_syn measures plausibility under P; concentration in high-density regions can improve it despite missing modes. The original baseline estimates E_P[log p(Z)], not a maximum score. L_test reverses the scoring direction and exposes missed original-test regions, subject to the chosen refit family. It is a density-refit proxy, not CTGAN/TVAE's own likelihood, and not claimed to duplicate the paper's estimator.

The L_test refits are explicit:

- Gaussian: one diagonal Gaussian; Grid: independent five-component univariate mixtures; Ring: eight-component spherical two-dimensional mixture. Mixtures use covariance regularization 1e-6.
- Mixed C|X: logistic intercept and X0*X1 coefficient, fitted with a unit Gaussian coefficient prior.
- Mixed Y|X,C: minimize training classification error exactly over beta in (0,1), choosing the midpoint of the first best interval. Estimate noise with the Beta(1/2,1/2) posterior mean, constrained to at most 1/2.
- BN: keep the known graph and refit every CPD using a fixed Dirichlet(1/2) prior for every child state in every parent configuration.

These are fixed statistical estimators, not error-handling fallbacks. The BN prior makes refitted probabilities positive even when a training count is zero; the exact original probabilities used for L_syn remain unchanged. With unlimited original samples, a correct refit approaches P. For a normalized q, E_P[log q] = E_P[log p] - KL(P || q); finite samples and the restricted family affect the estimate.

## Run

From the repository root, install `simulated_eval/requirements.txt`, then:

```powershell
python -m simulated_eval --output output/simulated_all --device cuda
```

Defaults: all seven datasets, seeds 42/43, 10,000 train/test/synthetic rows, 300 generator epochs, batch size 500. GPU selection applies to CTGAN/TVAE; labelers and evaluation DNNs run on CPU. Use `--datasets grid ring asia` to select datasets, `--labelers rf xgb` to select labelers, or `--labelers` alone for generator baselines only.

Small execution check, not a quality experiment:

```powershell
python -m simulated_eval --output output/simulated_all_smoke --seeds 7 --train-rows 2000 --test-rows 500 --synthetic-rows 2000 --epochs 1 --batch-size 500
```

The output directory must be new. Failures surface directly; there is no resume path, automatic device selection, or alternate generator.

Outputs: `config.json`, `per_run.csv`, `summary.csv` (seed means), and a five-panel `comparison.png`/`.svg` per dataset. Tables live under `<dataset>/seed_<seed>/`. Individual seeds appear as open circles; filled circles are means. Colors distinguish original, generator baseline, joint relabeling, and features-only relabeling. CTGAN/TVAE baseline means are vertical dashed/dotted references; oracle expected accuracy is marked separately. Finite test accuracy can exceed its expected reference.

Four focused checks cover exact probabilities, normalized refits, and evaluation data flow:

```powershell
python -m pytest simulated_eval/test_benchmark.py -q
```
