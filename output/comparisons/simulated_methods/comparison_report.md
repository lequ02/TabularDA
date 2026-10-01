# Comprehensive simulated-method comparison

Published references: [CTGAN supplement, Table 3](https://papers.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Supplemental.zip).

All 58 pasted rows are preserved. Deltas compare each variant with its generator's published baseline; positive means higher likelihood. These are descriptive differences, not statistical significance or an exact reproduction claim.

## How to read method names

`ctgan-rf` generates X-only features and predicts their targets with RF; `ctgan-full-rf` generates the table with the target present, then replaces its target with RF predictions. The same convention applies to TVAE, XGB, DNN, both naive Bayes methods and PCA-GMM. `*-full` retains the generated target. Every labeled variant is evaluated by a fresh random forest trained on its synthetic table.

GM likelihood ignores labels; repeated scores across labelers sharing features are expected. BN likelihood includes the target, so relabeling changes the joint distribution. Oracle consistency is defined only for GM.

## How the metrics are calculated

X means the feature columns; CTGAN itself generates X. For GM, CTGAN trained on the two original continuous columns generates two-column rows. Joint generation trains a separate CTGAN on those columns plus our added discrete target, then generates all three columns. For BN, the target already exists in the original table, so the baseline and generated-target variants reuse the same full-table sample.

L_syn is the mean log probability/density of synthetic rows under the known original oracle. L_test fits a density/probability model to the synthetic rows, then averages its log probability/density on the independent original test rows. GM refitting uses a diagonal-covariance Gaussian mixture with the original number of components; BN refitting keeps the original graph and estimates conditional probability tables. BN scoring uses log(p + 1e-8). GM scores features only; BN scores the complete table.

For prediction, GM receives an added target: label = 1 if feature_1 > 1.5 * feature_0 + 0.8, else 0. BN targets are dysp (Asia), BP (Alarm), Disease (Child), and Accident (Insurance). A fresh 100-tree random forest trains on synthetic features and targets and predicts targets on the independent original test set. Accuracy is the fraction correct. Macro F1 is the unweighted average of per-class F1, with zero_division=0. Labeler models learn using original training labels; they are distinct from the final evaluation RF. These prediction tasks are our extension, not the paper's simulated benchmark.

The GM and BN combined figures show likelihood, accuracy and macro F1 on aligned method rows. Not evaluated means that no prediction result was saved for that original paper-baseline row. All family figures average datasets equally and average the two seeds, rather than pooling their classification predictions.

The pasted summary alone does not identify coverage or variability. The subsequent server audit verified all seven datasets, seeds 42 and 43, 10,000 train/test/synthetic rows, and recorded settings of 300 epochs on CUDA. The full raw results were saved separately in audit/remote_simulated_methods_per_run.csv. The paper's real-data F1 values do not supply a baseline for the added simulated prediction tasks.

## Supplement-only references

All chart references and deltas below use unweighted dataset averages calculated from the rounded numbers in Supplement Table 3. Main-paper averages are not used. Our observations are the audited pre-fix run, not a new post-fix training run.

CTGAN mapping: the supplement labels its BN row 'TGAN' and duplicates 'TVAE' in GM; the second GM row is inferred as CTGAN.

Main Table 2 reports TVAE BN L_syn=-6.76 and L_test=-9.59. Supplement means are -10.1275 and -9.8675, close to ours (-10.149895, -9.874837). This is an unresolved internal publication inconsistency.

| Family | Model | Supplement mean L_syn | Supplement mean L_test |
|---|---|---:|---:|
| GM | identity | -2.606667 | -2.610000 |
| GM | ctgan | -5.723333 | -3.396667 |
| GM | tvae | -2.650000 | -5.416667 |
| BN | identity | -9.332500 | -9.360000 |
| BN | ctgan | -11.665000 | -10.602500 |
| BN | tvae | -10.127500 | -9.867500 |

## GM: all supplied methods

| Method | L_syn | L_test | Δ L_syn vs supplement | Δ L_test vs supplement | Accuracy % | Macro F1 % | Oracle consistency % |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity (paper) | -2.613825 | -2.620910 | -0.007158 | -0.010910 | — | — | — |
| ctgan (paper) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | — | — | — |
| tvae (paper) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | — | — | — |
| ctgan-full (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 89.132 | 88.923 | 87.967 |
| ctgan-full-categorical (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 95.578 | 95.480 | 93.568 |
| ctgan-full-gaussian (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 92.848 | 92.683 | 92.610 |
| ctgan-full-pca_gmm (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 98.875 | 98.853 | 98.765 |
| ctgan-full-rf (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 99.457 | 99.433 | 99.447 |
| ctgan-full-xgb (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 99.437 | 99.415 | 99.273 |
| ctgan-full-dnn (labeled_extension) | -3.543163 | -3.162643 | +2.180170 | +0.234024 | 99.630 | 99.617 | 99.912 |
| ctgan-categorical (labeled_extension) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | 95.567 | 95.466 | 92.392 |
| ctgan-gaussian (labeled_extension) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | 92.807 | 92.635 | 91.963 |
| ctgan-pca_gmm (labeled_extension) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | 98.845 | 98.821 | 98.628 |
| ctgan-rf (labeled_extension) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | 99.558 | 99.537 | 99.230 |
| ctgan-xgb (labeled_extension) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | 99.492 | 99.469 | 98.942 |
| ctgan-dnn (labeled_extension) | -3.520473 | -3.027658 | +2.202860 | +0.369009 | 99.683 | 99.672 | 99.870 |
| tvae-full (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 96.397 | 96.280 | 92.483 |
| tvae-full-categorical (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 95.222 | 95.113 | 94.137 |
| tvae-full-gaussian (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 92.797 | 92.625 | 91.312 |
| tvae-full-pca_gmm (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 98.858 | 98.834 | 98.577 |
| tvae-full-rf (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 99.597 | 99.578 | 99.578 |
| tvae-full-xgb (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 99.207 | 99.180 | 99.508 |
| tvae-full-dnn (labeled_extension) | -2.694971 | -3.322731 | -0.044971 | +2.093936 | 99.180 | 99.157 | 99.925 |
| tvae-categorical (labeled_extension) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | 95.517 | 95.416 | 93.942 |
| tvae-gaussian (labeled_extension) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | 92.840 | 92.670 | 91.330 |
| tvae-pca_gmm (labeled_extension) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | 98.772 | 98.748 | 98.718 |
| tvae-rf (labeled_extension) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | 99.443 | 99.422 | 99.518 |
| tvae-xgb (labeled_extension) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | 99.420 | 99.399 | 99.403 |
| tvae-dnn (labeled_extension) | -2.706210 | -3.172882 | -0.056210 | +2.243785 | 99.522 | 99.508 | 99.912 |

## BN: all supplied methods

| Method | L_syn | L_test | Δ L_syn vs supplement | Δ L_test vs supplement | Accuracy % | Macro F1 % | Oracle consistency % |
|---|---:|---:|---:|---:|---:|---:|---:|
| identity (paper) | -9.344308 | -9.383232 | -0.011808 | -0.023232 | — | — | — |
| ctgan (paper) | -12.422124 | -10.780463 | -0.757124 | -0.177963 | — | — | — |
| tvae (paper) | -10.149895 | -9.874837 | -0.022395 | -0.007337 | — | — | — |
| ctgan-full (labeled_extension) | -12.422124 | -10.780463 | -0.757124 | -0.177963 | 76.252 | 66.952 | — |
| ctgan-full-categorical (labeled_extension) | -12.162500 | -11.060589 | -0.497500 | -0.458089 | 81.392 | 75.480 | — |
| ctgan-full-gaussian (labeled_extension) | -12.213111 | -10.800768 | -0.548111 | -0.198268 | 79.280 | 73.160 | — |
| ctgan-full-pca_gmm (labeled_extension) | -12.150418 | -11.046069 | -0.485418 | -0.443569 | 82.034 | 76.786 | — |
| ctgan-full-rf (labeled_extension) | -12.074468 | -11.175628 | -0.409468 | -0.573128 | 87.112 | 81.979 | — |
| ctgan-full-xgb (labeled_extension) | -12.073426 | -11.174826 | -0.408426 | -0.572326 | 87.328 | 82.249 | — |
| ctgan-full-dnn (labeled_extension) | -12.072335 | -11.037839 | -0.407335 | -0.435339 | 87.056 | 82.647 | — |
| ctgan-categorical (labeled_extension) | -12.404039 | -11.088612 | -0.739039 | -0.486112 | 81.204 | 75.310 | — |
| ctgan-gaussian (labeled_extension) | -12.488471 | -10.844730 | -0.823471 | -0.242230 | 78.949 | 72.659 | — |
| ctgan-pca_gmm (labeled_extension) | -12.389757 | -11.067559 | -0.724757 | -0.465059 | 82.005 | 76.853 | — |
| ctgan-rf (labeled_extension) | -12.333721 | -11.135644 | -0.668721 | -0.533144 | 87.150 | 82.176 | — |
| ctgan-xgb (labeled_extension) | -12.330718 | -11.197879 | -0.665718 | -0.595379 | 87.245 | 82.202 | — |
| ctgan-dnn (labeled_extension) | -12.328298 | -11.047831 | -0.663298 | -0.445331 | 86.983 | 82.581 | — |
| tvae-full (labeled_extension) | -10.149895 | -9.874837 | -0.022395 | -0.007337 | 83.301 | 78.303 | — |
| tvae-full-categorical (labeled_extension) | -10.046412 | -10.338337 | +0.081088 | -0.470837 | 81.331 | 75.426 | — |
| tvae-full-gaussian (labeled_extension) | -10.112868 | -10.024438 | +0.014632 | -0.156938 | 79.168 | 72.846 | — |
| tvae-full-pca_gmm (labeled_extension) | -10.021307 | -10.258314 | +0.106193 | -0.390814 | 82.181 | 76.955 | — |
| tvae-full-rf (labeled_extension) | -9.887812 | -10.351578 | +0.239688 | -0.484078 | 86.961 | 82.030 | — |
| tvae-full-xgb (labeled_extension) | -9.879182 | -10.380180 | +0.248318 | -0.512680 | 87.312 | 82.336 | — |
| tvae-full-dnn (labeled_extension) | -9.877330 | -10.300312 | +0.250170 | -0.432812 | 87.026 | 82.591 | — |
| tvae-categorical (labeled_extension) | -10.008595 | -10.428074 | +0.118905 | -0.560574 | 81.405 | 75.595 | — |
| tvae-gaussian (labeled_extension) | -10.064878 | -10.093040 | +0.062622 | -0.225540 | 79.085 | 72.844 | — |
| tvae-pca_gmm (labeled_extension) | -9.986534 | -10.351237 | +0.140966 | -0.483737 | 82.230 | 77.015 | — |
| tvae-rf (labeled_extension) | -9.848546 | -10.427399 | +0.278954 | -0.559899 | 86.950 | 82.084 | — |
| tvae-xgb (labeled_extension) | -9.840198 | -10.449317 | +0.287302 | -0.581817 | 87.279 | 82.297 | — |
| tvae-dnn (labeled_extension) | -9.837159 | -10.389011 | +0.290341 | -0.521511 | 86.986 | 82.561 | — |
