"""Independent numeric checks for the simulated benchmark (no generator training)."""
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SDGym-research"))
from synthetic_data_benchmark import simulated_benchmark as bench
from synthetic_data_benchmark import run_simulated_methods as runner


def main():
    logging.getLogger("pgmpy").setLevel(logging.ERROR)
    report = {"mixtures": {}, "bayesian_networks": {}, "cache_reproduction": {}}
    for name in bench.MIXTURES:
        oracle = bench.mixture_oracle(name, 42)
        data = bench.sample_mixture(oracle, 1000, 43)
        probability = sum(w * multivariate_normal.pdf(data, mean=mean,
                                                     cov=np.eye(2) * oracle["variance"])
                          for w, mean in zip(oracle["weights"], oracle["means"]))
        error = float(np.max(np.abs(bench.mixture_log_prob(data, oracle) - np.log(probability))))
        assert error < 1e-12
        report["mixtures"][name] = {"independent_logpdf_max_error": error}
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.estimators import MaximumLikelihoodEstimator
    for name in ("asia", "alarm", "child", "insurance"):
        model = bench.bn_model(name)
        data = BayesianModelSampling(model).forward_sample(size=500, seed=43, show_progress=False).astype(str)
        probabilities = []
        for _, row in data.iterrows():
            probabilities.append(np.prod([cpd.get_value(**{v: row[v] for v in cpd.variables})
                                          for cpd in model.get_cpds()]))
        error = float(np.max(np.abs(bench.bn_log_prob(data, model) - np.log(np.array(probabilities)+1e-8))))
        assert error < 1e-12
        refitted = bench.bn_model(name)
        states = {v: model.get_cpds(v).state_names[v] for v in model.nodes()}
        refitted.fit(data, estimator=MaximumLikelihoodEstimator, state_names=states)
        cpd_error = 0.0
        for cpd in refitted.get_cpds():
            values = data[list(cpd.variables)].value_counts()
            for row in data[list(cpd.variables)].drop_duplicates().itertuples(index=False, name=None):
                denominator = sum(count for key, count in values.items()
                                  if tuple(np.atleast_1d(key))[1:] == row[1:])
                expected = values[row if len(row)>1 else row[0]] / denominator
                actual = cpd.get_value(**dict(zip(cpd.variables, row)))
                cpd_error = max(cpd_error, abs(actual-expected))
        assert cpd_error < 1e-12
        report["bayesian_networks"][name] = {"independent_joint_max_error": error,
                                              "empirical_cpd_max_error": cpd_error,
                                              "identity_l_syn_500_rows": float(bench.bn_log_prob(data, model).mean())}
    train = pd.DataFrame({"feature_0": [0., 1.], "feature_1": [1., 2.]})
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        with patch.object(runner, "generate", return_value=train.copy()) as generate:
            runner.sample_and_save(folder, "ctgan_paper", train, "grid", 42, 2, 1, "cpu")
            try:
                runner.sample_and_save(folder, "ctgan_paper", train, "grid", 42, 2, 300, "cpu")
                report["cache_reproduction"]["different_epoch_sample_reused"] = generate.call_count == 1
            except ValueError:
                report["cache_reproduction"]["different_epoch_sample_reused"] = False
    from synthesize_data.naive_bayes import _fit_quantile_bins
    binary = pd.DataFrame({"rare": [0] * 95 + [1] * 5})
    codes, _ = _fit_quantile_bins(binary, binary)
    report["categorical_nb"] = {"rare_binary_input_unique": [0, 1],
                                 "encoded_unique": np.unique(codes).tolist(),
                                 "rare_indicator_collapsed": len(np.unique(codes)) == 1}
    destination = ROOT / "audit" / "simulated_benchmark_numeric_checks.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
