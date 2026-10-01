"""Two focused checks: the joint formula and evaluation data flow."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score

from .benchmark import evaluate
from .oracle import COLUMNS, FEATURES, Oracle


def test_joint_density_matches_scalar_probability_and_normalizes_discrete_columns():
    oracle = Oracle()
    x0, x1 = 0.8, -0.6
    total = 0.0
    for c in (0, 1):
        for y in (0, 1):
            row = pd.DataFrame([[x0, x1, c, y]], columns=COLUMNS)
            clean = int(x1 + (1 + oracle.beta * (2*c-1)) * x0 > 0)
            category_p = expit(oracle.kappa * x0 * x1)
            expected = norm.pdf(x0) * norm.pdf(x1)
            expected *= category_p if c else 1-category_p
            expected *= 1-oracle.noise if y == clean else oracle.noise
            actual = np.exp(oracle.log_prob(row)[0])
            assert actual == pytest.approx(expected, rel=1e-12)
            total += actual
    assert total == pytest.approx(norm.pdf(x0)*norm.pdf(x1))


def test_evaluation_uses_training_sample_and_independent_test(monkeypatch):
    oracle = Oracle()
    sample = oracle.sample(1000, np.random.default_rng(1))
    test = oracle.sample(1000, np.random.default_rng(2))
    predictions = (test.feature_0 > 0).to_numpy(dtype=int)

    class FixedClassifier:
        def fit(self, x, y):
            pd.testing.assert_frame_equal(x, sample[FEATURES])
            pd.testing.assert_series_equal(y, sample.target)
            return self

        def predict(self, x):
            pd.testing.assert_frame_equal(x, test[FEATURES])
            return predictions

    monkeypatch.setattr("simulated_eval.benchmark.make_dnn", lambda seed: FixedClassifier())
    metrics = evaluate(sample, test, oracle, seed=42)
    assert metrics["joint_log_likelihood"] == pytest.approx(oracle.log_prob(sample).mean())
    assert metrics["accuracy"] == accuracy_score(test.target, predictions)
    assert metrics["macro_f1"] == f1_score(test.target, predictions, labels=[0, 1], average="macro")
    assert metrics["h_star_agreement"] == accuracy_score(oracle.clean_target(test), predictions)
