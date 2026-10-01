"""Focused checks for exact probabilities, normalized refits, and evaluation data flow."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score

from .benchmark import evaluate
from .oracle import COLUMNS, FEATURES, Oracle
from .bn import BNOracle
from .density import fit_boundary


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

    monkeypatch.setattr("simulated_eval.benchmark.make_dnn", lambda seed, oracle: FixedClassifier())
    metrics = evaluate(sample, test, oracle, seed=42)
    assert metrics["l_syn"] == pytest.approx(oracle.log_prob(sample).mean())
    assert np.isfinite(metrics["l_test"])
    assert metrics["accuracy"] == accuracy_score(test.target, predictions)
    assert metrics["macro_f1"] == f1_score(test.target, predictions, labels=[0, 1], average="macro")
    assert metrics["h_star_agreement"] == accuracy_score(oracle.clean_target(test), predictions)


def test_multimodal_density_and_boundary_refit():
    x = pd.DataFrame([[.2, -.6, 0, 1]], columns=COLUMNS)
    for dataset in ("grid", "ring"):
        oracle = Oracle(dataset=dataset)
        scale = np.sqrt((8 if dataset == "grid" else .5) + oracle.spread**2)
        sd = oracle.spread / scale
        if dataset == "grid":
            expected = np.prod([norm.pdf(x[c].iloc[0], np.arange(-4, 5, 2)/scale, sd).mean()
                                for c in FEATURES[:2]])
        else:
            angles = np.arange(8) * np.pi / 4
            expected = np.mean(norm.pdf(.2, np.cos(angles)/scale, sd) * norm.pdf(-.6, np.sin(angles)/scale, sd))
        assert np.exp(oracle.feature_log_prob(x)[0]) == pytest.approx(expected)
    sample = pd.DataFrame({"feature_0": [1.] * 4, "feature_1": [-1.2, -1.3, -1.7, -1.8],
                           "feature_3": [1] * 4, "target": [1, 1, 1, 0]})
    beta, noise = fit_boundary(sample)
    assert beta == pytest.approx(.75)
    assert noise == pytest.approx(.1)


def test_bn_joint_normalization_refit_and_posterior():
    from itertools import product

    oracle = BNOracle("asia")
    table = pd.DataFrame(product(*[oracle.categories[c] for c in oracle.columns]), columns=oracle.columns)
    probabilities = np.exp(oracle.log_prob(table))
    assert probabilities.sum() == pytest.approx(1)
    assert np.isneginf(oracle.log_prob(table)).any()
    sample = oracle.sample(1000, np.random.default_rng(42))
    assert np.isfinite(oracle.log_prob(sample)).all()
    assert np.exp(oracle.test_log_prob(sample, table, seed=42)).sum() == pytest.approx(1)
    posterior = oracle.posterior(sample)
    np.testing.assert_allclose(posterior.sum(axis=1), 1)
    np.testing.assert_array_equal(oracle.clean_target(sample), posterior.argmax(axis=1))
