import numpy as np
import pandas as pd
import pytest

from synthetic_data_benchmark.simulated_benchmark import (
    bn_log_prob,
    bn_model,
    evaluate,
    mixture_log_prob,
    mixture_oracle,
    prepare,
    sample_mixture,
)


def test_mixture_oracle_and_mode_collapse(tmp_path):
    oracle = mixture_oracle("grid", seed=42)
    assert len(oracle["means"]) == 25
    assert sum(oracle["weights"]) == pytest.approx(1)
    assert len(mixture_oracle("ring", seed=42)["means"]) == 8
    assert mixture_oracle("gridr", seed=42) == mixture_oracle("gridr", seed=42)

    prepare("grid", 42, 500, 500, tmp_path)
    identity = evaluate("grid", 42, "identity", tmp_path, None, 500, 1, "cpu")
    assert np.isfinite(identity["l_syn"])
    assert np.isfinite(identity["l_test"])

    collapsed = pd.DataFrame(
        np.random.default_rng(3).normal(scale=0.05, size=(500, 2)),
        columns=["feature_0", "feature_1"],
    )
    path = tmp_path / "collapsed.csv"
    collapsed.to_csv(path, index=False)
    collapsed_result = evaluate("grid", 42, "collapsed", tmp_path, path, 500, 1, "cpu")
    assert collapsed_result["l_test"] < identity["l_test"]


def test_oracle_scores_the_generating_distribution():
    oracle = mixture_oracle("ring", 7)
    real = sample_mixture(oracle, 800, 8)
    far_away = real + 50
    assert mixture_log_prob(real, oracle).mean() > mixture_log_prob(far_away, oracle).mean()


def test_bayesian_oracle_and_fixed_structure(tmp_path):
    prepare("asia", 42, 100, 100, tmp_path)
    result = evaluate("asia", 42, "identity", tmp_path, None, 100, 1, "cpu")
    assert np.isfinite(result["l_syn"])
    assert np.isfinite(result["l_test"])
    train = pd.read_csv(tmp_path / "seed_42" / "asia" / "train.csv", dtype=str)
    assert np.isfinite(bn_log_prob(train, bn_model("asia"))).all()


def test_rejects_changed_prepared_split(tmp_path):
    prepare("grid", 42, 100, 100, tmp_path)
    path = tmp_path / "seed_42" / "grid" / "test.csv"
    path.write_text(path.read_text(encoding="utf-8") + "0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test table changed"):
        evaluate("grid", 42, "identity", tmp_path, None, 100, 1, "cpu")
