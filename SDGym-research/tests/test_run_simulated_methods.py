import json

import numpy as np
import pandas as pd
import pytest

from synthetic_data_benchmark import run_simulated_methods as runner


def test_all_methods_cover_paper_and_labeled_matrix():
    methods = runner.requested_methods(["all"])
    assert len(methods) == 29
    assert {"identity", "ctgan", "tvae", "ctgan-full", "tvae-full",
            "ctgan-xgb", "ctgan-full-rf", "tvae-gaussian",
            "tvae-full-dnn"} <= set(methods)
    assert runner.requested_methods(["tvae-nb", "ctgan-rf"]) == [
        "tvae-nb", "ctgan-rf"]
    with pytest.raises(ValueError, match="Unknown methods"):
        runner.requested_methods(["made-up"])


def test_cpu_device_is_always_available():
    assert runner.resolve_device("cpu", 0) == "cpu"


def test_generator_sample_is_saved_and_reused(tmp_path, monkeypatch):
    train = pd.DataFrame({"feature_0": [0.0, 1.0],
                          "feature_1": [1.0, 2.0]})
    calls = []

    def generate_once(method, frame, dataset, seed, rows, epochs, device):
        calls.append((method, dataset, seed, rows, epochs, device))
        return pd.concat([frame] * 2, ignore_index=True)

    monkeypatch.setattr(runner, "generate", generate_once)
    arguments = (tmp_path, "ctgan_xonly", train, "grid", 42, 4, 1, "cpu")
    first, path = runner.sample_and_save(*arguments)
    second, same_path = runner.sample_and_save(*arguments)
    pd.testing.assert_frame_equal(first, second)
    assert path == same_path
    assert len(calls) == 1
    with pytest.raises(ValueError, match="settings differ"):
        runner.sample_and_save(tmp_path, "ctgan_xonly", train, "grid", 42,
                               5, 2, "cpu")


def test_mixture_extension_labels_a_saved_feature_table():
    features = pd.DataFrame({"feature_0": [0.0, 1.0, -1.0],
                             "feature_1": [0.9, 2.4, -1.0]})
    result = runner.labeled(features, "grid")
    assert result["label"].tolist() == [1, 1, 0]
    pd.testing.assert_frame_equal(features, result.drop(columns="label"))


def test_simulated_dnn_keeps_dev_selected_checkpoint_without_quality_gate(tmp_path):
    rng = np.random.default_rng(7)
    train = pd.DataFrame(rng.normal(size=(32, 2)), columns=["feature_0", "feature_1"])
    dev = pd.DataFrame(rng.normal(size=(16, 2)), columns=train.columns)
    synthetic = pd.DataFrame(rng.normal(size=(8, 2)), columns=train.columns)
    train_labels = pd.Series(rng.integers(0, 2, size=len(train)))
    dev_labels = pd.Series(rng.integers(0, 2, size=len(dev)))
    report_path = tmp_path / "dnn.json"

    labels = runner.predict_labels(
        "dnn", train, train_labels, dev, dev_labels, synthetic,
        "grid", 7, report_path, "cpu")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(labels) == len(synthetic)
    assert report["selected_epoch"] >= 1
    assert report["quality_gate_enforced"] is False


def test_run_one_skips_completed_labeler_fit(tmp_path, monkeypatch):
    folder = tmp_path / "seed_42" / "grid"
    folder.mkdir(parents=True)
    (folder / "ctgan-rf_labeled.json").write_text("{}", encoding="utf-8")
    features = pd.DataFrame({"feature_0": [0.0], "feature_1": [1.0]})
    source = folder / "ctgan_paper_sample.csv"
    source.write_text("feature_0,feature_1\n0,1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "verified_split", lambda *args: (
        folder, {}, features, features))
    monkeypatch.setattr(runner, "sample_and_save", lambda *args: (features, source))
    monkeypatch.setattr(runner, "predict_labels", lambda *args: pytest.fail(
        "completed labeler was fitted again"))
    seen = []
    monkeypatch.setattr(runner, "score_labeled", lambda *args: seen.append(args[5]))

    runner.run_one(tmp_path, "grid", 42, ["ctgan-rf"], 1, 1, 1, 1, "cpu")

    assert seen == [None]


def test_saved_generator_rejects_changed_epochs_or_training(tmp_path, monkeypatch):
    train = pd.DataFrame({"feature_0": [0., 1.], "feature_1": [1., 2.]})
    monkeypatch.setattr(runner, "generate", lambda *args: train.copy())
    runner.sample_and_save(tmp_path, "ctgan_paper", train, "grid", 42, 2, 1, "cpu")
    with pytest.raises(ValueError, match="settings differ"):
        runner.sample_and_save(tmp_path, "ctgan_paper", train, "grid", 42, 2, 300, "cpu")
    with pytest.raises(ValueError, match="settings differ"):
        runner.sample_and_save(tmp_path, "ctgan_paper", train + 1, "grid", 42, 2, 1, "cpu")


def test_saved_generator_rejects_changed_sample(tmp_path, monkeypatch):
    train = pd.DataFrame({"feature_0": [0., 1.], "feature_1": [1., 2.]})
    monkeypatch.setattr(runner, "generate", lambda *args: train.copy())
    _, path = runner.sample_and_save(tmp_path, "ctgan_paper", train, "grid", 42, 2, 1, "cpu")
    (train + 3).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sample changed"):
        runner.sample_and_save(tmp_path, "ctgan_paper", train, "grid", 42, 2, 1, "cpu")


def test_rare_binary_feature_survives_categorical_nb_preprocessing():
    from synthesize_data.naive_bayes import _fit_quantile_bins

    train = pd.DataFrame({"rare": [0] * 95 + [1] * 5})
    train_codes, test_codes = _fit_quantile_bins(train, pd.DataFrame({"rare": [0, 1]}))
    assert train_codes[:, 0].tolist() == train["rare"].tolist()
    assert test_codes[:, 0].tolist() == [0, 1]


def test_rf_labeler_is_independent_of_global_random_state(tmp_path):
    rng = np.random.default_rng(18)
    train = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    labels = pd.Series(rng.integers(0, 2, size=100))
    synthetic = pd.DataFrame(rng.normal(size=(100, 2)), columns=train.columns)
    arguments = ("rf", train, labels, train, labels, synthetic, "grid", 7,
                 tmp_path / "labeler.json", "cpu")
    np.random.seed(1)
    first = runner.predict_labels(*arguments)
    np.random.seed(999)
    second = runner.predict_labels(*arguments)
    np.testing.assert_array_equal(first, second)


def test_summary_rejects_result_identity_mismatch(tmp_path):
    folder = tmp_path / "seed_42" / "grid"
    folder.mkdir(parents=True)
    (folder / "identity_result.json").write_text(json.dumps({
        "dataset": "ring", "seed": 42, "method": "identity", "l_syn": -2., "l_test": -3.
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runner.summarize_runs(tmp_path, ["grid"], [42], ["identity"])


def test_cached_likelihood_rejects_changed_oracle_provenance(tmp_path):
    folder = runner.prepare("grid", 42, 100, 100, tmp_path)
    result = runner.checked_score(folder, tmp_path, "grid", 42, "identity",
                                  folder / "train.csv", 100, 1, "cpu")
    assert runner.checked_score(folder, tmp_path, "grid", 42, "identity",
                                folder / "train.csv", 100, 1, "cpu") == result
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["oracle_sha256"] = "changed-oracle"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs differ"):
        runner.checked_score(folder, tmp_path, "grid", 42, "identity",
                             folder / "train.csv", 100, 1, "cpu")


def test_labeled_result_reuses_only_matching_oracle(tmp_path):
    folder = runner.prepare("grid", 42, 100, 100, tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    train = runner.read_table(folder / "train.csv", "grid")
    test = runner.labeled(runner.read_table(folder / "test.csv", "grid"), "grid")
    args = (folder, tmp_path, "grid", 42, "ctgan-rf", runner.labeled(train, "grid"),
            test, folder / "train.csv", manifest, 100, 1, "cpu")
    first = runner.score_labeled(*args)
    assert runner.score_labeled(*args) == first
    path = folder / "ctgan-rf_labeled.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["oracle_sha256"] = "different-oracle"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs differ"):
        runner.score_labeled(*args)
