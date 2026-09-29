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
    with pytest.raises(ValueError, match="row count"):
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
