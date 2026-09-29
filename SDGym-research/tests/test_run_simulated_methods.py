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
