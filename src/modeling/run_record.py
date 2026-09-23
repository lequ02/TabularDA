"""Write one reproducible record for a corrected classifier run."""

import json
import hashlib
import subprocess
from importlib.metadata import version
from pathlib import Path


def write_run_record(path, *, dataset, seed, train_option, augment_option,
                     synthetic_path, synthetic_label_counts, split_manifest_path,
                     classifier, batch_size, learning_rate, epoch_budget,
                     selected_epoch, selection_metric, test_loss, test_scores,
                     predictions_path, weight_path):
    root = Path(__file__).resolve().parents[2]
    code_version = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    code_hash = hashlib.sha256()
    for source in sorted((root / "src").rglob("*.py")):
        code_hash.update(str(source.relative_to(root)).encode("utf-8"))
        code_hash.update(source.read_bytes())
    with open(split_manifest_path, encoding="utf-8") as split_file:
        split_manifest = json.load(split_file)
    generator_provenance_path = None
    generator_provenance = None
    synthetic_quality_path = None
    synthetic_quality = None
    dnn_dev_report_path = None
    dnn_dev_report = None
    generator_model_path = None
    predictor_model_path = None
    prepared_root = root / "data" / "corrected_v2" / dataset / f"seed_{seed}"
    prepared_real_paths = {
        split: {
            "raw": str(prepared_root / f"{dataset}_{split}.csv"),
            "onehot": str(prepared_root / f"onehot_{dataset}_{split}.csv"),
        }
        for split in ("train", "dev", "test")
    }
    for split_paths in prepared_real_paths.values():
        for prepared_path in split_paths.values():
            if not Path(prepared_path).is_file():
                raise FileNotFoundError(f"Missing prepared real split: {prepared_path}")
    if not Path(weight_path).is_file():
        raise FileNotFoundError(f"Missing downstream model weights: {weight_path}")
    if synthetic_path is not None:
        if not Path(synthetic_path).is_file():
            raise FileNotFoundError(f"Missing synthetic table: {synthetic_path}")
        synthetic_quality_path = Path(synthetic_path).with_suffix(".quality.json")
        with synthetic_quality_path.open(encoding="utf-8") as quality_file:
            synthetic_quality = json.load(quality_file)
        model_root = root / "sdv trained model" / "corrected_v2" / dataset / f"seed_{seed}"
        if augment_option == "ctgan" or augment_option.startswith("compare_"):
            model_name = f"{dataset}_synthesizer"
        elif augment_option == "tvae" or augment_option.startswith("tvae_compare_"):
            model_name = f"{dataset}_TVAE_synthesizer"
        elif augment_option.startswith("tvae_"):
            model_name = f"{dataset}_tvae_synthesizer_onlyX"
        else:
            model_name = f"{dataset}_synthesizer_onlyX"
        generator_model_path = model_root / f"{model_name}.pkl"
        if not generator_model_path.is_file():
            raise FileNotFoundError(f"Missing generator model: {generator_model_path}")
        generator_provenance_path = model_root / f"{model_name}.provenance.json"
        with generator_provenance_path.open(encoding="utf-8") as provenance_file:
            generator_provenance = json.load(provenance_file)
        if augment_option not in {"ctgan", "tvae"}:
            extension = ".predictor.pt" if augment_option.endswith("dnn") else ".predictor.pkl"
            predictor_model_path = Path(synthetic_path).with_suffix(extension)
            if not predictor_model_path.is_file():
                raise FileNotFoundError(f"Missing target predictor model: {predictor_model_path}")
        if augment_option.endswith("dnn"):
            dnn_dev_report_path = Path(synthetic_path).with_suffix(".dnn.json")
            with dnn_dev_report_path.open(encoding="utf-8") as report_file:
                dnn_dev_report = json.load(report_file)
    record = {
        "code_version": code_version,
        "source_sha256": code_hash.hexdigest(),
        "package_versions": {
            package: version(package)
            for package in ("sdv", "ctgan", "rdt", "torch", "pandas", "numpy", "scikit-learn")
        },
        "dataset": dataset,
        "seed": seed,
        "train_option": train_option,
        "augment_option": augment_option,
        "synthetic_path": synthetic_path,
        "synthetic_label_counts": synthetic_label_counts,
        "synthetic_quality_path": str(synthetic_quality_path) if synthetic_quality_path else None,
        "synthetic_quality": synthetic_quality,
        "prepared_real_paths": prepared_real_paths,
        "generator_model_path": str(generator_model_path) if generator_model_path else None,
        "generator_provenance_path": str(generator_provenance_path) if generator_provenance_path else None,
        "generator_provenance": generator_provenance,
        "predictor_model_path": str(predictor_model_path) if predictor_model_path else None,
        "dnn_dev_report_path": str(dnn_dev_report_path) if dnn_dev_report_path else None,
        "dnn_dev_report": dnn_dev_report,
        "split_manifest_path": split_manifest_path,
        "split_manifest": split_manifest,
        "classifier": classifier,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epoch_budget": epoch_budget,
        "selected_dev_epoch": selected_epoch,
        "selection_metric": selection_metric,
        "test_loss": test_loss,
        "test_scores": test_scores,
        "downstream_weight_path": str(weight_path),
        "predictions_path": predictions_path,
    }
    with open(path, "w", encoding="utf-8") as record_file:
        json.dump(record, record_file, indent=2)
