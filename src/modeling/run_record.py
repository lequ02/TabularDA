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
                     predictions_path):
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
    if synthetic_path is not None:
        synthetic_quality_path = Path(synthetic_path).with_suffix(".quality.json")
        with synthetic_quality_path.open(encoding="utf-8") as quality_file:
            synthetic_quality = json.load(quality_file)
        model_root = root / "sdv trained model" / "corrected_v2" / dataset / f"seed_{seed}"
        if augment_option == "ctgan":
            model_name = f"{dataset}_synthesizer"
        elif augment_option == "tvae":
            model_name = f"{dataset}_TVAE_synthesizer"
        elif augment_option.startswith("tvae_"):
            model_name = f"{dataset}_tvae_synthesizer_onlyX"
        else:
            model_name = f"{dataset}_synthesizer_onlyX"
        generator_provenance_path = model_root / f"{model_name}.provenance.json"
        with generator_provenance_path.open(encoding="utf-8") as provenance_file:
            generator_provenance = json.load(provenance_file)
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
        "generator_provenance_path": str(generator_provenance_path) if generator_provenance_path else None,
        "generator_provenance": generator_provenance,
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
        "predictions_path": predictions_path,
    }
    with open(path, "w", encoding="utf-8") as record_file:
        json.dump(record, record_file, indent=2)
