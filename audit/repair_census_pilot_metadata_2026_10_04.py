"""Recover completed real-only pilot metadata from its preserved log; no fitting."""
import json
from pathlib import Path
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path.cwd()
output = ROOT / "output/census_weighted_pilot_20261004"
name = "census_kdd_seed42_real_original"
path = output / "census_kdd/acc" / (name + ".run.json")
record = json.loads(path.read_text())
assert "pilot_protocol" not in record
assert record["selected_dev_epoch"] == 100
assert record["selection_metric"] == "f1_macro"
assert Path(record["downstream_weight_path"]).is_file()
history = pd.read_csv(output / "census_kdd/acc" / (name + ".epochs.csv"))
assert len(history) == 100
assert history.iloc[-1]["dev_f1_macro"] == history["dev_f1_macro"].max()
predictions = pd.read_csv(record["predictions_path"])
assert abs(f1_score(predictions.y_true, predictions.y_pred) - record["test_scores"]["f1_binary"]) < 1e-12
log = (output / "logs/seed42_original.log").read_text()
assert "Finish training!" in log
assert "TypeError: Object of type bool_ is not JSON serializable" in log
dev = json.loads([line.removeprefix("PILOT development ") for line in log.splitlines()
                  if line.startswith("PILOT development ")][-1])
assert abs(dev["scores"]["f1_macro"] - history.iloc[-1]["dev_f1_macro"]) < 1e-12
preflight = json.loads((output / "preflight.json").read_text())
record["pilot_protocol"] = {
    "namespace": "census_weighted_pilot_20261004", "input_namespace": "corrected_v2",
    "started_at_chicago": "2026-10-04T14:46:05.888732-05:00",
    "metadata_recovered_at_chicago": datetime.now(ZoneInfo("America/Chicago")).isoformat(),
    "metadata_recovery_reason": "Training and final evaluation completed; numpy.bool_ serialization stopped metadata finalization",
    "metadata_repair_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "pilot_source_sha256": preflight["pilot_source_sha256"],
    "objective": "BCEWithLogitsLoss", "model_output": "logits",
    "pos_weight_rule": "actual_training_negatives / actual_training_positives",
    "pos_weight": 153031 / 9931, "training_label_counts": {"0": 153031, "1": 9931},
    "shuffle": False, "threshold": .5,
    "test_loss_definition": "weighted BCE using this arm's training pos_weight",
    "selected_development": dev,
    "test_diagnostics": {"rows": len(predictions),
                         "predicted_positive": int(predictions.y_pred.sum()),
                         "positive_rate": float(predictions.y_pred.mean()),
                         "probability_min": float(predictions.score.min()),
                         "probability_max": float(predictions.score.max()),
                         "scores": record["test_scores"]},
    "development_collapse_resolved": bool(0 < dev["predicted_positive"] < dev["rows"]
                                          and dev["scores"]["recall_binary"] > 0)}
path.write_text(json.dumps(record, indent=2))
print(json.dumps({"preserved_run": name, "test_scores": record["test_scores"]}, indent=2))
