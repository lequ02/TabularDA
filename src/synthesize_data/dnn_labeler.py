"""Small real-data MLP used to assign targets to synthetic feature rows."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def fit_predict_dnn(
    x_train,
    y_train,
    x_dev,
    y_dev,
    x_synthetic,
    *,
    target_name,
    is_classification,
    seed,
    report_path,
    dataset_name,
):
    """Fit on real train rows, select on real dev rows, and label synthetic X."""
    if not x_train.columns.equals(x_dev.columns) or not x_train.columns.equals(x_synthetic.columns):
        raise ValueError("Train, dev, and synthetic feature columns must match in order")

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = _numeric_array(x_train, "train")
    dev_x = _numeric_array(x_dev, "dev")
    synthetic_x = _numeric_array(x_synthetic, "synthetic")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    train_x = (train_x - mean) / scale
    dev_x = (dev_x - mean) / scale
    synthetic_x = (synthetic_x - mean) / scale
    if not all(np.isfinite(a).all() for a in (train_x, dev_x, synthetic_x)):
        raise ValueError("Standardized DNN features contain non-finite values")

    y_train = np.asarray(y_train).reshape(-1)
    y_dev = np.asarray(y_dev).reshape(-1)
    if len(train_x) != len(y_train) or len(dev_x) != len(y_dev):
        raise ValueError("Feature and target row counts must match")
    if len(train_x) == 0 or len(dev_x) == 0:
        raise ValueError("DNN labeling requires nonempty real train and dev splits")

    if is_classification:
        encoder = LabelEncoder().fit(y_train)
        if not np.isin(y_dev, encoder.classes_).all():
            raise ValueError("Dev targets contain a class absent from real training data")
        train_y = encoder.transform(y_train).astype(np.int64)
        dev_y = encoder.transform(y_dev).astype(np.int64)
        class_count = len(encoder.classes_)
        if class_count < 2:
            raise ValueError("Classification DNN requires at least two training classes")
        counts = np.bincount(train_y, minlength=class_count)
        class_weights = (len(train_y) / (class_count * np.maximum(counts, 1))) ** 0.5
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        output_count = class_count
    else:
        train_y = y_train.astype(np.float32)
        dev_y = y_dev.astype(np.float32)
        if not np.isfinite(train_y).all() or not np.isfinite(dev_y).all():
            raise ValueError("Regression targets contain non-finite values")
        target_mean = float(train_y.mean())
        target_scale = float(train_y.std())
        if not np.isfinite(target_scale) or target_scale == 0:
            target_scale = 1.0
        train_y = (train_y - target_mean) / target_scale
        dev_y = (dev_y - target_mean) / target_scale
        output_count = 1
        criterion = nn.MSELoss()

    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, output_count),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.long if is_classification else torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, generator=generator)
    dev_tensor = torch.as_tensor(dev_x, dtype=torch.float32, device=device)
    dev_target = torch.as_tensor(
        dev_y, dtype=torch.long if is_classification else torch.float32, device=device
    )

    max_epochs, patience = 500, 30
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    dev_loss_history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x)
            if not is_classification:
                predictions = predictions.squeeze(-1)
            loss = criterion(predictions, batch_y)
            if not torch.isfinite(loss):
                raise ValueError(f"DNN training loss became non-finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_output = model(dev_tensor)
            if not is_classification:
                dev_output = dev_output.squeeze(-1)
            dev_loss = float(criterion(dev_output, dev_target).item())
        if not np.isfinite(dev_loss):
            raise ValueError(f"DNN dev loss became non-finite at epoch {epoch}")
        dev_loss_history.append(dev_loss)
        if best_state is None or dev_loss < best_loss - max(abs(best_loss) * 0.001, 1e-8):
            best_loss = dev_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                break

    if best_state is None:
        raise ValueError("DNN did not produce a finite dev-selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dev_output = model(dev_tensor).cpu().numpy()
        synthetic_batches = []
        for start in range(0, len(synthetic_x), 4096):
            batch = torch.as_tensor(synthetic_x[start:start + 4096], dtype=torch.float32, device=device)
            synthetic_batches.append(model(batch).cpu().numpy())
        synth_output = np.concatenate(synthetic_batches, axis=0)

    if is_classification:
        dev_pred_ids = dev_output.argmax(axis=1)
        dev_pred = encoder.inverse_transform(dev_pred_ids)
        synthetic_pred = encoder.inverse_transform(synth_output.argmax(axis=1))
        accuracy = float(accuracy_score(y_dev, dev_pred))
        macro_f1 = float(f1_score(y_dev, dev_pred, average="macro", zero_division=0))
        majority = pd.Series(y_train).value_counts().idxmax()
        majority_pred = np.full(len(y_dev), majority)
        majority_accuracy = float(accuracy_score(y_dev, majority_pred))
        majority_macro_f1 = float(f1_score(y_dev, majority_pred, average="macro", zero_division=0))
        report = {
            "dataset": dataset_name, "task": "classification", "seed": seed,
            "selected_epoch": best_epoch, "epochs_run": epoch, "dev_loss": best_loss,
            "dev_loss_history": dev_loss_history,
            "dev_accuracy": accuracy, "dev_macro_f1": macro_f1,
            "majority_dev_accuracy": majority_accuracy,
            "majority_dev_macro_f1": majority_macro_f1,
            "converged": stopped_early,
        }
        if str(dataset_name).lower() == "credit":
            # In the credit dataset, the rarest observed class is the fraud class.
            fraud_class = encoder.classes_[int(np.argmin(counts))]
            fraud_id = int(encoder.transform([fraud_class])[0])
            fraud_mask = y_dev == fraud_class
            if not fraud_mask.any():
                raise ValueError("Credit dev split contains no fraud rows; cannot quality-gate DNN")
            fraud_recall = float(np.mean(dev_pred[fraud_mask] == fraud_class))
            fraud_probability = torch.softmax(torch.as_tensor(dev_output), dim=1).numpy()[:, fraud_id]
            prevalence = float(np.mean(y_dev == fraud_class))
            pr_auc = float(average_precision_score((y_dev == fraud_class).astype(int), fraud_probability))
            report.update({"fraud_class": str(fraud_class), "fraud_recall": fraud_recall,
                           "fraud_prevalence": prevalence, "fraud_pr_auc": pr_auc})
            quality_passed = macro_f1 > majority_macro_f1 and fraud_recall > 0 and pr_auc > prevalence
        else:
            quality_passed = macro_f1 > majority_macro_f1 and accuracy > majority_accuracy
    else:
        synthetic_pred = synth_output.reshape(-1) * target_scale + target_mean
        dev_pred = dev_output.reshape(-1) * target_scale + target_mean
        r2 = float(r2_score(y_dev, dev_pred))
        baseline_r2 = float(r2_score(y_dev, np.full(len(y_dev), target_mean)))
        report = {
            "dataset": dataset_name, "task": "regression", "seed": seed,
            "selected_epoch": best_epoch, "epochs_run": epoch, "dev_loss": best_loss,
            "dev_loss_units": "standardized_target_mse",
            "dev_loss_history": dev_loss_history,
            "dev_r2": r2, "mean_baseline_dev_r2": baseline_r2,
            "converged": stopped_early,
        }
        quality_passed = r2 > max(0.0, baseline_r2)
    report["quality_gate_passed"] = bool(quality_passed)
    _write_report(report_path, report)
    if not stopped_early:
        raise ValueError(f"DNN did not converge within the {max_epochs}-epoch budget")
    if not quality_passed:
        raise ValueError(f"DNN failed dev quality gate: {report}")
    target = pd.DataFrame({target_name: synthetic_pred}, index=x_synthetic.index)
    return pd.concat([x_synthetic.copy(), target], axis=1)


def _numeric_array(frame, split_name):
    values = frame.to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"DNN {split_name} features must be a nonempty finite matrix")
    return values


def _write_report(report_path, report):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
