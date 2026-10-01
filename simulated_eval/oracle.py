"""Generate and score the exact distribution used by the benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = ["feature_0", "feature_1", "feature_3"]
COLUMNS = FEATURES + ["target"]


@dataclass(frozen=True)
class Oracle:
    kappa: float = 2.0
    beta: float = 0.75
    noise: float = 0.10

    def __post_init__(self):
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        if not 0 < self.beta < 1:
            raise ValueError("beta must be between 0 and 1")
        if not 0 < self.noise < 0.5:
            raise ValueError("noise must be between 0 and 0.5")

    def clean_target(self, features):
        score = features["feature_1"] + (
            1 + self.beta * (2 * features["feature_3"] - 1)
        ) * features["feature_0"]
        return (score > 0).to_numpy(dtype=int)

    def sample(self, rows, rng):
        if rows <= 0:
            raise ValueError("rows must be positive")
        x0, x1 = rng.normal(size=(2, rows))
        category = rng.binomial(1, expit(self.kappa * x0 * x1))
        table = pd.DataFrame({"feature_0": x0, "feature_1": x1,
                              "feature_3": category})
        table["target"] = self.clean_target(table) ^ rng.binomial(1, self.noise, rows)
        return table

    def log_prob(self, table):
        """Per-row joint log density, with counting measure for binary columns."""
        validate(table)
        x0 = table["feature_0"].to_numpy()
        x1 = table["feature_1"].to_numpy()
        category = table["feature_3"].to_numpy()
        logits = self.kappa * x0 * x1
        continuous = -np.log(2 * np.pi) - (x0**2 + x1**2) / 2
        categorical = -np.logaddexp(0, (1 - 2 * category) * logits)
        correct = table["target"].to_numpy() == self.clean_target(table)
        target = np.where(correct, np.log1p(-self.noise), np.log(self.noise))
        return continuous + categorical + target


def validate(table, with_target=True):
    columns = COLUMNS if with_target else FEATURES
    if list(table.columns) != columns or table.empty:
        raise ValueError(f"Expected a nonempty table with columns {columns}")
    if not all(pd.api.types.is_numeric_dtype(table[c]) for c in columns):
        raise ValueError("All columns must be numeric")
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Table contains nonfinite values")
    discrete = ["feature_3", "target"] if with_target else ["feature_3"]
    if not table[discrete].isin([0, 1]).all().all():
        raise ValueError("feature_3 and target must contain only 0 and 1")
