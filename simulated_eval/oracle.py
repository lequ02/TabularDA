"""Generate and score the exact distribution used by the benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import norm

FEATURES = ["feature_0", "feature_1", "feature_3"]
COLUMNS = FEATURES + ["target"]


@dataclass(frozen=True)
class Oracle:
    kappa: float = 2.0
    beta: float = 0.75
    noise: float = 0.10
    dataset: str = "gaussian"
    spread: float = 0.20

    features = FEATURES
    columns = COLUMNS
    discrete = ["feature_3", "target"]
    classes = np.arange(2)
    categories = {"feature_3": [0, 1], "target": [0, 1]}

    def __post_init__(self):
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        if not 0 < self.beta < 1:
            raise ValueError("beta must be between 0 and 1")
        if not 0 < self.noise < 0.5:
            raise ValueError("noise must be between 0 and 0.5")
        if self.dataset not in ("gaussian", "grid", "ring"):
            raise ValueError(f"Unknown mixed dataset: {self.dataset}")
        if not np.isfinite(self.spread) or self.spread <= 0:
            raise ValueError("spread must be finite and positive")

    def validate(self, table, with_target=True):
        validate(table, with_target)

    def accuracy_ceiling(self, test):
        return 1 - self.noise

    def posterior(self, features):
        probability = self.noise + (1 - 2*self.noise) * self.clean_target(features)
        return np.column_stack((1 - probability, probability))

    def feature_log_prob(self, table):
        x = table[["feature_0", "feature_1"]].to_numpy()
        if self.dataset == "gaussian":
            return -np.log(2 * np.pi) - (x**2).sum(axis=1) / 2
        if self.dataset == "grid":
            scale = np.sqrt(8 + self.spread**2)
            means = np.arange(-4, 5, 2) / scale
            components = norm.logpdf(x[:, :, None], means, self.spread / scale)
            return (logsumexp(components, axis=2) - np.log(5)).sum(axis=1)
        scale = np.sqrt(.5 + self.spread**2)
        angles = np.arange(8) * np.pi / 4
        centers = np.column_stack((np.cos(angles), np.sin(angles))) / scale
        variance = (self.spread / scale)**2
        squared = ((x[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        return logsumexp(-np.log(2*np.pi*variance) - squared/(2*variance), axis=1) - np.log(8)

    def clean_target(self, features):
        score = features["feature_1"] + (
            1 + self.beta * (2 * features["feature_3"] - 1)
        ) * features["feature_0"]
        return (score > 0).to_numpy(dtype=int)

    def sample(self, rows, rng):
        if rows <= 0:
            raise ValueError("rows must be positive")
        if self.dataset == "gaussian":
            x0, x1 = rng.normal(size=(2, rows))
        elif self.dataset == "grid":
            x = rng.choice(np.arange(-4, 5, 2), size=(rows, 2)) + self.spread * rng.normal(size=(rows, 2))
            x0, x1 = (x / np.sqrt(8 + self.spread**2)).T
        else:
            angles = rng.integers(8, size=rows) * np.pi / 4
            centers = np.column_stack((np.cos(angles), np.sin(angles)))
            x = centers + self.spread * rng.normal(size=(rows, 2))
            x0, x1 = (x / np.sqrt(.5 + self.spread**2)).T
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
        continuous = self.feature_log_prob(table)
        categorical = -np.logaddexp(0, (1 - 2 * category) * logits)
        correct = table["target"].to_numpy() == self.clean_target(table)
        target = np.where(correct, np.log1p(-self.noise), np.log(self.noise))
        return continuous + categorical + target

    def test_log_prob(self, sample, test, seed):
        from .density import mixed_test_log_prob

        return mixed_test_log_prob(self, sample, test, seed)


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
