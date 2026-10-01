"""Supervised labelers trained only on the original training split."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from .oracle import FEATURES

LABELERS = ["gaussian", "categorical", "pca_gmm", "rf", "xgb", "dnn"]


def make_dnn(seed):
    """A fresh model; scaling and early stopping use only its training data."""
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 64), batch_size=256, max_iter=100,
        early_stopping=True, random_state=seed))


class PCAGMMClassifier(ClassifierMixin, BaseEstimator):
    """One four-component Gaussian mixture per class in a two-dimensional PCA space."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, x, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(counts / counts.sum())
        self.models_ = [GaussianMixture(n_components=4, random_state=self.random_state).fit(x[np.asarray(y) == c])
                        for c in self.classes_]
        return self

    def predict(self, x):
        scores = np.column_stack([m.score_samples(x) for m in self.models_]) + self.log_priors_
        return self.classes_[scores.argmax(axis=1)]


def fit_labeler(method, train, seed):
    if method == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif method == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=seed, n_jobs=1, eval_metric="logloss")
    elif method == "dnn":
        model = make_dnn(seed)
    elif method == "gaussian":
        model = GaussianNB()
    elif method == "categorical":
        bins = ColumnTransformer([
            ("continuous", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile", subsample=None),
             ["feature_0", "feature_1"]),
            ("binary", "passthrough", ["feature_3"]),
        ])
        model = make_pipeline(bins, CategoricalNB(min_categories=[10, 10, 2]))
    elif method == "pca_gmm":
        model = make_pipeline(StandardScaler(), PCA(n_components=2), PCAGMMClassifier(random_state=seed))
    else:
        raise ValueError(f"Unknown labeler: {method}")
    return model.fit(train[FEATURES], train["target"])
