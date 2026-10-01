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
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

LABELERS = ["gaussian", "categorical", "pca_gmm", "rf", "xgb", "dnn"]


def feature_encoder(oracle):
    categorical = [c for c in oracle.features if c in oracle.discrete]
    continuous = [c for c in oracle.features if c not in oracle.discrete]
    return ColumnTransformer([
        ("continuous", "passthrough", continuous),
        ("categorical", OneHotEncoder(categories=[oracle.categories[c] for c in categorical],
                                     sparse_output=False), categorical),
    ])


def make_dnn(seed, oracle):
    """A fresh model; scaling and early stopping use only its training data."""
    return make_pipeline(feature_encoder(oracle), StandardScaler(), MLPClassifier(
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


def fit_labeler(method, train, seed, oracle):
    classes, labels = np.unique(train.target, return_inverse=True)
    if method == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif method == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=seed, n_jobs=1,
                              eval_metric="mlogloss" if len(classes) > 2 else "logloss")
    elif method == "dnn":
        return make_dnn(seed, oracle).fit(train[oracle.features], labels), classes
    elif method == "gaussian":
        model = GaussianNB()
    elif method == "categorical":
        continuous = [c for c in oracle.features if c not in oracle.discrete]
        categorical = [c for c in oracle.features if c in oracle.discrete]
        bins = ColumnTransformer([
            ("continuous", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile", subsample=None),
             continuous),
            ("categorical", "passthrough", categorical),
        ])
        sizes = [10] * len(continuous) + [len(oracle.categories[c]) for c in categorical]
        return make_pipeline(bins, CategoricalNB(min_categories=sizes)).fit(train[oracle.features], labels), classes
    elif method == "pca_gmm":
        model = make_pipeline(StandardScaler(), PCA(n_components=2), PCAGMMClassifier(random_state=seed))
    else:
        raise ValueError(f"Unknown labeler: {method}")
    return make_pipeline(feature_encoder(oracle), model).fit(train[oracle.features], labels), classes
