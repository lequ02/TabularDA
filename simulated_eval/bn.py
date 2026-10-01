"""Ancestral samples, exact joint probabilities, and fixed-graph BN refits."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from scipy.special import logsumexp, xlogy

TARGETS = {"asia": "dysp", "alarm": "BP", "child": "Disease", "insurance": "Accident"}


class BNOracle:
    def __init__(self, dataset):
        self.dataset = dataset
        self.target = TARGETS[dataset]
        model = BIFReader(str(Path(__file__).parent / "oracles" / f"{dataset}.bif"), n_jobs=1).get_model()
        model.check_model()
        self.nodes = list(nx.topological_sort(model))
        self.tables = [(cpd.variables, np.asarray(cpd.values)) for node in self.nodes for cpd in [model.get_cpds(node)]]
        self.cardinalities = {node: model.get_cardinality(node) for node in self.nodes}
        self.features = [node for node in self.nodes if node != self.target]
        self.columns = self.features + ["target"]
        self.discrete = self.columns
        self.categories = {("target" if node == self.target else node): list(range(self.cardinalities[node]))
                           for node in self.nodes}
        self.classes = np.arange(self.cardinalities[self.target])

    def validate(self, table, with_target=True):
        columns = self.columns if with_target else self.features
        if list(table.columns) != columns or table.empty:
            raise ValueError(f"Expected a nonempty table with columns {columns}")
        for column in columns:
            if not table[column].isin(self.categories[column]).all():
                raise ValueError(f"Unknown state in {column}")

    def sample(self, rows, rng):
        if rows <= 0:
            raise ValueError("rows must be positive")
        codes = {}
        for variables, values in self.tables:
            probabilities = (values[(slice(None), *[codes[p] for p in variables[1:]])]
                             if len(variables) > 1 else np.broadcast_to(values[:, None], (len(values), rows)))
            cumulative = probabilities.cumsum(axis=0)
            cumulative[-1] = 1
            codes[variables[0]] = (rng.random(rows) >= cumulative).sum(axis=0)
        return pd.DataFrame(codes).rename(columns={self.target: "target"})[self.columns]

    def _log_prob(self, table, tables):
        codes = {c: table[c].to_numpy(dtype=int) for c in self.columns}
        result = np.zeros(len(table))
        for variables, values in tables:
            indices = tuple(codes["target" if v == self.target else v] for v in variables)
            result += xlogy(1, values[indices])
        return result

    def log_prob(self, table):
        self.validate(table)
        return self._log_prob(table, self.tables)

    def test_log_prob(self, sample, test, seed):
        """Dirichlet(1/2) posterior-mean CPDs, applied to every parent configuration."""
        fitted = []
        for variables, values in self.tables:
            indices = tuple(sample["target" if v == self.target else v].to_numpy(dtype=int) for v in variables)
            flat = np.ravel_multi_index(indices, values.shape)
            counts = np.bincount(flat, minlength=values.size).reshape(values.shape) + .5
            fitted.append((variables, counts / counts.sum(axis=0)))
        return self._log_prob(test, fitted)

    def posterior(self, features):
        scores = []
        for state in self.classes:
            complete = features[self.features].copy()
            complete["target"] = state
            scores.append(self.log_prob(complete))
        scores = np.column_stack(scores)
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

    def clean_target(self, features):
        return self.posterior(features).argmax(axis=1)

    def accuracy_ceiling(self, test):
        return float(self.posterior(test).max(axis=1).mean())
