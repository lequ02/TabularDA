"""Explicit normalized refits for L_test on the mixed-data benchmarks."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.mixture import GaussianMixture


def fit_boundary(sample):
    """Exact least-error beta over (0,1); Beta(1/2,1/2) posterior noise estimate."""
    a = (sample.feature_0 + sample.feature_1).to_numpy()
    b = ((2*sample.feature_3 - 1) * sample.feature_0).to_numpy()
    y = sample.target.to_numpy()
    crossings = np.divide(-a, b, out=np.full(len(a), np.inf), where=b != 0)
    indices = np.flatnonzero((crossings > 0) & (crossings < 1))
    indices = indices[np.argsort(crossings[indices])]
    roots, starts = np.unique(crossings[indices], return_index=True)
    deltas = np.where(y[indices] == (b[indices] > 0), -1, 1)
    changes = np.add.reduceat(deltas, starts)
    initial_errors = np.count_nonzero((np.where(a == 0, b, a) > 0) != y)
    errors = np.r_[initial_errors, initial_errors + np.cumsum(changes)]
    best = errors.argmin()
    bounds = np.r_[0, roots, 1]
    beta = (bounds[best] + bounds[best+1]) / 2
    noise = min((errors[best] + .5) / (len(sample) + 1), .5)
    return beta, noise


def mixed_test_log_prob(oracle, sample, test, seed):
    x_train = sample[["feature_0", "feature_1"]].to_numpy()
    x_test = test[["feature_0", "feature_1"]].to_numpy()
    if oracle.dataset == "grid":
        feature_log = sum(GaussianMixture(5, random_state=seed).fit(x_train[:, [j]]).score_samples(x_test[:, [j]])
                          for j in range(2))
    else:
        count, covariance = (1, "diag") if oracle.dataset == "gaussian" else (8, "spherical")
        feature_log = GaussianMixture(count, covariance_type=covariance, random_state=seed).fit(x_train).score_samples(x_test)

    design = np.column_stack((np.ones(len(sample)), sample.feature_0 * sample.feature_1))
    category = sample.feature_3.to_numpy()

    def objective(theta):
        logits = design @ theta
        loss = np.logaddexp(0, logits).sum() - category @ logits + .5 * theta @ theta
        gradient = design.T @ (expit(logits) - category) + theta
        return loss, gradient

    fitted = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError(fitted.message)
    logits = fitted.x[0] + fitted.x[1] * (test.feature_0 * test.feature_1).to_numpy()
    category_log = -np.logaddexp(0, (1-2*test.feature_3.to_numpy()) * logits)
    beta, noise = fit_boundary(sample)
    clean = (test.feature_1 + (1 + beta * (2*test.feature_3-1)) * test.feature_0 > 0).to_numpy()
    target_log = np.where(test.target.to_numpy() == clean, np.log1p(-noise), np.log(noise))
    return feature_log + category_log + target_log
