"""Strict log/share conversion shared by the opt-in News experiment."""

import numpy as np


def log_target(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Log targets require finite, strictly positive shares')
    return np.log(values)


def inverse_log_target(values):
    with np.errstate(over='raise', invalid='raise', under='raise'):
        shares = np.exp(np.asarray(values, dtype=np.float64))
    if not np.isfinite(shares).all() or np.any(shares <= 0):
        raise ValueError('Inverse-log predictions must be finite, strictly positive shares')
    return shares
