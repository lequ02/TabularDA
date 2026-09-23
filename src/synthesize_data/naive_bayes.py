import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
import numpy as np
from GaussMix_nb import GMMNaiveBayes


def _fit_quantile_bins(xtrain, xtest, n_bins=10):
  """Discretize continuous inputs using cut points learned from training rows."""
  train_values = np.asarray(xtrain, dtype=float)
  test_values = np.asarray(xtest, dtype=float)
  if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
    raise ValueError("CategoricalNB inputs must be finite numeric values.")
  bins = []
  for column in range(train_values.shape[1]):
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(train_values[:, column], quantiles))
    bins.append(edges)
  train_codes = np.column_stack([
    np.digitize(train_values[:, column], bins[column])
    for column in range(train_values.shape[1])
  ]).astype(np.int64)
  test_codes = np.column_stack([
    np.digitize(test_values[:, column], bins[column])
    for column in range(test_values.shape[1])
  ]).astype(np.int64)
  for column in range(train_codes.shape[1]):
    test_codes[:, column] = np.clip(
      test_codes[:, column], train_codes[:, column].min(), train_codes[:, column].max()
    )
  return train_codes, test_codes

def create_label_gaussianNB(xtrain, ytrain, xtest, target_name, filename=None):

  # xtrain and ytrain are one hot encoded
  xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  gnb = GaussianNB().fit(xtrain, ytrain)
  ytest = gnb.predict(xtest)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test


def create_label_categoricalNB(xtrain, ytrain, xtest, target_name, filename=None, alpha=1.0, force_alpha=True):
  """
  alpha and force_alpha are parameters for Laplace smoothing (sklearn default is alpha=1.0 and force_alpha=True)
  for no smoothing, set alpha=0 and force_alpha=False
  """

  # CategoricalNB needs discrete category IDs. Quantile cut points are fitted
  # from training data only; min-max scaling continuous values collapses them
  # to the same integer category after CategoricalNB's internal conversion.
  xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  xtrain_codes, xtest_codes = _fit_quantile_bins(xtrain, xtest)
  cnb = CategoricalNB(alpha=alpha, force_alpha=force_alpha).fit(xtrain_codes, ytrain)
  ytest = cnb.predict(xtest_codes)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test


def create_label_gmmNB(xtrain, ytrain, xtest, target_name, filename=None):

  # xtrain and ytrain are one hot encoded
  xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  numeric_cols = [
    column for column in xtrain.columns
    if not set(xtrain[column].dropna().unique()).issubset({0, 1, 0.0, 1.0})
  ]
  gmmnb = GMMNaiveBayes(n_components=3).fit(xtrain, ytrain, numeric_cols=numeric_cols)
  ytest = gmmnb.predict(xtest)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test
