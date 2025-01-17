import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import MinMaxScaler
from GaussMix_nb import GMMNaiveBayes

def create_label_gaussianNB(xtrain, ytrain, xtest, target_name, filename=None):
  
  x_null = xtrain.isnull().sum()
  print("\n\nNull values in xtrain: ", x_null)

  x_null = xtest.isnull().sum()
  print("\n\nNull values in xtest: ", x_null)

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


def create_label_categoricalNB(xtrain, ytrain, xtest, target_name, filename=None):

  x_null = xtrain.isnull().sum()
  print("\n\nNull values in xtrain: ", x_null)

  x_null = xtest.isnull().sum()
  print("\n\nNull values in xtest: ", x_null)

  # xtrain and ytrain are one hot encoded
  # normalize data to [0,1] because CategoricalNB only accepts non negative values
  scaler = MinMaxScaler()
  xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)

  xtrain_scaled = scaler.fit_transform(xtrain)
  xtest_scaled = scaler.transform(xtest)

  # xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
  # xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  xtrain_scaled = pd.DataFrame(xtrain_scaled, columns = xtrain.columns)
  xtest_scaled = pd.DataFrame(xtest_scaled, columns = xtest.columns)
  # xtrain_scaled = xtrain_scaled.reindex(sorted(xtrain_scaled.columns), axis=1)
  # xtest_scaled = xtest_scaled.reindex(sorted(xtest_scaled.columns), axis=1)

  cnb = CategoricalNB().fit(xtrain_scaled, ytrain)
  ytest = cnb.predict(xtest_scaled)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test


def create_label_gmmNB(xtrain, ytrain, xtest, target_name, filename=None):

  x_null = xtrain.isnull().sum()
  print("\n\nNull values in xtrain: ", x_null)

  x_null = xtest.isnull().sum()
  print("\n\nNull values in xtest: ", x_null)

  # xtrain and ytrain are one hot encoded
  xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  gmmnb = GMMNaiveBayes(n_components=3).fit(xtrain, ytrain)
  ytest = gmmnb.predict(xtest)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test