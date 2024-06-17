import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB

def create_label_gaussianNB(xtrain, ytrain, xtest, target_name, filename=None):
  xtrain =xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  gnb = GaussianNB().fit(xtrain, ytrain)
  ytest = gnb.predict(xtest)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test


def create_label_categoricalNB(xtrain, ytrain, xtest, target_name, filename=None):
  xtrain =xtrain.reindex(sorted(xtrain.columns), axis=1)
  xtest = xtest.reindex(sorted(xtest.columns), axis=1)
  cnb = CategoricalNB().fit(xtrain, ytrain)
  ytest = cnb.predict(xtest)

  ytest_df = pd.DataFrame(ytest, columns = [target_name])
  test = pd.concat([xtest, ytest_df], axis=1)
  if filename:
    test.to_csv(filename)
  return test