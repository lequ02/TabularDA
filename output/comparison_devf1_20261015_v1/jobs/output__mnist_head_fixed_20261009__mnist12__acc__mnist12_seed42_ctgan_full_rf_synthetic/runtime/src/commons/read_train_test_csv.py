import os
import pandas as pd

def read_train_test_csv(train_csv, test_csv,
                    target_name,
                    categorical_columns):
  """
  read train and test data from csv files
  """
  
  # check if the csv files exist
  if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    raise FileNotFoundError("The csv file(s) do not exist")

  # read adult data
  train_data = pd.read_csv(train_csv)
  xtrain = train_data.drop(columns=[target_name])
  ytrain = train_data[target_name]

  test_data = pd.read_csv(test_csv)
  xtest = test_data.drop(columns=[target_name])
  ytest = test_data[target_name]

  return xtrain, xtest, ytrain, ytest, target_name, categorical_columns