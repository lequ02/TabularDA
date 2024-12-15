from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from .customErrors import *

def prepend_col_name(df, columns):
    for col in columns:
        # print(type(col), col)
        df[col] = df[col].apply(lambda x: f"{col}_{x}")
    return df

def onehot(xtrain, xtest, categorical_columns, verbose=False):
  # copy so that the original data is not modified. 
  # if original data is modified, the rows values in train.csv and test.csv will have prepend column names
  xtrain_copy = xtrain.copy()
  xtest_copy = xtest.copy()

  # Prepend column names to train and test
  xtrain_copy = prepend_col_name(xtrain_copy, categorical_columns)
  xtest_copy = prepend_col_name(xtest_copy, categorical_columns)

  # Automatically determine the numerical columns
  numerical_cols = list(set(xtrain.columns) - set(categorical_columns))
  if verbose:
    print("xtrain shape: ", xtrain.shape)
    print("xtest shape: ", xtest.shape)
    print("Numerical columns: ", numerical_cols)

  # Prepare train_prep and test_prep
  xtrain_prep = xtrain[numerical_cols]
  xtest_prep = xtest[numerical_cols]

  # xtrain_prep.to_csv("xtrain_prep_ori.csv")

    # One-Hot Encoding with modified categorical values
  for col in categorical_columns:

      # One-Hot Encoding for xtrain_copy
      train_ohe = OneHotEncoder()
      train_ohe.set_output(transform="default") # Disable the pandas output to return an array because Pandas output does not support sparse data
      xtrain_onehot = train_ohe.fit_transform(xtrain_copy[col].values.reshape(-1,1))
      xtrain_onehot = xtrain_onehot.toarray()
      xtrain_onehot = pd.DataFrame(xtrain_onehot, columns = xtrain_copy[col].unique())


      # print("xtrain_onehot", xtrain_onehot)
      # print("xtrain_prep: ", xtrain_prep)
      # print("xtrain_onehot shape: ", xtrain_onehot.shape)
      # print("xtrain_prep shape: ", xtrain_prep.shape)

      xtrain_prep = pd.concat([xtrain_prep, xtrain_onehot], axis=1) # when dropping missing values, index won't be continuous, so concat (xtrain_prep, xtrain_onehot, axis=1) will not match

      # One-Hot Encoding for xtest_copy
      test_ohe = OneHotEncoder()
      test_ohe.set_output(transform="default") # Disable the pandas output to return an array because Pandas output does not support sparse data
      xtest_onehot = test_ohe.fit_transform(xtest_copy[col].values.reshape(-1,1))
      xtest_onehot = xtest_onehot.toarray()
      xtest_onehot = pd.DataFrame(xtest_onehot, columns = xtest_copy[col].unique())
      xtest_onehot = xtest_onehot.set_index(xtest_copy.index)
      xtest_prep = pd.concat([xtest_prep, xtest_onehot], axis=1)

      # Check differences between xtrain_copy and xtest_copy
      dif1 = set(xtest_copy[col].unique()) - set(xtrain_copy[col].unique())
      if dif1 != set():
        # safeguard 
        # make sure that the categorical values in xtest_copy are the same as in xtrain_copy
        error_message = f"""
        Differences found between xtest and xtrain in column: "{col}"
        Number of unique values in test (test - train): {len(dif1)}
        Unique values: {dif1}
        """
        raise TestTrainDiffError(error_message)

      if verbose:
        print(f"Differences between xtest and xtrain in column: {col}")
        print(len(dif1), dif1)

        
  return xtrain_prep, xtest_prep
