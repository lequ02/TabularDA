from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def prepend_col_name(df, columns):
    for col in columns:
        # print(type(col), col)
        df[col] = df[col].apply(lambda x: f"{col}_{x}")
    return df

def onehot(xtrain, xtest, categorical_columns, verbose=False):
  # Prepend column names to train and test
  xtrain = prepend_col_name(xtrain, categorical_columns)
  xtest = prepend_col_name(xtest, categorical_columns)

  # Automatically determine the numerical columns
  numerical_cols = list(set(xtrain.columns) - set(categorical_columns))
  if verbose:
    print("Numerical columns: ", numerical_cols)

  # Prepare train_prep and test_prep
  xtrain_prep = xtrain[numerical_cols]
  xtest_prep = xtest[numerical_cols]

    # One-Hot Encoding with modified categorical values
  for col in categorical_columns:

      # One-Hot Encoding for xtrain
      xtrain_onehot = OneHotEncoder().fit_transform(xtrain[col].values.reshape(-1,1))
      xtrain_onehot = xtrain_onehot.toarray()
      xtrain_onehot = pd.DataFrame(xtrain_onehot, columns = xtrain[col].unique())
      xtrain_prep = pd.concat([xtrain_prep, xtrain_onehot], axis=1)

      # One-Hot Encoding for xtest
      xtest_onehot = OneHotEncoder().fit_transform(xtest[col].values.reshape(-1,1))
      xtest_onehot = xtest_onehot.toarray()
      xtest_onehot = pd.DataFrame(xtest_onehot, columns = xtest[col].unique())
      xtest_onehot = xtest_onehot.set_index(xtest.index)
      xtest_prep = pd.concat([xtest_prep, xtest_onehot], axis=1)

      # Check differences between xtrain and xtest
      if verbose:
        print(f"Differences between xtrain and xtest in column: {col}")
        dif1 = set(xtest[col].unique()) - set(xtrain[col].unique())
        # print(col)
        print(len(dif1), dif1)

  return xtrain_prep, xtest_prep
