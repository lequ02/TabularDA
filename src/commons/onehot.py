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
    print("xtrain shape: ", xtrain.shape)
    print("xtest shape: ", xtest.shape)
    print("Numerical columns: ", numerical_cols)

  # Prepare train_prep and test_prep
  xtrain_prep = xtrain[numerical_cols]
  xtest_prep = xtest[numerical_cols]

  # xtrain_prep.to_csv("xtrain_prep_ori.csv")

    # One-Hot Encoding with modified categorical values
  for col in categorical_columns:

      # One-Hot Encoding for xtrain
      train_ohe = OneHotEncoder()
      train_ohe.set_output(transform="default") # Disable the pandas output to return an array because Pandas output does not support sparse data
      xtrain_onehot = train_ohe.fit_transform(xtrain[col].values.reshape(-1,1))
      xtrain_onehot = xtrain_onehot.toarray()
      xtrain_onehot = pd.DataFrame(xtrain_onehot, columns = xtrain[col].unique())


      print("xtrain_onehot", xtrain_onehot)
      print("xtrain_prep: ", xtrain_prep)
      print("xtrain_onehot shape: ", xtrain_onehot.shape)
      print("xtrain_prep shape: ", xtrain_prep.shape)

      xtrain_prep = pd.concat([xtrain_prep, xtrain_onehot], axis=1) # when dropping missing values, index won't be continuous, so concat (xtrain_prep, xtrain_onehot, axis=1) will not match

      # One-Hot Encoding for xtest
      test_ohe = OneHotEncoder()
      test_ohe.set_output(transform="default") # Disable the pandas output to return an array because Pandas output does not support sparse data
      xtest_onehot = test_ohe.fit_transform(xtest[col].values.reshape(-1,1))
      xtest_onehot = xtest_onehot.toarray()
      xtest_onehot = pd.DataFrame(xtest_onehot, columns = xtest[col].unique())
      xtest_onehot = xtest_onehot.set_index(xtest.index)
      xtest_prep = pd.concat([xtest_prep, xtest_onehot], axis=1)

      # Check differences between xtrain and xtest
      if verbose:
        print(f"Differences between xtest and xtrain in column: {col}")
        dif1 = set(xtest[col].unique()) - set(xtrain[col].unique())
        # print(col)
        print(len(dif1), dif1)


      # print("xtrain_prep:", xtrain_prep.shape)
      # print("xtest_prep:", xtest_prep.shape)



      # xtrain_prep.to_csv("xtrain_prep.csv")
  return xtrain_prep, xtest_prep
