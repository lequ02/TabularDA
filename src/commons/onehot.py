from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from .customErrors import *


def onehot(xtrain, xtest, categorical_columns, verbose=False):
    # Copy data to avoid modifying originals
    xtrain_copy = xtrain.copy()
    xtest_copy = xtest.copy()

    # Check for differences between train and test categories before encoding
    for col in categorical_columns:
        dif1 = set(xtest_copy[col].unique()) - set(xtrain_copy[col].unique())
        if dif1:
            error_message = f"""
            Differences found between xtest and xtrain in column: "{col}"
            Number of unique values in test (test - train): {len(dif1)}
            Unique values: {dif1}
            """
            raise TestTrainDiffError(error_message)

    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(xtrain_copy[categorical_columns])

    # Transform both train and test sets
    xtrain_encoded = encoder.transform(xtrain_copy[categorical_columns])
    xtest_encoded = encoder.transform(xtest_copy[categorical_columns])

    # Create DataFrame for one-hot encoded columns
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    xtrain_onehot_df = pd.DataFrame(xtrain_encoded, columns=encoded_columns, index=xtrain_copy.index)
    xtest_onehot_df = pd.DataFrame(xtest_encoded, columns=encoded_columns, index=xtest_copy.index)

    # Concatenate numerical columns and one-hot encoded columns
    numerical_cols = list(set(xtrain.columns) - set(categorical_columns))
    xtrain_prep = pd.concat([xtrain[numerical_cols], xtrain_onehot_df], axis=1)
    xtest_prep = pd.concat([xtest[numerical_cols], xtest_onehot_df], axis=1)

    if verbose:
        print("xtrain_prep shape:", xtrain_prep.shape)
        print("xtest_prep shape:", xtest_prep.shape)

    return xtrain_prep, xtest_prep