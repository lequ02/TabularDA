from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


def onehot(xtrain, xtest, categorical_columns, verbose=False):
    xtrain_prep, xtest_prep = onehot_many(xtrain, (xtest,), categorical_columns)

    if verbose:
        print("xtrain_prep shape:", xtrain_prep.shape)
        print("xtest_prep shape:", xtest_prep.shape)

    return xtrain_prep, xtest_prep


def onehot_many(xtrain, holdouts, categorical_columns):
    """Fit once on training data; keep binary indicators in one-byte columns."""
    frames = (xtrain, *holdouts)
    if not categorical_columns:
        return tuple(frame.copy() for frame in frames)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.uint8)
    encoder.fit(xtrain[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    numerical_cols = [column for column in xtrain.columns if column not in categorical_columns]
    results = []
    for frame in frames:
        encoded = pd.DataFrame(encoder.transform(frame[categorical_columns]),
                               columns=encoded_columns, index=frame.index)
        results.append(pd.concat([frame[numerical_cols], encoded], axis=1))
    return tuple(results)
