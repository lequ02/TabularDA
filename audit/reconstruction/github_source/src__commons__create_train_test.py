from sklearn.model_selection import train_test_split
import pandas as pd

def create_train_test_old(data, target_name, test_size=0.2, random_state=42, stratify=None):
    X = data.drop(columns=[target_name])
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=stratify)
    # always reset the index after splitting so that the index is continuous
    # or else sklearn will cause an error in onehot.py
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def create_train_test(data, target_name, categorical_columns, test_size=0.2, random_state=42, stratify=None):
    """
    This function is used to split the data into train and test data
    It makes sure that the unique values in the categorical columns are the same in both train and test data
    If there are unique values in train and not in test, redistribute values to both train and test. And vice versa
    If there is only 1 occurence of a unique value in the dataset, it will be in the train data    
    """


    df = data.copy()
    test_size_frac  = test_size/df.shape[0]
    print("\ntest_size_frac", test_size_frac)
    print("stratify", stratify)
    # initial split
    dftrain, dftest = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)

    # make sure that the unique values in the categorical columns are the same in both train and test data
    for cat_col in categorical_columns:
        # check if there are unique values in train data that are not in test data
        diff_train_test = set(dftrain[cat_col]) - set(dftest[cat_col])

        # print("train-test", set(dftrain[cat_col]) - set(dftest[cat_col]))
        # print("test-train",set(dftest[cat_col]) - set(dftrain[cat_col]))
        if diff_train_test != set():
            print(f"Train data has unique values in col '{cat_col}' not in test (train-test)")
            print("Unique values in train but not in test: ", diff_train_test)
            print("Redisributing the unique values to train and test data")
            for val in diff_train_test:
                matching_rows = dftrain[dftrain[cat_col] == val]
                print("\n\nmatching_rows train", matching_rows)

                if matching_rows.shape[0] == 1:
                    # if there is only 1 occurence of a unique value in the dataset, keep it in the train data 
                    print("Only 1 occurence of a unique value in the dataset")
                    print("Keeping it in the train data")
                    pass
                else:
                    dftest = dftest[dftest[cat_col] != val]
                    # use test_size_frac because if test_size > 1, and matching_rows < test_size, it will cause an error
                    # use matching_rows[target_name] because if use stratify will cause shape mismatch error
                    temp_train, temp_test = train_test_split(matching_rows, test_size=test_size_frac, random_state=random_state, stratify=matching_rows[target_name]) 
                    dftrain = pd.concat([dftrain, temp_train]).reset_index(drop=True)
                    dftest = pd.concat([dftest, temp_test]).reset_index(drop=True)

        # check if there are unique values in test data that are not in train data
        diff_test_train = set(dftest[cat_col]) - set(dftrain[cat_col])
        if diff_test_train != set():
            print(f"Test data has unique values in col '{cat_col}' not in train (test-train)")
            print("Unique values in test but not in train: ", diff_test_train)
            print("Redisributing the unique values to train and test data")
            for val in diff_test_train:
                matching_rows = dftest[dftest[cat_col] == val]

                if matching_rows.shape[0] == 1:
                    # if there is only 1 occurence of a unique value in the dataset, put it in the train data 
                    dftrain = pd.concat([dftrain, matching_rows]).reset_index(drop=True)
                    dftest = dftest[dftest[cat_col] != val]
                else:
                    dftest = dftest[dftest[cat_col] != val]
                    # use test_size_frac because if test_size > 1, and matching_rows < test_size, it will cause an error
                    # use matching_rows[target_name] because if use stratify will cause shape mismatch error
                    temp_train, temp_test = train_test_split(matching_rows, test_size=test_size_frac, random_state=random_state, stratify=matching_rows[target_name]) 
                    dftrain = pd.concat([dftrain, temp_train]).reset_index(drop=True)
                    dftrain = pd.concat([dftrain, temp_train]).reset_index(drop=True)
                    dftest = pd.concat([dftest, temp_test]).reset_index(drop=True)

        # if there is only 1 occurence of a unique value in the dataset, it will be in the train data 

    X_train = dftrain.drop(columns=[target_name])
    y_train = dftrain[target_name]
    X_test = dftest.drop(columns=[target_name])
    y_test = dftest[target_name]

    # always reset the index after splitting so that the index is continuous
    # or else sklearn will cause an error in onehot.py
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


