from sklearn.model_selection import train_test_split

def create_train_test(data, target_name, test_size=0.2, random_state=42, stratify=None):
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