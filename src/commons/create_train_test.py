from sklearn.model_selection import train_test_split

def create_train_test(data, target_name, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_name])
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test