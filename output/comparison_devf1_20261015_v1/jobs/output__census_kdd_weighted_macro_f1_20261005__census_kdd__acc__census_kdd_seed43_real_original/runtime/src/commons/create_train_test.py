from sklearn.model_selection import train_test_split


def create_train_test_old(data, target_name, test_size=0.2, random_state=42, stratify=None):
    X = data.drop(columns=[target_name])
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))


def create_train_test(data, target_name, categorical_columns=None, test_size=0.2,
                      random_state=42, stratify=None, return_source_ids=False):
    """Split source rows exactly once; holdout-only categories remain in holdout."""
    if target_name not in data.columns:
        raise KeyError(f"Target column {target_name!r} is missing")
    stratify_values = data[stratify] if isinstance(stratify, str) else stratify
    positions = range(len(data))
    train_positions, test_positions = train_test_split(
        positions, test_size=test_size, random_state=random_state,
        stratify=stratify_values
    )
    train_df = data.iloc[train_positions]
    test_df = data.iloc[test_positions]
    X_train = train_df.drop(columns=[target_name]).reset_index(drop=True)
    y_train = train_df[target_name].reset_index(drop=True)
    X_test = test_df.drop(columns=[target_name]).reset_index(drop=True)
    y_test = test_df[target_name].reset_index(drop=True)
    if return_source_ids:
        return (X_train, X_test, y_train, y_test,
                train_df.index.to_numpy(), test_df.index.to_numpy())
    return X_train, X_test, y_train, y_test
