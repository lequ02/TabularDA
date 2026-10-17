from sklearn import impute
from sklearn import set_config
import pandas as pd
# return a df instead of array
set_config(transform_output = "pandas")


def handle_missing_values(x, y, target_name, strategy, fitted_imputer=None,
                          return_imputer=False):
    """Impute features only; pass the returned imputer to transform holdouts."""
    x = x.copy()
    y = y.copy()
    if isinstance(y, pd.DataFrame):
        if target_name not in y:
            raise KeyError(f"Target column {target_name!r} is missing")
        target = y[target_name]
    else:
        target = y.rename(target_name)
    if strategy == "drop":
        keep = x.notna().all(axis=1) & target.notna()
        x = x.loc[keep].reset_index(drop=True)
        target = target.loc[keep].reset_index(drop=True)
        result = (x, target.to_frame(name=target_name))
        return (*result, None) if return_imputer else result
    if fitted_imputer is None:
        fitted_imputer = impute.SimpleImputer(strategy=strategy)
        transformed = fitted_imputer.fit_transform(x)
    else:
        transformed = fitted_imputer.transform(x)
    x = pd.DataFrame(transformed, columns=x.columns, index=x.index)
    result = (x.reset_index(drop=True), target.reset_index(drop=True).to_frame(name=target_name))
    return (*result, fitted_imputer) if return_imputer else result
