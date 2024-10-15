from sklearn import impute
from sklearn import set_config
import pandas as pd
# return a df instead of array
set_config(transform_output = "pandas")


def handle_missing_values(x, y, target_name, strategy):
    data = pd.concat([x, y], axis=1)

    # handle missing values
    if strategy == 'drop':
        data.dropna(inplace=True)
        print(data)
        print(data.shape)
        print(data.isnull().sum())
        data.reset_index(drop=True, inplace=True) # must always reset index so there's no missing index
    else:
        imputer = impute.SimpleImputer(strategy)
        data = imputer.fit_transform(data)

    y = pd.DataFrame(data[target_name])
    x = data.drop(columns=[target_name])
    return x, y