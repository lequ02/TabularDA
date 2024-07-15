
import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_dataset(dataset_id, verbose=False):
    # fetch dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    if verbose:
        # metadata
        print(dataset.metadata)
        # variable information
        print(dataset.variables)

    return X, y

def load_adult(verbose=False):
    x, y = load_dataset(2, verbose)
    # the y values are <=50K, <=50K., >50K >50K.
    # We need to remove the '.' from the values
    y['income'] = y['income'].str.replace('.', '', regex=False)
    # print(y.income.unique())
    return x, y

def load_news(verbose=False):
    return load_dataset(332, verbose)

# x_news, y_news = load_news()
# print("News dataset features shape:", x_news.shape)
# print("News dataset target shape:", y_news.shape)


def load_census(verbose=False):
    x, y = load_dataset(20, verbose)
    # the y values are <=50K, <=50K., >50K >50K.
    # We need to remove the '.' from the values
    y['income'] = y['income'].str.replace('.', '', regex=False)
    # print(y.income.unique())
    return x, y





