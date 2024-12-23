
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml


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
    # print(y.income.unique())
    # the y values are "<=50K", "<=50K."", "">50K", ">50K."
    # We need to remove the '.' from the values
    y['income'] = y['income'].str.replace('.', '', regex=False)\

    # # handle missing values
    # xy = pd.concat([x, y], axis=1)
    # xy.dropna(inplace=True)
    # xy.reset_index(drop=True, inplace=True) # must always reset index
    # y = pd.DataFrame(xy['income'])
    # x = xy.drop(columns=['income'])
    # print(x.shape, y.shape)
    # print(x.isnull().sum())

# missing values
# (48842, 14)
# workclass         963 ~ 5%
# occupation        966 ~ 5%
# native-country    274 ~ 1%

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


def load_census_kdd(verbose=False):  # link in paper could be wrong. id=117
    # x, y = load_dataset(117, verbose)
    # # the y values are <=50K, <=50K., >50K >50K.
    # # We need to remove the '.' from the values
    # y['income'] = y['income'].str.replace('.', '', regex=False)
    # # print(y.income.unique())
    # return x, y

    pass


def load_covertype(verbose=False):

    # dummy = load_dataset(31, verbose)
    # print(dummy)
    # print(dummy[0].shape)
    # print(dummy[1].shape)
    # print(dummy[0].columns)
    # print(dummy[1].columns)
    # print(type(dummy[0]))
    # print(type(dummy[1]))
    # print(dummy[0].head())
    # print(dummy[1].head())

    return load_dataset(31, verbose)

def load_intrusion(verbose=False):
    df = pd.read_csv('../data/intrusion/kddcup.data.corrected.csv')
    y = df[['target']]
    x = df.drop(columns=['target'])

    return x, y

def load_credit(verbose=False):
    df = pd.read_csv('../data/credit/creditcard.csv')
    y = df[['Class']]
    x = df.drop(columns=['Class'])

    return x, y


def load_mnist28(verbose=False):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X, y = mnist['data'], mnist['target']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y, columns=['label'])

    return X, y
