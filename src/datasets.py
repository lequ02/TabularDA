
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
    return load_dataset(117, verbose)



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
    # CTGAN/SDGym source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    # Fetch its public OpenML copy by stable dataset ID.
    # The paper's benchmark excludes Time, leaving 29 continuous features.
    # https://www.openml.org/d/1597
    dataset = fetch_openml(data_id=1597, target_column='Class', as_frame=True)
    x = dataset.data
    y = pd.to_numeric(dataset.target, errors='raise').rename('Class').to_frame()
    expected_features = [*(f'V{i}' for i in range(1, 29)), 'Amount']
    source_columns = set(x.columns)
    if (source_columns not in (set(expected_features), set(expected_features) | {'Time'})
            or len(x.columns) != len(source_columns) or len(x) != 284_807):
        raise ValueError('OpenML creditcard dataset has an unexpected feature schema or row count')
    if y['Class'].value_counts().to_dict() != {0: 284_315, 1: 492}:
        raise ValueError('OpenML creditcard dataset has unexpected target labels')
    if verbose:
        print(dataset.details)
    return x.loc[:, expected_features], y


def load_mnist28(verbose=False):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    X, y = mnist['data'], mnist['target']
    X = pd.DataFrame(X)
    y = pd.DataFrame(y, columns=['label'])

    return X, y
