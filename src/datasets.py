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
    return load_dataset(2, verbose)

def load_news(verbose=False):
    return load_dataset(332, verbose)

def load_census(verbose=False):
    return load_dataset(20, verbose)
