import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def discretize_simple(data, num_bins):
    """
    Function to discretize a continuous variable into bins.
    
    Parameters:
    data (pandas.Series): The continuous variable to be discretized.
    num_bins (int): The number of bins to split the data into.

    Returns:
    pandas.Series: The discretized variable.
    """
    discretized_data = pd.cut(data, bins=num_bins, labels=False)
    return discretized_data


def optimal_binning(xtrain_col, ytrain, xtest_col, max_leaf_nodes):
    """
    Function to discretize a continuous variable into optimal bins using decision trees.
    
    Parameters:
    xtrain_col (pandas.Series): The continuous variable to be discretized.
    ytrain (pandas.Series): The ytrain variable.
    max_leaf_nodes (int): The maximum number of leaf nodes for the decision tree.

    Returns:
    pandas.Series: The discretized variable.
    """
    # Reshape xtrain_col for sklearn
    xtrain_col = xtrain_col.values.reshape(-1, 1)
    ytrain = ytrain.values
    
    # Fit decision tree
    tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    tree.fit(xtrain_col, ytrain)
    
    # Create bins using decision tree boundaries
    bins = tree.tree_.threshold[tree.tree_.children_left != tree.tree_.children_right]
    bins = np.sort(bins)
    
    # Discretize column
    discretized_xtrain_col = np.digitize(xtrain_col, bins)
    discretized_xtest_col = np.digitize(xtest_col, bins)
    
    return discretized_xtrain_col, discretized_xtest_col



def select_features(df, target_name, corr_threshold=0.05, n_features_rfe=10):
    """
    Function to select features based on correlation and RFE:
    - Filter Method - Correlation Matrix
    - Wrapper Method - Recursive Feature Elimination (RFE)
    """
    print(f"Selecting features for target: {target_name}")
    print(f"Correlation threshold: {corr_threshold}")
    corr = df.corr()
    corr_target = abs(corr[target_name])
    relevant_features = corr_target[corr_target > corr_threshold].index.drop(target_name)
    
    print('Starting RFE...')
    X = df.drop(columns=[target_name])
    y = df[target_name]
    model = RandomForestRegressor()
    rfe = RFE(model, n_features_to_select=n_features_rfe)
    fit = rfe.fit(X, y)
    selected_features_rfe = X.columns[fit.support_]
    
    combined_features = list(set(relevant_features) | set(selected_features_rfe))
    combined_features.append(target_name)
    print(f"Selected features: {combined_features}")
    return combined_features

def train_BN_BE(xtrain, ytrain, target_name, BN_filename=None, verbose=False):
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import BayesianNetwork

    # Select features based on correlation and RFE
    # selected_features = select_features(pd.concat([xtrain, ytrain], axis=1), target_name)
    selected_features = [' shares', ' LDA_03', ' weekday_is_saturday', ' is_weekend']
    # xtrain = xtrain[selected_features]
    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    data = pd.concat([xtrain, ytrain], axis=1)
    data = data[selected_features]

    # discretized_cols = discretize_simple(data[' LDA_03'], num_bins=10)
    # print(discretized_cols)

    # discretized_cols = optimal_binning(data[' LDA_03'], data[target_name], max_leaf_nodes=10)

    # data[' LDA_03'] = discretized_cols
    print("data columns: ", data.columns)
    print(data.head())


    # structure learning
    print("Starting BN structure learning...")
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))
    print(type(best_model), best_model)
    print("best model nodes", best_model.nodes())
    print("best model edges", best_model.edges())
    # parameter learning
    print("Starting BN parameter learning...")
    model = BayesianNetwork(best_model)
    print(type(model), model)
    print("model nodes", model.nodes())
    print('model edges', model.edges())
    if verbose:
        pass
        # print("Nodes in BN: ", model.nodes)
        # nx.draw(best_model, with_labels=True)
        # plt.show()
        # plt.savefig('BN.png')
        # print('model structured saved as BN.png')
    print('fitting data to model')
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")

    # Save the model
    if not BN_filename:
        BN_filename = f"{target_name}_BN_BE_model.pkl"
    with open(BN_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"BN model saved at {BN_filename}")
    return model

def train_BN_MLE(xtrain, ytrain, target_name, BN_filename=None, verbose=False):
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import BayesianNetwork

    # Select features
    selected_features = select_features(pd.concat([xtrain, ytrain], axis=1), target_name)
    # xtrain = xtrain[selected_features]
    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    data = pd.concat([xtrain, ytrain], axis=1)
    data = data[selected_features]
    print("data columns: ", data.columns)

    # structure learning
    print("Starting BN structure learning...")
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))
    # parameter learning
    print("Starting BN parameter learning...")
    model = BayesianNetwork(best_model.edges())
    if verbose:
        print("Nodes in BN: ", model.nodes)
        nx.draw(model, with_labels=True)
        plt.show()
        plt.savefig('BN.png')
        print('model structured saved as BN.png')
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Save the model
    if not BN_filename:
        BN_filename = f"{target_name}_BN_MLE_model.pkl"
    with open(BN_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"BN model saved at {BN_filename}")
    return model

def create_label_BN(xtrain, ytrain, xtest, target_name, BN_type, continuous_cols=[], BN_filename=None, filename=None, verbose=False):
    """
    BN_filename: filename to save the trained BN model
    """
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    for col in continuous_cols:
        xtrain_discretized_cols, xtest_discretized_cols = optimal_binning(xtrain[col], ytrain, xtest[col], max_leaf_nodes=10)
        xtrain[col] = xtrain_discretized_cols
        xtest[col] = xtest_discretized_cols
    xtest = xtest[:1000]

    # print(xtrain.columns)
    # print(xtest.columns)
    if BN_type == 'BE':
        model = train_BN_BE(xtrain, ytrain, target_name, BN_filename, verbose=verbose)
    elif BN_type == 'MLE':
        model = train_BN_MLE(xtrain, ytrain, target_name, BN_filename, verbose=verbose)

    # if verbose:
    #     nx.draw(model, with_labels=True)
    #     plt.show()

    infer = VariableElimination(model)
    predictions = []

    # discretized_cols = discretize_simple(xtest[' LDA_03'], num_bins=10)

    # for _, row in xtest.iterrows():
    #     evidence = row.to_dict()
    #     query_result = infer.map_query(variables=[target_name], evidence=evidence)
    #     predictions.append(query_result[target_name])

    for _, row in xtest.iterrows():
        evidence = {var: val for var, val in row.to_dict().items() if var in model.nodes()}
        print(evidence)
        query_result = infer.map_query(variables=[target_name], evidence=evidence)
        predictions.append(query_result[target_name])
    xtest[target_name] = predictions
    
    if filename:
        xtest.to_csv(filename)
    return xtest

def create_label_BN_from_trained(xtrain, ytrain, xtest, target_name, BN_model, filename=None, verbose=False):
    """
    BN_model: file name of the trained BN model
    """
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    with open(BN_model, 'rb') as f:
        model = pickle.load(f)

    if verbose:
        print("drawing BN structure")
        nx.draw(model, with_labels=True)
        plt.show()
        plt.savefig('BN.png')
        print('model structured saved as BN.png')

    # print(xtrain.columns)
    # print(xtest.columns)
    print("model nodes: ",model.nodes())
    infer = VariableElimination(model)
    predictions = []
    # for _, row in xtest.iterrows():
    #     evidence = row.to_dict()
    #     query_result = infer.map_query(variables=[target_name], evidence=evidence)
    #     predictions.append(query_result[target_name])
    print("starting inference for rows")
    for _, row in xtest.iterrows():
        evidence = {var: val for var, val in row.to_dict().items() if var in model.nodes()}
        query_result = infer.map_query(variables=[target_name], evidence=evidence)
        predictions.append(query_result[target_name])

    xtest[target_name] = predictions
    if filename:
        xtest.to_csv(filename)
    return xtest