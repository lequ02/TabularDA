import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from PIL import Image

def binning_simple(xtrain:pd.Series, xtest:pd.Series, num_bins:int):
    """
    Function to discretize a continuous variable into bins.
    
    Parameters:
    xtrain (pandas.Series): The continuous variable to be discretized.
    num_bins (int): The number of bins to split the data into.

    Returns:
    pandas.Series: The discretized variable.
    """
    discretized_xtrain = pd.cut(xtrain, bins=num_bins, labels=False)
    discretized_xtest = pd.cut(xtest, bins=num_bins, labels=False)
    return discretized_xtrain, discretized_xtest


def binning_optimal(xtrain_col:pd.Series, ytrain:pd.Series, xtest_col:pd.Series, max_leaf_nodes:int):
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


def select_features(df, target_name, corr_threshold=0.1, n_features_rfe=6):
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
    
    print(f'Starting RFE with {n_features_rfe} features...')
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
    selected_features = select_features(pd.concat([xtrain, ytrain], axis=1), target_name)
    # selected_features = [' global_rate_positive_words', ' kw_min_avg', ' rate_positive_words', ' min_negative_polarity', ' n_tokens_title', ' LDA_03', ' shares']
    # xtrain = xtrain[selected_features]
    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    data = pd.concat([xtrain, ytrain], axis=1)
    data = data[selected_features]

    # print("data columns: ", data.columns)
    # print(data.head())


    # structure learning
    print("Starting BN structure learning...")
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))

    # print(type(best_model), best_model)
    # print("best model nodes", best_model.nodes())
    # print("best model edges", best_model.edges())
    # parameter learning
    print("Starting BN parameter learning...")
    model = BayesianNetwork(best_model)

    # ensure that all feature nodes are connected to the target node
    for node in data:
        if node!=target_name:
            model.add_edge(node, target_name)

    # print(type(model), model)
    # print("model nodes", model.nodes())
    # print('model edges', model.edges())
    if verbose:
        # # pass
        # print("Nodes in BN: ", model.nodes)

        # G = nx.DiGraph()
        # G.add_nodes_from(model.nodes())
        # G.add_edges_from(model.edges())

        # nx.draw(G, with_labels=True)
        # # plt.show(block=False) # do NOT hold the execution of your program until you close the plot window.
        # plt.savefig('BN_BE.png')
        # print('model structured saved as BN_BE.png')
        # # Display image file
        # img = Image.open('BN_BE.png')
        # img.show()

        try: 
            draw_BN_graphviz(model, 'BN.png')
        except:
            print("Error in drawing BN with graphviz")
            print("Drawing with networkx instead")
            draw_BN_nx(model, 'BN.png')

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
    # selected_features = [' shares', ' LDA_03', ' weekday_is_saturday', ' is_weekend', ' LDA_02']

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    data = pd.concat([xtrain, ytrain], axis=1)
    data = data[selected_features]
    # print("data columns: ", data.columns)

    # structure learning
    print("Starting BN structure learning...")
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))

    # parameter learning
    print("Starting BN parameter learning...")
    model = BayesianNetwork(best_model)
    
    # ensure that all feature nodes are connected to the target node
    for node in data:
        if node!=target_name:
            model.add_edge(node, target_name)

    if verbose:
        # print("Nodes in BN: ", model.nodes)
        # G = nx.DiGraph()
        # G.add_nodes_from(model.nodes())
        # G.add_edges_from(model.edges())
        # nx.draw(G, with_labels=True)
        # # plt.show(block=False)
        # plt.savefig('BN_MLE.png')
        # # Display image file
        # img = Image.open('BN_MLE.png')
        # img.show()
        # print('model structured saved as BN_MLE.png')

        try: 
            draw_BN_graphviz(model, 'BN.png')
        except:
            print("Error in drawing BN with graphviz")
            print("Drawing with networkx instead")
            draw_BN_nx(model, 'BN.png')

            
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Save the model
    if not BN_filename:
        BN_filename = f"{target_name}_BN_MLE_model.pkl"
    with open(BN_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"BN model saved at {BN_filename}")
    return model

def create_label_BN(xtrain, ytrain, xtest, target_name, BN_type, BN_filename=None, filename=None, verbose=False):
    """
    BN_filename: filename to save the trained BN model
    """
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    for col in xtrain.columns:
        # all the categorical columns are already one-hot encoded
        # so the columns with more than 2 special values are continuous
        if len(set(xtrain[col].values))>2:
            print(f"Discretizing column {col}")
            xtrain[col], xtest[col] = binning_simple(xtrain[col], xtest[col], num_bins=10)

    # xtest = xtest[:1000]

    if BN_type == 'BE':
        model = train_BN_BE(xtrain, ytrain, target_name, BN_filename, verbose=verbose)
    elif BN_type == 'MLE':
        model = train_BN_MLE(xtrain, ytrain, target_name, BN_filename, verbose=verbose)


    infer = VariableElimination(model)
    predictions = []

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

    for col in xtrain.columns:
        # all the categorical columns are already one-hot encoded
        # so the columns with more than 2 special values are continuous
        if len(set(xtrain[col].values))>2:
            print(f"Discretizing column {col}")
            xtrain[col], xtest[col] = binning_simple(xtrain[col], xtest[col], num_bins=10)

    with open(BN_model, 'rb') as f:
        model = pickle.load(f)

    if verbose:
        # G = nx.DiGraph()
        # G.add_nodes_from(model.nodes())
        # G.add_edges_from(model.edges())

        # nx.draw(G, with_labels=True)
        # # plt.show(block=False)
        # plt.savefig('BN.png')
        # # Display image file
        # img = Image.open('BN.png')
        # img.show()
        # print('model structured saved as BN.png')
        try: 
            draw_BN_graphviz(model, 'BN.png')
        except:
            print("Error in drawing BN with graphviz")
            print("Drawing with networkx instead")
            draw_BN_nx(model, 'BN.png')

    # print(xtrain.columns)
    # print(xtest.columns)
    # print("model nodes: ",model.nodes())
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


def draw_BN_nx(model, filename=None):
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G)  # generates a layout that positions the nodes in a way that minimizes the number of crossing edges.

    plt.figure(figsize=(15,10))  # You may need to adjust the figure size depending on your graph

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)

    # Shift the labels so they do not overlap the nodes and stay in the plot
    pos_labels = {}
    for node, coords in pos.items():
        # Calculate a shift value based on the length of the label
        # shift_value = len(node) * 0.005  # Adjust the multiplier value based on your specific graph
        pos_labels[node] = (coords[0], coords[1] + 0.05)

    nx.draw_networkx_labels(G, pos_labels)

    plt.axis('off')
    plt.title('Network Graph')
    plt.tight_layout() # adjusts the margins so that all nodes & labels displayed in plot
    plt.savefig(filename)

    # Display image file
    img = Image.open(filename)
    img.show()
    print(f'Network structure saved as {filename}')

    # plt.show()

# import bnlearn as bn

# def draw_BN(model, filename=None):
#     # Plot the Bayesian Network and save it as a PNG file
#     G = bn.plot(model)

#     # Display the saved image file
#     if filename:
#         plt.savefig(filename)
#         img = Image.open(filename)
#         img.show()
#         print(f'Network structure saved as {filename}')



from graphviz import Digraph

def draw_BN_graphviz(model, filename=None):
    # Create a new directed graph
    dot = Digraph()

    # Add nodes and edges to the graph
    for node in model.nodes():
        dot.node(node)

    for edge in model.edges():
        dot.edge(*edge)

    # Save the graph to a file if a filename is provided
    if filename is not None:
        dot.render(filename, view=True)
        print(f'Network structure saved as {filename}')
    else:
        # Else, just display the graph
        dot.view()

    return dot
