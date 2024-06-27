import pandas as pd
import pickle

def train_BN_BE(xtrain, ytrain, target_name, BN_filename=None):
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import BayesianNetwork

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    data = pd.concat([xtrain, ytrain], axis=1)

    hc = HillClimbSearch(xtrain)
    best_model = hc.estimate(scoring_method=BicScore(xtrain))

    model = BayesianNetwork(best_model.edges())
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")

    # Save the model
    if not BN_filename:
        BN_filename = f"{target_name}_BN_BE_model.pkl"
    with open(BN_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"BN model saved at {BN_filename}")
    return model

def train_BN_MLE(xtrain, ytrain, target_name, BN_filename=None):
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import BayesianNetwork

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    data = pd.concat([xtrain, ytrain], axis=1)

    hc = HillClimbSearch(xtrain)
    best_model = hc.estimate(scoring_method=BicScore(xtrain))

    model = BayesianNetwork(best_model.edges())
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Save the model
    if not BN_filename:
        BN_filename = f"{target_name}_BN_MLE_model.pkl"
    with open(BN_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"BN model saved at {BN_filename}")
    return model

def create_label_BN(xtrain, ytrain, xtest, target_name, BN_type, BN_filename=None, filename=None):
    """
    BN_filename: filename to save the trained BN model
    """
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    if BN_type == 'BE':
        model = train_BN_BE(xtrain, ytrain, target_name, BN_filename)
    elif BN_type == 'MLE':
        model = train_BN_MLE(xtrain, ytrain, target_name, BN_filename)

    infer = VariableElimination(model)
    predictions = []
    for _, row in xtest.iterrows():
        evidence = row.to_dict()
        query_result = infer.map_query(variables=[target_name], evidence=evidence)
        predictions.append(query_result[target_name])

    xtest[target_name] = predictions
    if filename:
        xtest.to_csv(filename)
    return xtest

def create_label_BN_from_trained(xtrain, ytrain, xtest, target_name, BN_model, filename=None):
    """
    BN_model: file name of the trained BN model
    """
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    with open(BN_model, 'rb') as f:
        model = pickle.load(f)

    print(xtrain.columns)
    print(xtest.columns)
    print(model.nodes)
    infer = VariableElimination(model)
    predictions = []
    # for _, row in xtest.iterrows():
    #     evidence = row.to_dict()
    #     query_result = infer.map_query(variables=[target_name], evidence=evidence)
    #     predictions.append(query_result[target_name])

    for _, row in xtest.iterrows():
        evidence = {var: val for var, val in row.to_dict().items() if var in model.nodes()}
        query_result = infer.map_query(variables=[target_name], evidence=evidence)
        predictions.append(query_result[target_name])

    xtest[target_name] = predictions
    if filename:
        xtest.to_csv(filename)
    return xtest