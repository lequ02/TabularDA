import pandas as pd

def create_label_BN_BayesEstimator(xtrain, ytrain, xtest, target_name, BN_filename=None, filename=None):
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import BayesianNetwork
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    hc = HillClimbSearch(xtrain)
    best_model = hc.estimate(scoring_method=BicScore(xtrain))

    model = BayesianNetwork(best_model.edges())
    model.fit(xtrain, estimator=BayesianEstimator, prior_type="BDeu")

    # Save the model
    if not BN_filename:
        filename = f"{target_name}_BN_BayesEstimator_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

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

def create_label_BN_MLE(xtrain, ytrain, xtest, target_name, BN_filename=None, filename=None):
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import BayesianNetwork
    from pgmpy.inference import VariableElimination

    xtrain = xtrain.reindex(sorted(xtrain.columns), axis=1)
    xtest = xtest.reindex(sorted(xtest.columns), axis=1)

    hc = HillClimbSearch(xtrain)
    best_model = hc.estimate(scoring_method=BicScore(xtrain))

    model = BayesianNetwork(best_model.edges())
    model.fit(xtrain, estimator=MaximumLikelihoodEstimator)

    # Save the model
    if not BN_filename:
        filename = f"{target_name}_BN_MLE_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

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