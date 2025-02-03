import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

class Ensemble():
    def __init__(self, x_original, y_original, x_synthesized, target_name, target_synthesizer, filename, verbose=True):
        self.x_original = x_original
        self.y_original = y_original
        self.x_synthesized = x_synthesized
        self.target_name = target_name
        self.filename = filename
        self.ensemble_model = None
        self.verbose = verbose
        self.target_synthesizer = target_synthesizer
    
    # def xgboost(x_original, y_original, x_synthesized):
    #     xgb = xgb.XGBClassifier()
    #     xgb.fit(x_original, y_original)

    #     y_pred = xgb.predict(x_synthesized)
    #     return y_pred
    
    def fit(self, x_original, y_original, x_synthesized):
        model = self.ensemble_model()
        if self.verbose:
            print("Training ensemble model...")
        model.fit(x_original, y_original)
        y_syn_pred = model.predict(x_synthesized)

        if self.verbose:
            print("Ensemble model training results:")
            y_hat_train = model.predict(x_original)
            try: # try to calculate f1 score, if no error then the model is classification
                train_f1 = {}
                train_f1['weighted'] = sklearn.metrics.f1_score(y_original, y_hat_train, average='weighted')
                train_f1['macro'] = sklearn.metrics.f1_score(y_original, y_hat_train, average='macro')
                train_f1['micro'] = sklearn.metrics.f1_score(y_original, y_hat_train, average='micro')
                accuracy = sklearn.metrics.accuracy_score(y_original, y_hat_train)

                eval_metrics = {'accuracy': accuracy, 'f1': train_f1}

            except: # if error, then the model is regression
                mae = sklearn.metrics.mean_absolute_error(y_original, y_hat_train)
                mape = sklearn.metrics.mean_absolute_percentage_error(y_original, y_hat_train)
                r2 = sklearn.metrics.r2_score(y_original, y_hat_train)

                eval_metrics = {'mae': mae, 'mape': mape, 'r2': r2}

            print(eval_metrics)

        df_syn = pd.concat([x_synthesized, pd.DataFrame(y_syn_pred, columns=[self.target_name])], axis=1)
        df_syn.to_csv(self.filename, index=False)
        return eval_metrics, df_syn
    
    def get_ensemble_model(self):
        if self.target_synthesizer == 'xgboost':
            return xgb.XGBClassifier()
        elif self.target_synthesizer == 'rf':
            return sklearn.ensemble.RandomForestClassifier()
        # elif self.target_synthesizer == 'logistic_regression':
        #     return sklearn.linear_model.LogisticRegression()
        # elif self.target_synthesizer == 'decision_tree':
        #     return sklearn.tree.DecisionTreeClassifier()
        else:
            raise ValueError("Invalid target synthesizer")



