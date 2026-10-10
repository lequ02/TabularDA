import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
import sklearn
import sklearn.ensemble
import sklearn.metrics

class Ensemble():
    def __init__(self, x_original, y_original, x_synthesized, target_name, target_synthesizer, filename, verbose=True, is_classification=True, artifact_path=None, random_state=None, target_transform='raw'):
        if target_transform not in {'raw', 'log'} or (is_classification and target_transform != 'raw'):
            raise ValueError('Log targets are available only for regression labelers')
        self.target_transform = target_transform
        # Model feature names must not change the caller's schema.
        self.x_original = x_original.copy()
        # self.y_original = y_original
        self.x_synthesized = x_synthesized.copy()
        self.target_name = target_name
        self.filename = filename
        self.ensemble_model = None
        self.verbose = verbose
        self.target_synthesizer = target_synthesizer
        self.is_classification = is_classification
        self.artifact_path = artifact_path
        self.random_state = random_state
        
        if self.is_classification:
            self.label_encoder, self.y_original = self.label_encode(y_original) # have to label encode the y_original or xgboost will throw error
        else:
            self.y_original = y_original
            self.label_encoder = None
        
        # Store original column names for later restoration
        self.original_x_columns = self.x_original.columns.tolist()
        self.original_synthesized_columns = self.x_synthesized.columns.tolist()

        # XGBoost requires safe, unique feature names; keep the source schema for output.
        model_columns = [f'feature_{i}' for i in range(len(self.original_x_columns))]
        self.x_original.columns = model_columns
        self.x_synthesized.columns = model_columns
    
    def label_encode(self, y):
        le = sklearn.preprocessing.LabelEncoder()
        return le, le.fit_transform(y)
    
    def fit(self):
        if self.target_transform == 'log':
            from commons.log_target import log_target, inverse_log_target
        model = self.get_ensemble_model()
        if self.verbose:
            print("Training ensemble model...")
        
        model.fit(self.x_original, log_target(self.y_original) if self.target_transform == 'log' else self.y_original)
        if self.artifact_path:
            Path(self.artifact_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.artifact_path, "wb") as artifact_file:
                pickle.dump({"estimator": model, "label_encoder": self.label_encoder,
                             "feature_columns": self.original_x_columns,
                             "sanitized_feature_columns": list(self.x_original.columns),
                             "is_classification": self.is_classification,
                             "target_transform": self.target_transform}, artifact_file)
        
        # Predict on the synthesized data
        if self.is_classification:
            y_syn_pred = self.label_encoder.inverse_transform(model.predict(self.x_synthesized))  # inverse transform the label encoded y
        else:
            y_syn_pred = model.predict(self.x_synthesized)

        y_hat_train = model.predict(self.x_original)
        if self.target_transform == 'log':
            y_syn_pred = inverse_log_target(y_syn_pred)
            y_hat_train = inverse_log_target(y_hat_train)
        if self.is_classification:
            train_f1 = {}
            train_f1['weighted'] = sklearn.metrics.f1_score(self.y_original, y_hat_train, average='weighted')
            train_f1['macro'] = sklearn.metrics.f1_score(self.y_original, y_hat_train, average='macro')
            train_f1['micro'] = sklearn.metrics.f1_score(self.y_original, y_hat_train, average='micro')
            accuracy = sklearn.metrics.accuracy_score(self.y_original, y_hat_train)
            eval_metrics = {'accuracy': accuracy, 'f1': train_f1}
        else:  # regression
            mae = sklearn.metrics.mean_absolute_error(self.y_original, y_hat_train)
            mape = sklearn.metrics.mean_absolute_percentage_error(self.y_original, y_hat_train)
            r2 = sklearn.metrics.r2_score(self.y_original, y_hat_train)
            eval_metrics = {'mae': mae, 'mape': mape, 'r2': r2}
        if self.verbose:
            print("Ensemble model training results:")
            print(eval_metrics)

        # Combine synthesized data with the predictions
        df_syn = pd.concat([self.x_synthesized, pd.DataFrame(y_syn_pred, columns=[self.target_name])], axis=1)
        
        # Restore the original column names in the synthesized dataframe
        df_syn.columns = self.original_synthesized_columns + [self.target_name]
        
        # Save the result to a CSV file
        df_syn.to_csv(self.filename, index=False)
        
        return eval_metrics, df_syn
    
    def get_ensemble_model(self):
        if self.is_classification:
            if self.target_synthesizer == 'xgb':
                return xgb.XGBClassifier(random_state=self.random_state)
            elif self.target_synthesizer == 'rf':
                return sklearn.ensemble.RandomForestClassifier(random_state=self.random_state)
        elif not self.is_classification:
            if self.target_synthesizer == 'xgb':
                return xgb.XGBRegressor(random_state=self.random_state)
            elif self.target_synthesizer == 'rf':
                return sklearn.ensemble.RandomForestRegressor(random_state=self.random_state)
        raise ValueError("Invalid target synthesizer. Must be one of ['xgb', 'rf']")
