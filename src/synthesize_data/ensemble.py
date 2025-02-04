import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn

def sanitize_column_names(columns):
    """
    Sanitizes column names for XGBoost compatibility.
    XGBoost does not allow special characters ([, ] or <) in column names.
    """
    sanitized_columns = []
    
    for i, col in enumerate(columns):
        sanitized_col = col
        # Replace special characters with safe alternatives
        sanitized_col = sanitized_col.replace('<', '_lt_').replace('>', '_gt_').replace('[', '_lbracket_').replace(']', '_rbracket_')
        
        # If two columns end up with the same name, add a unique suffix to distinguish them
        if sanitized_col in sanitized_columns:
            sanitized_col = f"{sanitized_col}_{i}"  # Append a number to make it unique
        
        sanitized_columns.append(sanitized_col)
    
    return sanitized_columns

class Ensemble():
    def __init__(self, x_original, y_original, x_synthesized, target_name, target_synthesizer, filename, verbose=True, is_classification=True):
        self.x_original = x_original
        # self.y_original = y_original
        self.x_synthesized = x_synthesized
        self.target_name = target_name
        self.filename = filename
        self.ensemble_model = None
        self.verbose = verbose
        self.target_synthesizer = target_synthesizer
        self.is_classification = is_classification
        
        if self.is_classification:
            self.label_encoder, self.y_original = self.label_encode(y_original) # have to label encode the y_original or xgboost will throw error
        else:
            self.y_original = y_original
            self.label_encoder = None
        
        # Store original column names for later restoration
        self.original_x_columns = self.x_original.columns.tolist()
        self.original_synthesized_columns = self.x_synthesized.columns.tolist()

        # Sanitize column names for XGBoost
        self.x_original.columns = sanitize_column_names(self.x_original.columns)
        self.x_synthesized.columns = sanitize_column_names(self.x_synthesized.columns)
    
    def label_encode(self, y):
        le = sklearn.preprocessing.LabelEncoder()
        return le, le.fit_transform(y)
    
    def fit(self):
        model = self.get_ensemble_model()
        if self.verbose:
            print("Training ensemble model...")
        
        model.fit(self.x_original, self.y_original)
        
        # Predict on the synthesized data
        if self.is_classification:
            y_syn_pred = self.label_encoder.inverse_transform(model.predict(self.x_synthesized))  # inverse transform the label encoded y
        else:
            y_syn_pred = model.predict(self.x_synthesized)

        if self.verbose:
            print("Ensemble model training results:")
            y_hat_train = model.predict(self.x_original)
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
                return xgb.XGBClassifier()
            elif self.target_synthesizer == 'rf':
                return sklearn.ensemble.RandomForestClassifier()
        elif not self.is_classification:
            if self.target_synthesizer == 'xgb':
                return xgb.XGBRegressor()
            elif self.target_synthesizer == 'rf':
                return sklearn.ensemble.RandomForestRegressor()
        else:
            raise ValueError("Invalid target synthesizer. Must be one of ['xgb', 'rf']")
