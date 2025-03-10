import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test, handle_missing_values, check_directory, read_train_test_csv, onehot


class CreateSyntheticData:
    def __init__(self, ds_name, load_data_func, target_name, categorical_columns, features_synthesizer='CTGAN',
                 sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2, is_classification=True,
                 numerical_cols_pca_gmm=None):
        self.ds_name = ds_name
        self.load_data_func = load_data_func
        self.target_name = target_name
        self.categorical_columns = categorical_columns
        self.sample_size_to_synthesize = sample_size_to_synthesize
        self.missing_values_strategy = missing_values_strategy
        self.test_size = test_size
        self.is_classification = is_classification
        self.numerical_cols_pca_gmm = numerical_cols_pca_gmm
        self.features_synthesizer = features_synthesizer

        if features_synthesizer == 'CTGAN':
            fsyn = ''
        else:
            fsyn = features_synthesizer

        self.paths = {
            'synthesizer_dir': f'../sdv trained model/{ds_name}/',
            'data_dir': f'../data/{ds_name}/',
            'train_csv': f'{ds_name}_train.csv',
            'test_csv': f'{ds_name}_test.csv',
            'train_csv_onehot': f'onehot_{ds_name}_train.csv',
            'test_csv_onehot': f'onehot_{ds_name}_test.csv',

            'sdv_only_synthesizer': f'{ds_name}_synthesizer.pkl',
            'sdv_only_csv': f'onehot_{ds_name}_sdv_100k.csv',

            f'sdv_{fsyn}_gaussian_synthesizer': f'{ds_name}_{fsyn}_synthesizer_onlyX.pkl',
            f'sdv_{fsyn}_gaussian_csv': f'onehot_{ds_name}_sdv_{fsyn}_gaussian_100k.csv',

            f'sdv_{fsyn}_categorical_synthesizer': f'{ds_name}_{fsyn}_synthesizer_onlyX.pkl',
            f'sdv_{fsyn}_categorical_csv': f'onehot_{ds_name}_sdv_{fsyn}_categorical_100k.csv',

            f'sdv_{fsyn}_pca_gmm_synthesizer': f'{ds_name}_{fsyn}_synthesizer_onlyX.pkl',
            f'sdv_{fsyn}_pca_gmm_csv': f'onehot_{ds_name}_sdv_{fsyn}_pca_gmm_100k.csv',

            f'sdv_{fsyn}_xgb_synthesizer': f'{ds_name}_{fsyn}_synthesizer_onlyX.pkl',
            f'sdv_{fsyn}_xgb_csv': f'onehot_{ds_name}_sdv_{fsyn}_xgb_100k.csv',
            f'sdv_{fsyn}_rf_synthesizer': f'{ds_name}_{fsyn}_synthesizer_onlyX.pkl',
            f'sdv_{fsyn}_rf_csv': f'onehot_{ds_name}_sdv_{fsyn}_rf_100k.csv',

            'sdv_tvae_only_synthesizer': f'{ds_name}_TVAE_synthesizer.pkl',
            'sdv_tvae_only_csv': f'onehot_{ds_name}_sdv_tvae_100k.csv',

            

            ## don't need. file names for comparison is defined in synthesize_comparison_from_trained_model()
            # 'sdv_compare_synthesizer': f'{ds_name}_synthesizer.pkl',
            # 'sdv_compare_csv': f'onehot_{ds_name}_sdv_compare_100k.csv',
        }

    def create_synthetic_data(self):
        """
        wrapper function to create synthetic data, including: CTGAN (SDV), GaussianNB, CategoricalNB, PCA-GMM, Ensemble methods (XGBoost, RandomForest), and TVAE
        """
        if self.features_synthesizer == 'CTGAN':
            self.create_synthetic_data_sdv_only()
        elif self.features_synthesizer == 'TVAE':
            # this function does not do train-test split. 
            # Assumes the data is already split and saved to csv through running ctgan's create_synthetic_data_sdv_only()
            self.create_synthetic_data_tvae_only()

        self.create_synthetic_data_sdv_gaussian()
        self.create_synthetic_data_sdv_categorical()
        self.create_synthetic_data_pca_gmm()
        self.create_synthetic_data_ensemble()
        self.create_synthetic_data_tvae_only()

        self.create_comparison_from_trained_model()


    def create_synthetic_data_sdv_only(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.prepare_train_test()
        # need this line or the xy data will be double one-hot encoded. dont know why
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        xytrain = pd.concat([xtrain, ytrain], axis=1)
        self.synthesize_data(xytrain, ytrain, categorical_columns, 'sdv_only', '')

    def create_synthetic_data_sdv_gaussian(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        self.synthesize_data(xtrain, ytrain, categorical_columns, 'sdv_gaussian', 'gaussianNB', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_sdv_categorical(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, 'sdv_categorical', 'categoricalNB', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_pca_gmm(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, 'sdv_pca_gmm', 'pca_gmm', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_ensemble(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        emsemble_methods = ['xgb', 'rf']
        for method in emsemble_methods:
            self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, f'sdv_{method}', method, features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_tvae_only(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        xytrain = pd.concat([xtrain, ytrain], axis=1)
        self.synthesize_data(xytrain, ytrain, categorical_columns, 'sdv_tvae_only', '', features_synthesizer='TVAE')

    def create_comparison_from_trained_model(self):
        xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
        print(f"Creating comparison from trained model for '{self.ds_name}' with features synthesizer: {self.features_synthesizer}")
        target_synthesizers = ['gaussianNB', 'categoricalNB', 'pca_gmm', 'xgb', 'rf']

        if self.features_synthesizer == 'CTGAN':
            fsyn = ''
        else:
            fsyn = self.features_synthesizer

        for target_synthesizer in target_synthesizers:
            self.synthesize_comparison_from_trained_model(xtrain, ytrain, categorical_columns, f'sdv_{fsyn}compare_{target_synthesizer}', self.features_synthesizer, target_synthesizer)


    def prepare_train_test(self):
        """
        only for the first time, prepare data, train-test split data, handle missing values, and save to csv
        after that, read the data from the csv files
        """
        x_original, y_original = self.load_data_func()
        data = pd.concat([x_original, y_original], axis=1)
        # xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=self.target_name, test_size=test_size, random_state=42, stratify=y_original, categorical_columns=self.categorical_columns)
        # xtrain, ytrain = handle_missing_values.handle_missing_values(xtrain, ytrain, target_name=self.target_name, strategy=missing_values_strategy)
        # xtest, ytest = handle_missing_values.handle_missing_values(xtest, ytest, target_name=self.target_name, strategy=missing_values_strategy)
        xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot = self.test_split_and_handle_missing_onehot(data, self.missing_values_strategy, self.test_size)
        # save data to csv
        self.save_to_csv(xtrain, ytrain, xtest, ytest, self.paths['data_dir']+self.paths['train_csv'], self.paths['data_dir']+self.paths['test_csv'])
        # save onehot encoded data to csv
        # xtrain_onehot, xtest_onehot = onehot.onehot(xtrain, xtest, self.categorical_columns)
        self.save_to_csv(xtrain_onehot, ytrain, xtest_onehot, ytest, self.paths['data_dir']+self.paths['train_csv_onehot'], self.paths['data_dir']+self.paths['test_csv_onehot'])
        return xtrain, xtest, ytrain, ytest, self.target_name, self.categorical_columns

    def read_data(self):
        return read_train_test_csv.read_train_test_csv(self.paths['data_dir'] + self.paths['train_csv'], self.paths['data_dir'] + self.paths['test_csv'],
                                                       target_name=self.target_name, categorical_columns=self.categorical_columns)

    def synthesize_data(self, xtrain, ytrain, categorical_columns, synth_type, target_synthesizer, features_synthesizer='CTGAN'):
        # xytrain = pd.concat([xtrain, ytrain], axis=1)
        synthesize_data(xtrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                        features_synthesizer=features_synthesizer, numerical_columns_pca_gmm=self.numerical_cols_pca_gmm,
                        target_name=self.target_name, synthesizer_file_name=self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer'],
                        csv_file_name=self.paths['data_dir'] + self.paths[f'{synth_type}_csv'], verbose=True, is_classification=self.is_classification)
        
    def synthesize_from_trained_model(self, xtrain, ytrain, categorical_columns, synth_type, target_synthesizer):
        synthesize_from_trained_model(xtrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                                      numerical_columns_pca_gmm=self.numerical_cols_pca_gmm,
                                      target_name=self.target_name, synthesizer_file_name=self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer'],
                                      csv_file_name=self.paths['data_dir'] + self.paths[f'{synth_type}_csv'], verbose=True, is_classification=self.is_classification)
        

    def synthesize_comparison_from_trained_model(self, xtrain, ytrain, categorical_columns, synth_type, feature_synthesizer, target_synthesizer):
        # xytrain = pd.concat([xtrain, ytrain], axis=1)
        if feature_synthesizer == 'CTGAN':
            synthesizer_file_name = self.paths['synthesizer_dir'] + f'{self.ds_name}_synthesizer.pkl'
        elif feature_synthesizer == 'TVAE':
            synthesizer_file_name = self.paths['synthesizer_dir'] + f'{self.ds_name}_TVAE_synthesizer.pkl'
        else:
            raise ValueError(f"Unknown feature synthesizer: {feature_synthesizer}")
        csv_file_name = self.paths['data_dir'] + f'onehot_{self.ds_name}_{synth_type}_100k.csv'

        synthesize_comparison_from_trained_model(xtrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                        # features_synthesizer='CTGAN', # not used because loading synthesizer from file
                        numerical_columns_pca_gmm=self.numerical_cols_pca_gmm,
                        target_name=self.target_name, synthesizer_file_name=synthesizer_file_name,
                        csv_file_name=csv_file_name, verbose=True, is_classification=self.is_classification)

    def save_to_csv(self, xtrain, ytrain, xtest, ytest, train_csv, test_csv):
        train_set = pd.concat([xtrain, ytrain], axis=1)
        test_set = pd.concat([xtest, ytest], axis=1)
        check_directory.check_directory(train_csv)
        check_directory.check_directory(test_csv)
        train_set.to_csv(train_csv, index=False)
        test_set.to_csv(test_csv, index=False)
        print(f"Data saved to csv at {train_csv} and {test_csv}")

    def test_split_and_handle_missing_onehot(self, data, missing_values_strategy='drop', test_size=0.2):
        xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=self.target_name, test_size=test_size, random_state=42, 
                                                                           stratify=data[self.target_name], categorical_columns=self.categorical_columns)
        xtrain, ytrain = handle_missing_values.handle_missing_values(xtrain, ytrain, target_name=self.target_name, strategy=missing_values_strategy)
        xtest, ytest = handle_missing_values.handle_missing_values(xtest, ytest, target_name=self.target_name, strategy=missing_values_strategy)
        xtrain_onehot, xtest_onehot = onehot.onehot(xtrain, xtest, self.categorical_columns)
        return xtrain, xtest, ytrain, ytest, xtrain_onehot, xtest_onehot


