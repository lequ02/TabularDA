import sys
import os
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test, handle_missing_values, check_directory, read_train_test_csv, onehot


class CreateSyntheticData:
    def __init__(self, ds_name, load_data_func, target_name, categorical_columns, features_synthesizer='CTGAN',
                 sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2, is_classification=True,
                 numerical_cols_pca_gmm=None, output_root=None, seed=42):
        self.ds_name = ds_name
        self.load_data_func = load_data_func
        self.target_name = target_name
        self.categorical_columns = categorical_columns
        self.sample_size_to_synthesize = sample_size_to_synthesize
        self.missing_values_strategy = missing_values_strategy
        self.test_size = test_size
        self.is_classification = is_classification
        self.seed = seed
        self.numerical_cols_pca_gmm = numerical_cols_pca_gmm
        self.features_synthesizer = features_synthesizer.lower()

        if self.features_synthesizer == 'ctgan':
            self.fsyn_name = ''
        else:
            self.fsyn_name = self.features_synthesizer+"_" # e.g: 'TVAE' will be 'TVAE_'

        project_root = Path(__file__).resolve().parents[3]
        corrected_data_root = Path(output_root) if output_root is not None else project_root / 'data' / 'corrected_v2'
        corrected_model_root = project_root / 'sdv trained model' / 'corrected_v2'
        self.paths = {
            'synthesizer_dir': str(corrected_model_root / ds_name / f'seed_{seed}') + os.sep,
            'data_dir': str(corrected_data_root / ds_name / f'seed_{seed}') + os.sep,
            'train_csv': f'{ds_name}_train.csv',
            'test_csv': f'{ds_name}_test.csv',
            'train_csv_onehot': f'onehot_{ds_name}_train.csv',
            'test_csv_onehot': f'onehot_{ds_name}_test.csv',

            'sdv_only_synthesizer': f'{ds_name}_synthesizer.pkl',
            'sdv_only_csv': f'onehot_{ds_name}_sdv_100k.csv',

            f'sdv_{self.fsyn_name}gaussian_synthesizer': f'{ds_name}_{self.fsyn_name}synthesizer_onlyX.pkl',
            f'sdv_{self.fsyn_name}gaussian_csv': f'onehot_{ds_name}_sdv_{self.fsyn_name}gaussian_100k.csv',

            f'sdv_{self.fsyn_name}categorical_synthesizer': f'{ds_name}_{self.fsyn_name}synthesizer_onlyX.pkl',
            f'sdv_{self.fsyn_name}categorical_csv': f'onehot_{ds_name}_sdv_{self.fsyn_name}categorical_100k.csv',

            f'sdv_{self.fsyn_name}pca_gmm_synthesizer': f'{ds_name}_{self.fsyn_name}synthesizer_onlyX.pkl',
            f'sdv_{self.fsyn_name}pca_gmm_csv': f'onehot_{ds_name}_sdv_{self.fsyn_name}pca_gmm_100k.csv',

            f'sdv_{self.fsyn_name}xgb_synthesizer': f'{ds_name}_{self.fsyn_name}synthesizer_onlyX.pkl',
            f'sdv_{self.fsyn_name}xgb_csv': f'onehot_{ds_name}_sdv_{self.fsyn_name}xgb_100k.csv',
            f'sdv_{self.fsyn_name}rf_synthesizer': f'{ds_name}_{self.fsyn_name}synthesizer_onlyX.pkl',
            f'sdv_{self.fsyn_name}rf_csv': f'onehot_{ds_name}_sdv_{self.fsyn_name}rf_100k.csv',

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
        if self.features_synthesizer not in {'ctgan', 'tvae'}:
            raise ValueError("features_synthesizer must be 'ctgan' or 'tvae'")
        self.create_synthetic_data_sdv_only()

        if self.is_classification:
            self.create_synthetic_data_sdv_gaussian()
            self.create_synthetic_data_sdv_categorical()
        self.create_synthetic_data_pca_gmm()
        self.create_synthetic_data_ensemble()


    def create_synthetic_data_sdv_only(self):
        self.prepare_train_test()
        # need this line or the xy data will be double one-hot encoded. dont know why
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        xytrain = pd.concat([xtrain, ytrain], axis=1)
        synth_type = 'sdv_only' if self.features_synthesizer == 'ctgan' else 'sdv_tvae_only'
        self.synthesize_data(xytrain, ytrain, categorical_columns, synth_type, '', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_sdv_gaussian(self):
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        self.synthesize_data(xtrain, ytrain, categorical_columns, f'sdv_{self.fsyn_name}gaussian', 'gaussianNB', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_sdv_categorical(self):
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, f'sdv_{self.fsyn_name}categorical', 'categoricalNB', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_pca_gmm(self):
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        if self.is_classification:
            self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, f'sdv_{self.fsyn_name}pca_gmm', 'pca_gmm', features_synthesizer=self.features_synthesizer)
        else:
            self.synthesize_data(xtrain, ytrain, categorical_columns, f'sdv_{self.fsyn_name}pca_gmm', 'pca_gmm', features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_ensemble(self):
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        emsemble_methods = ['xgb', 'rf']
        for method in emsemble_methods:
            self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, f'sdv_{self.fsyn_name}{method}', method, features_synthesizer=self.features_synthesizer)

    def create_synthetic_data_tvae_only(self):
        self.prepare_train_test()
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        xytrain = pd.concat([xtrain, ytrain], axis=1)
        self.synthesize_data(xytrain, ytrain, categorical_columns, 'sdv_tvae_only', '', features_synthesizer='TVAE')

    def create_comparison_from_trained_model(self):
        xtrain, ytrain, target_name, categorical_columns = self.read_train_data()
        print(f"Creating comparison from trained model for '{self.ds_name}' with features synthesizer: {self.features_synthesizer}")
        target_synthesizers = ['pca_gmm', 'xgb', 'rf']
        if self.is_classification:
            target_synthesizers = ['gaussianNB', 'categoricalNB'] + target_synthesizers


        for target_synthesizer in target_synthesizers:
            self.synthesize_comparison_from_trained_model(xtrain, ytrain, categorical_columns, f'sdv_{self.fsyn_name}compare_{target_synthesizer}', self.features_synthesizer, target_synthesizer)


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
        splits = self.test_split_and_handle_missing_onehot(data, self.missing_values_strategy, self.test_size)
        xtrain, xdev, xtest, ytrain, ydev, ytest, xtrain_onehot, xdev_onehot, xtest_onehot = splits
        self.save_train_dev_test(xtrain, xdev, xtest, ytrain, ydev, ytest,
                                 xtrain_onehot, xdev_onehot, xtest_onehot)
        self.save_split_manifest()
        return xtrain, xdev, xtest, ytrain, ydev, ytest, self.target_name, self.categorical_columns

    def read_data(self):
        return read_train_test_csv.read_train_test_csv(self.paths['data_dir'] + self.paths['train_csv'], self.paths['data_dir'] + self.paths['test_csv'],
                                                       target_name=self.target_name, categorical_columns=self.categorical_columns)

    def read_train_data(self):
        train_data = pd.read_csv(self.paths['data_dir'] + self.paths['train_csv'])
        xtrain = train_data.drop(columns=[self.target_name])
        ytrain = train_data[self.target_name]
        return xtrain, ytrain, self.target_name, self.categorical_columns

    def synthesize_data(self, xtrain, ytrain, categorical_columns, synth_type, target_synthesizer, features_synthesizer='CTGAN'):
        # xytrain = pd.concat([xtrain, ytrain], axis=1)
        synthesize_data(xtrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                        features_synthesizer=features_synthesizer, numerical_columns_pca_gmm=self.numerical_cols_pca_gmm,
                        target_name=self.target_name, synthesizer_file_name=self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer'],
                        csv_file_name=self.paths['data_dir'] + self.paths[f'{synth_type}_csv'], verbose=True,
                        is_classification=self.is_classification, seed=self.seed)

    def synthesize_from_trained_model(self, xtrain, ytrain, categorical_columns, synth_type, target_synthesizer, features_synthesizer='CTGAN'):
        synthesize_from_trained_model(xtrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                                      numerical_columns_pca_gmm=self.numerical_cols_pca_gmm,
                                      target_name=self.target_name, synthesizer_file_name=self.paths['synthesizer_dir'] + self.paths[f'{synth_type}_synthesizer'],
                                      csv_file_name=self.paths['data_dir'] + self.paths[f'{synth_type}_csv'], verbose=True,
                                      is_classification=self.is_classification, seed=self.seed)


    def synthesize_comparison_from_trained_model(self, xtrain, ytrain, categorical_columns, synth_type, feature_synthesizer, target_synthesizer):
        # xytrain = pd.concat([xtrain, ytrain], axis=1)
        if feature_synthesizer.lower() == 'ctgan':
            synthesizer_file_name = self.paths['synthesizer_dir'] + f'{self.ds_name}_synthesizer_onlyX.pkl'
        elif feature_synthesizer.lower() == 'tvae':
            synthesizer_file_name = self.paths['synthesizer_dir'] + f'{self.ds_name}_tvae_synthesizer_onlyX.pkl'
        else:
            raise ValueError(f"Unknown feature synthesizer: {feature_synthesizer}")
        csv_file_name = self.paths['data_dir'] + f'onehot_{self.ds_name}_{synth_type}_100k.csv'

        synthesize_comparison_from_trained_model(xtrain, ytrain, categorical_columns, sample_size=self.sample_size_to_synthesize, target_synthesizer=target_synthesizer,
                        # features_synthesizer='CTGAN', # not used because loading synthesizer from file
                        numerical_columns_pca_gmm=self.numerical_cols_pca_gmm,
                        target_name=self.target_name, synthesizer_file_name=synthesizer_file_name,
                        csv_file_name=csv_file_name, verbose=True,
                        is_classification=self.is_classification, seed=self.seed)

    def save_to_csv(self, xtrain, ytrain, xtest, ytest, train_csv, test_csv):
        train_set = pd.concat([xtrain, ytrain], axis=1)
        test_set = pd.concat([xtest, ytest], axis=1)
        check_directory.check_directory(train_csv)
        check_directory.check_directory(test_csv)
        train_set.to_csv(train_csv, index=False)
        test_set.to_csv(test_csv, index=False)
        print(f"Data saved to csv at {train_csv} and {test_csv}")

    def save_train_dev_test(self, xtrain, xdev, xtest, ytrain, ydev, ytest,
                            xtrain_onehot, xdev_onehot, xtest_onehot):
        sets = [
            (xtrain, ytrain, self.paths['train_csv']),
            (xdev, ydev, f'{self.ds_name}_dev.csv'),
            (xtest, ytest, self.paths['test_csv']),
            (xtrain_onehot, ytrain, self.paths['train_csv_onehot']),
            (xdev_onehot, ydev, f'onehot_{self.ds_name}_dev.csv'),
            (xtest_onehot, ytest, self.paths['test_csv_onehot']),
        ]
        for x, y, filename in sets:
            path = self.paths['data_dir'] + filename
            save_set = pd.concat([x, y], axis=1)
            check_directory.check_directory(path)
            save_set.to_csv(path, index=False)

    def save_split_manifest(self):
        split_details = {}
        files = {
            'train': (self.paths['train_csv'], self.paths['train_csv_onehot']),
            'dev': (f'{self.ds_name}_dev.csv', f'onehot_{self.ds_name}_dev.csv'),
            'test': (self.paths['test_csv'], self.paths['test_csv_onehot']),
        }
        loaded = {}
        for split_name, (raw_name, onehot_name) in files.items():
            split_details[split_name] = {}
            for view, filename in [('raw', raw_name), ('onehot', onehot_name)]:
                path = self.paths['data_dir'] + filename
                frame = pd.read_csv(path)
                file_hash = hashlib.sha256()
                with open(path, 'rb') as source_file:
                    for block in iter(lambda: source_file.read(1024 * 1024), b''):
                        file_hash.update(block)
                loaded[(split_name, view)] = frame
                split_details[split_name][view] = {
                    'rows': len(frame),
                    'columns': [str(column) for column in frame.columns],
                    'target_counts': (
                        {str(label): int(count) for label, count in frame[self.target_name].value_counts(dropna=False).items()}
                        if self.is_classification else None
                    ),
                    'target_range': (
                        [float(frame[self.target_name].min()), float(frame[self.target_name].max())]
                        if not self.is_classification else None
                    ),
                    'sha256': file_hash.hexdigest(),
                }
        overlaps = {}
        for view in ('raw', 'onehot'):
            train = loaded[('train', view)]
            train_full = set(pd.util.hash_pandas_object(train, index=False).tolist())
            train_features = set(pd.util.hash_pandas_object(train.drop(columns=[self.target_name]), index=False).tolist())
            for split_name in ('dev', 'test'):
                candidate = loaded[(split_name, view)]
                candidate_full = pd.util.hash_pandas_object(candidate, index=False)
                candidate_features = pd.util.hash_pandas_object(candidate.drop(columns=[self.target_name]), index=False)
                overlap = {
                    'exact_full_rows': int(candidate_full.isin(train_full).sum()),
                    'exact_feature_rows': int(candidate_features.isin(train_features).sum()),
                }
                overlaps[f'train_vs_{split_name}_{view}'] = overlap
                if overlap['exact_feature_rows']:
                    raise ValueError(
                        f"Exact feature overlap between train and {split_name} in {view} data: "
                        f"{overlap['exact_feature_rows']} rows"
                    )
        manifest = {
            'dataset': self.ds_name,
            'seed': self.seed,
            'split_random_state': 42,
            'source_row_count': self._source_row_count,
            'splits': {name: ids.tolist() for name, ids in self._source_ids.items()},
            'files': split_details,
            'train_overlap': overlaps,
        }
        path = self.paths['data_dir'] + 'split_manifest.json'
        check_directory.check_directory(path)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(manifest, file, indent=2)

    def test_split_and_handle_missing_onehot(self, data, missing_values_strategy='drop', test_size=0.2):
        train_source, dev_source, test_source = self.split_source_data(data, test_size)
        return self.transform_source_splits(train_source, dev_source, test_source, missing_values_strategy)

    def split_source_data(self, data, test_size=None, groups=None):
        """Partition source rows before any feature preprocessing or generation."""
        data = data.reset_index(drop=True)
        self._source_row_count = len(data)
        row_ids = np.arange(len(data))
        if groups is None:
            groups = pd.util.hash_pandas_object(data.drop(columns=[self.target_name]), index=False).to_numpy()
        groups = np.asarray(groups)
        requested_test = self.test_size if test_size is None else test_size
        test_count = int(np.ceil(len(data) * requested_test)) if requested_test <= 1 else int(requested_test)
        dev_count = max(1, int(round((len(data) - test_count) * 0.1)))
        test_fraction = test_count / len(data)
        train_dev_ids, test_ids = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=42).split(row_ids, groups=groups))
        dev_fraction = dev_count / len(train_dev_ids)
        train_local, dev_local = next(GroupShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=42).split(
            train_dev_ids, groups=groups[train_dev_ids]))
        train_ids, dev_ids = train_dev_ids[train_local], train_dev_ids[dev_local]
        self._source_ids = {'train': train_ids, 'dev': dev_ids, 'test': test_ids}
        return tuple(data.iloc[ids].reset_index(drop=True) for ids in (train_ids, dev_ids, test_ids))

    def transform_source_splits(self, train_source, dev_source, test_source, missing_values_strategy=None):
        strategy = missing_values_strategy or self.missing_values_strategy
        source_frames = [train_source, dev_source, test_source]
        ids_by_split = [self._source_ids[name] for name in ('train', 'dev', 'test')]
        prepared = []
        for frame, ids in zip(source_frames, ids_by_split):
            y = frame[[self.target_name]].copy()
            x = frame.drop(columns=[self.target_name]).copy()
            if strategy == 'drop':
                keep = x.notna().all(axis=1).to_numpy() & y[self.target_name].notna().to_numpy()
                x, y, ids = x.loc[keep], y.loc[keep], ids[keep]
            else:
                keep = y[self.target_name].notna().to_numpy()
                x, y, ids = x.loc[keep], y.loc[keep], ids[keep]
            prepared.append((x, y, ids))
        xtrain, ytrain, train_ids = prepared[0]
        xdev, ydev, dev_ids = prepared[1]
        xtest, ytest, test_ids = prepared[2]
        xtrain, ytrain, fitted_imputer = handle_missing_values.handle_missing_values(
            xtrain, ytrain, target_name=self.target_name, strategy=strategy, return_imputer=True)
        xdev, ydev = handle_missing_values.handle_missing_values(
            xdev, ydev, target_name=self.target_name, strategy=strategy, fitted_imputer=fitted_imputer)
        xtest, ytest = handle_missing_values.handle_missing_values(
            xtest, ytest, target_name=self.target_name, strategy=strategy, fitted_imputer=fitted_imputer)
        xtrain_onehot, xdev_onehot = onehot.onehot(xtrain, xdev, self.categorical_columns)
        _, xtest_onehot = onehot.onehot(xtrain, xtest, self.categorical_columns)
        self._source_ids = {'train': train_ids, 'dev': dev_ids, 'test': test_ids}
        return xtrain, xdev, xtest, ytrain, ydev, ytest, xtrain_onehot, xdev_onehot, xtest_onehot


