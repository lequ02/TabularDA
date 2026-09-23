import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets import load_adult, load_news, load_census


class data_loader:
    def __init__(self, file_name, dataset_name, train_option, test_option,
                 test_ratio, batch_size, numerical_columns=None):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.numerical_columns = list(numerical_columns or [])
        self.file_name = file_name
        self.train_option = train_option
        self.test_option = test_option
        if test_option != 'original':
            raise ValueError("Final evaluation requires the original real test set")
        self.test_ratio = test_ratio
        self.target_name = {'adult': 'income', 'census': 'income', 'news': 'shares'}.get(dataset_name)
        if self.target_name is None:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.scaler = None
        self.feature_columns = None

        real_source = self._load_option('original')
        real_train, real_test = train_test_split(real_source, test_size=0.2, random_state=42)
        real_train, dev_source = train_test_split(real_train, test_size=0.2, random_state=42)
        if train_option == 'original':
            train_source = real_train
        elif train_option == 'synthetic':
            train_source = self._load_option('synthetic')
        elif train_option == 'mix':
            real_features = pd.get_dummies(real_train.drop(columns=[self.target_name]), dtype=float)
            real_encoded = pd.concat([real_features, real_train[[self.target_name]]], axis=1)
            synthetic = self._load_option('synthetic')
            if set(real_encoded.columns) != set(synthetic.columns):
                raise ValueError("Mixed real and synthetic feature columns do not match")
            train_source = pd.concat([real_encoded, synthetic[real_encoded.columns]], ignore_index=True)
        else:
            raise ValueError(f"Unsupported data option: {train_option}")
        self.real_test_source = real_test
        self._to_loader(real_train, fit=True, shuffle=False)
        self.train_data = self._to_loader(train_source, fit=False, shuffle=True)
        self.dev_data = self._to_loader(dev_source, fit=False, shuffle=False)
        self.test_data = None

    def _load_option(self, option):
        if option == 'original':
            x, y = self.load_clean_ori_data()
            return pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        if option == 'synthetic':
            return pd.read_csv(self.file_name)
        if option == 'mix':
            original = self._load_option('original')
            synthetic = self._load_option('synthetic')
            return pd.concat([original, synthetic], ignore_index=True)
        raise ValueError(f"Unsupported data option: {option}")

    def _to_loader(self, frame, fit, shuffle):
        if self.target_name not in frame.columns:
            raise KeyError(f"Target column {self.target_name!r} is missing")
        y = frame[self.target_name].copy()
        X = pd.get_dummies(frame.drop(columns=[self.target_name]), dtype=float)
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        numeric = [c for c in self.numerical_columns if c in X.columns]
        if fit:
            self.scaler = StandardScaler().fit(X[numeric]) if numeric else None
        if numeric:
            if self.scaler is None:
                raise RuntimeError("Training data must be loaded before holdout data")
            X.loc[:, numeric] = self.scaler.transform(X[numeric])
        dataset = TensorDataset(torch.tensor(X.to_numpy(), dtype=torch.float32),
                                torch.tensor(y.to_numpy(), dtype=torch.float32))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def load_clean_ori_data(self):
        if self.dataset_name == 'adult':
            x, y = load_adult()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            return x, y
        if self.dataset_name == 'census':
            x, y = load_census()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            return x, y
        if self.dataset_name == 'news':
            return load_news()
        raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def load_test_data(self):
        test_source = self.real_test_source
        if isinstance(self.test_ratio, int) and 0 < self.test_ratio < len(test_source):
            test_source = test_source.sample(n=self.test_ratio, random_state=42)
        self.test_data = self._to_loader(test_source, fit=False, shuffle=False)
        return self.test_data

    def load_train_data(self):
        return self.train_data

    def print_sample_data(self):
        for name, loader in [('training', self.train_data), ('test', self.test_data)]:
            print(f"First batches from {name} data:")
            for index, (inputs, labels) in enumerate(loader):
                if index == 5:
                    break
                print(f"Batch {index + 1}: input shape {inputs.shape}, label shape {labels.shape}")
