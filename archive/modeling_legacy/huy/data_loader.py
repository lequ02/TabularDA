import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import constants


class data_loader:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.paths = constants.IN_DATA_PATHS[dataset_name]
        self.target_name = self.paths.get(
            'target_name', {'adult': 'income', 'census': 'income', 'news': ' shares'}.get(dataset_name)
        )
        if self.target_name is None:
            raise ValueError(f"No target column is configured for {dataset_name!r}")
        self.train_columns = None
        self.scaler = None
        self.dev_data = None

    def load_train_augment_data(self, train_option, augment_option, validation=0.2):
        if train_option not in {'original', 'synthetic', 'mix'}:
            raise ValueError(f"Unsupported training option: {train_option}")
        if train_option != 'original' and augment_option not in {'ctgan', 'categorical', 'gaussian'}:
            raise ValueError(f"Unsupported synthetic option: {augment_option}")

        real = self._read(self.paths['train_original'])
        self._require_target(real, 'real training data')
        stratify = real[self.target_name] if self.dataset_name.lower() != 'news' else None
        real_train, dev_df = train_test_split(
            real, test_size=validation, random_state=42, stratify=stratify
        )
        self.train_columns = [column for column in real_train.columns if column != self.target_name]
        self.scaler = StandardScaler().fit(real_train[self.train_columns])

        if train_option == 'original':
            training = real_train.copy()
        else:
            synthetic_path = self.paths['synthetic'][augment_option]
            synthetic = self._read(synthetic_path)
            self._require_schema(synthetic, self.train_columns, 'synthetic data')
            training = (synthetic if train_option == 'synthetic'
                        else pd.concat([real_train, synthetic], ignore_index=True))

        train_data = self._to_loader(training, shuffle=True)
        self.dev_data = self._to_loader(dev_df, shuffle=False)
        return train_data, self.dev_data

    def load_test_data(self):
        test_df = self._read(self.paths['test'])
        return self._to_loader(test_df, shuffle=False)

    def _read(self, path):
        frame = pd.read_csv(path).drop(columns=['Unnamed: 0'], errors='ignore')
        if not frame.columns.is_unique:
            raise ValueError(f"Duplicate column names in {path}")
        return frame

    def _require_target(self, frame, source):
        if self.target_name not in frame.columns:
            raise ValueError(f"{source} is missing target column {self.target_name!r}")

    def _require_schema(self, frame, feature_columns, source):
        self._require_target(frame, source)
        actual = set(frame.columns) - {self.target_name}
        expected = set(feature_columns)
        if actual != expected:
            raise ValueError(
                f"{source} feature schema mismatch; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

    def _to_loader(self, frame, shuffle):
        if self.train_columns is None or self.scaler is None:
            raise RuntimeError("Load training data before dev or test data")
        self._require_schema(frame, self.train_columns, 'dataset')
        y = frame[self.target_name].to_numpy()
        X = frame[self.train_columns]
        X = self.scaler.transform(X)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float),
                               torch.tensor(y, dtype=torch.float))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
