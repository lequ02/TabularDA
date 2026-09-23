import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import constants


class data_loader:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.paths = constants.IN_DATA_PATHS[dataset_name]
        self.target_name = self.paths['target_name'].strip()
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
        if self.dataset_name == 'covertype':
            real_labels = set(pd.to_numeric(real[self.target_name], errors='raise').tolist())
            if real_labels != set(range(1, 8)):
                raise ValueError(f"Covertype real labels must be exactly 1..7; found {sorted(real_labels)}")
        is_regression = self.dataset_name == 'news'
        stratify = None if is_regression else real[self.target_name]
        real_train, dev_df = train_test_split(
            real, test_size=validation, random_state=42, stratify=stratify
        )
        self.train_columns = [column for column in real_train.columns if column != self.target_name]
        self.scaler = StandardScaler().fit(real_train[self.train_columns])

        if train_option == 'original':
            training = real_train.copy()
        else:
            synthetic = self._read(self.paths['synthetic'][augment_option])
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
        frame.columns = [str(column).strip() for column in frame.columns]
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

    def _target_values(self, labels):
        values = pd.to_numeric(labels, errors='raise').to_numpy()
        if self.dataset_name == 'covertype':
            classes = set(values.tolist())
            if not classes.issubset(set(range(1, 8))):
                raise ValueError(f"Covertype labels must be in 1..7; found {sorted(classes)}")
            values = values - 1
        return values, self.dataset_name in {'news', 'adult', 'census', 'credit'}

    def _to_loader(self, frame, shuffle):
        if self.train_columns is None or self.scaler is None:
            raise RuntimeError("Load training data before dev or test data")
        self._require_schema(frame, self.train_columns, 'dataset')
        labels, use_float = self._target_values(frame[self.target_name])
        X = self.scaler.transform(frame[self.train_columns])
        label_dtype = torch.float if use_float else torch.long
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float),
                               torch.tensor(labels, dtype=label_dtype))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
