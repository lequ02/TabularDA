import re
import json
from numbers import Real

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import constants


def _natural_pixel_order(columns):
    """Order image columns by their numeric pixel index, not lexicographically."""
    def key(column):
        pieces = re.split(r"(\d+)", str(column))
        return tuple(int(piece) if piece.isdigit() else piece.lower() for piece in pieces)

    return sorted(columns, key=key)


def _take_exact(df, count, target_name, problem_type):
    if count < 0 or count > len(df):
        raise ValueError(f"Requested {count} rows from a source containing {len(df)} rows.")
    if count == len(df):
        return df.copy()
    if count == 0:
        return df.iloc[:0].copy()
    stratify = df[target_name] if problem_type == "classification" else None
    sampled, _ = train_test_split(
        df, train_size=count, random_state=42, stratify=stratify
    )
    return sampled


def _fit_label_encoder(train_df, dev_df, target_name, problem_type):
    if problem_type != "classification":
        return None, train_df, dev_df, None
    encoder = LabelEncoder().fit(train_df[target_name])
    num_classes = len(encoder.classes_)
    return (
        encoder,
        _encode_labels(encoder, train_df, target_name),
        _encode_labels(encoder, dev_df, target_name),
        num_classes,
    )


def _encode_labels(encoder, df, target_name):
    encoded = df.copy()
    labels = encoded[target_name]
    unknown = pd.unique(labels[~labels.isin(encoder.classes_)])
    if len(unknown):
        raise ValueError(f"Found labels absent from real training data: {unknown.tolist()}")
    encoded[target_name] = encoder.transform(labels)
    return encoded


def _label_counts(df, target_name):
    return {str(label): int(count) for label, count in df[target_name].value_counts(dropna=False).items()}


def _paths_for_seed(paths, seed):
    return {
        key: _paths_for_seed(value, seed) if isinstance(value, dict)
        else value.replace("{seed}", str(seed)) if isinstance(value, str)
        else value
        for key, value in paths.items()
    }


def _read_test_source_ids(paths, seed, dataset_name, row_count):
    with open(paths["split_manifest"], "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("seed") != seed or manifest.get("dataset") != dataset_name:
        raise ValueError("Corrected split manifest does not match the requested dataset and seed.")
    splits = manifest["splits"]
    ids = splits["test"]
    if len(ids) != row_count:
        raise ValueError(
            f"Test source ID count ({len(ids)}) does not match corrected test rows ({row_count})."
        )
    combined = splits["train"] + splits["dev"] + ids
    if len(combined) != len(set(combined)):
        raise ValueError("Corrected split manifest contains repeated source IDs across partitions.")
    return ids


class data_loader:
    def __init__(self, dataset_name, batch_size, multi_y=True, problem_type="classification", seed=42):
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        self.dataset_name = dataset_name
        self.seed = seed
        self.batch_size = batch_size
        self.paths = _paths_for_seed(constants.IN_DATA_PATHS[dataset_name], seed)
        self.manifest_path = self.paths["split_manifest"]
        self.synthetic_path = None
        self.synthetic_label_counts = None
        self.train_columns = None
        self.feature_columns = None
        self.multi_y = multi_y
        self.problem_type = problem_type
        self.scaler = None
        self.label_encoder = None
        self.num_classes = None
        self.test_source_ids = None

    @staticmethod
    def drop_index_col(df):
        return df.drop(columns=["Unnamed: 0"], errors="ignore")

    def _align_table(self, df):
        target = self.paths["target_name"]
        if target not in df.columns:
            raise ValueError(f"Input data is missing target column {target!r}.")
        features = set(df.columns) - {target}
        if features != set(self.feature_columns):
            raise ValueError("Input feature columns do not match corrected real training data.")
        return df.reindex(columns=self.feature_columns + [target])

    def load_train_augment_data(self, train_option, augment_option, mix_ratio=-1, n_sample=-1, validation=0.2):
        if train_option not in {"original", "synthetic", "mix"}:
            raise ValueError("train_option must be 'original', 'synthetic', or 'mix'.")
        target = self.paths["target_name"]
        real_df = self.drop_index_col(pd.read_csv(self.paths["train_original"]))
        if target not in real_df.columns:
            raise ValueError(f"Training data is missing target column {target!r}.")
        self.feature_columns = sorted(column for column in real_df.columns if column != target)
        real_df = real_df.reindex(columns=self.feature_columns + [target])

        # These corrected_v2 partitions were created before generator fitting.
        real_train = real_df
        dev_df = self.drop_index_col(pd.read_csv(self.paths["dev"]))
        real_train = self._align_table(real_train)
        dev_df = self._align_table(dev_df)
        self.label_encoder, real_train, dev_df, self.num_classes = _fit_label_encoder(
            real_train, dev_df, target, self.problem_type
        )
        self.scaler = StandardScaler().fit(real_train[self.feature_columns])
        self.train_columns = self.feature_columns + [target]

        if train_option == "original":
            train_df = real_train
        else:
            if not augment_option or augment_option not in self.paths.get("synthetic", {}):
                raise ValueError(f"Unsupported augment_option: {augment_option!r}")
            self.synthetic_path = self.paths["synthetic"][augment_option]
            synthetic_df = self.drop_index_col(
                pd.read_csv(self.synthetic_path)
            )
            synthetic_df = self._align_table(synthetic_df)
            self.synthetic_label_counts = (
                _label_counts(synthetic_df, target)
                if self.problem_type == "classification" else None
            )
            if self.label_encoder is not None:
                synthetic_df = _encode_labels(self.label_encoder, synthetic_df, target)
            if train_option == "synthetic":
                train_df = synthetic_df
            else:
                train_df = self.concat(
                    real_train, synthetic_df, concat_ratio=mix_ratio,
                    n_sample=n_sample,
                )

        return self._load_data_in_batches(train_df), self._load_data_in_batches(dev_df)

    def load_test_data(self):
        if self.train_columns is None or self.scaler is None:
            raise ValueError("Training data must be loaded before test data.")
        test_df = self.drop_index_col(pd.read_csv(self.paths["test"]))
        test_df = self._align_table(test_df)
        if self.label_encoder is not None:
            test_df = _encode_labels(self.label_encoder, test_df, self.paths["target_name"])
        self.test_source_ids = _read_test_source_ids(
            self.paths, self.seed, self.dataset_name, len(test_df)
        )
        return self._load_data_in_batches(test_df, shuffle=False)

    def _load_data_in_batches(self, df, shuffle=False):
        target = self.paths["target_name"]
        df = df.reindex(columns=self.feature_columns + [target])
        X = self.scaler.transform(df[self.feature_columns])
        y = df[target].to_numpy()
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_dtype = torch.float32 if not self.multi_y else torch.long
        y_tensor = torch.as_tensor(y, dtype=y_dtype)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=shuffle)

    def split_data(self, df, stratify_column=None, validation=0.2):
        if validation > 1:
            validation = int(validation)
            if validation <= 0 or validation >= len(df):
                raise ValueError("Validation row count must be between 1 and the dataset size minus 1.")
        elif not 0 < validation < 1:
            raise ValueError("Validation fraction must be greater than 0 and less than 1.")
        stratify = df[stratify_column] if stratify_column is not None else None
        return train_test_split(df, test_size=validation, stratify=stratify, random_state=42)

    def concat(self, df1, df2, axis=0, concat_ratio=-1, n_sample=-1):
        if n_sample == -1 or concat_ratio == -1:
            return pd.concat([df1, df2], axis=axis, join="inner", ignore_index=True)
        if not isinstance(n_sample, (int, np.integer)) or n_sample <= 0:
            raise ValueError("n_sample must be a positive integer or -1.")
        if isinstance(concat_ratio, Real) and 0 <= concat_ratio <= 1:
            df1_count = int(n_sample * concat_ratio)
        elif isinstance(concat_ratio, (int, np.integer)) and concat_ratio >= 0:
            df1_count = int(concat_ratio)
        else:
            raise ValueError("concat_ratio must be in [0, 1] or a nonnegative original-row count.")
        df2_count = n_sample - df1_count
        if df1_count > len(df1) or df2_count < 0 or df2_count > len(df2):
            raise ValueError("Requested mix counts exceed the available original or synthetic rows.")
        target = self.paths["target_name"]
        original = _take_exact(df1, df1_count, target, self.problem_type)
        synthetic = _take_exact(df2, df2_count, target, self.problem_type)
        return pd.concat([original, synthetic], axis=axis, join="inner", ignore_index=True)


class DataLoaderMNIST:
    def __init__(self, dataset_name, batch_size, multi_y=True, problem_type="classification", seed=42):
        if dataset_name.lower() not in {"mnist12", "mnist28"}:
            raise ValueError(f"Unsupported MNIST dataset: {dataset_name}")
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        self.dataset_name = dataset_name
        self.seed = seed
        self.batch_size = batch_size
        self.paths = _paths_for_seed(constants.IN_DATA_PATHS[dataset_name], seed)
        self.manifest_path = self.paths["split_manifest"]
        self.synthetic_path = None
        self.synthetic_label_counts = None
        self.train_columns = None
        self.multi_y = multi_y
        self.problem_type = problem_type
        self.expected_features = 144 if dataset_name.lower() == "mnist12" else 784
        self.label_encoder = None
        self.num_classes = None
        self.test_source_ids = None

    @staticmethod
    def drop_index_col(df):
        return df.drop(columns=["Unnamed: 0"], errors="ignore")

    def _ordered_columns(self, df):
        target = self.paths["target_name"]
        if target not in df.columns:
            raise ValueError(f"Input data is missing target column {target!r}.")
        features = [column for column in df.columns if column != target]
        features = _natural_pixel_order(features)
        if len(features) != self.expected_features:
            raise ValueError(
                f"Expected {self.expected_features} features for {self.dataset_name}, got {len(features)}."
            )
        return features

    def _align_image_table(self, df):
        target = self.paths["target_name"]
        if target not in df.columns:
            raise ValueError(f"Input data is missing target column {target!r}.")
        features = set(df.columns) - {target}
        if features != set(self.train_columns):
            raise ValueError("Input MNIST feature columns do not match corrected real training data.")
        return df.reindex(columns=self.train_columns + [target])

    def load_train_augment_data(self, train_option, augment_option, mix_ratio=-1, n_sample=-1, validation=0.2):
        if train_option not in {"original", "synthetic", "mix"}:
            raise ValueError("train_option must be 'original', 'synthetic', or 'mix'.")
        target = self.paths["target_name"]
        real_df = self.drop_index_col(pd.read_csv(self.paths["train_original"]))
        features = self._ordered_columns(real_df)
        real_df = real_df.reindex(columns=features + [target])
        stratify_column = target if self.problem_type == "classification" else None
        # Consume the explicit corrected_v2 dev partition; never split after fitting.
        real_train = real_df
        dev_df = self.drop_index_col(pd.read_csv(self.paths["dev"]))
        self.train_columns = features
        dev_features = self._ordered_columns(dev_df)
        if set(dev_features) != set(features):
            raise ValueError("Corrected MNIST train and dev feature columns do not match.")
        dev_df = dev_df.reindex(columns=features + [target])
        self.label_encoder, real_train, dev_df, self.num_classes = _fit_label_encoder(
            real_train, dev_df, target, self.problem_type
        )

        if train_option == "original":
            train_df = real_train
        else:
            if not augment_option or augment_option not in self.paths.get("synthetic", {}):
                raise ValueError(f"Unsupported augment_option: {augment_option!r}")
            self.synthetic_path = self.paths["synthetic"][augment_option]
            synthetic_df = self.drop_index_col(
                pd.read_csv(self.synthetic_path)
            )
            synthetic_df = self._align_image_table(synthetic_df)
            self.synthetic_label_counts = (
                _label_counts(synthetic_df, target)
                if self.problem_type == "classification" else None
            )
            if self.label_encoder is not None:
                synthetic_df = _encode_labels(self.label_encoder, synthetic_df, target)
            if train_option == "synthetic":
                train_df = synthetic_df
            else:
                train_df = self.concat(real_train, synthetic_df, mix_ratio, n_sample)

        return self._load_data_in_batches(train_df), self._load_data_in_batches(dev_df)

    def load_test_data(self):
        if self.train_columns is None:
            raise ValueError("Training data must be loaded before test data.")
        target = self.paths["target_name"]
        test_df = self.drop_index_col(pd.read_csv(self.paths["test"]))
        test_df = self._align_image_table(test_df)
        if self.label_encoder is not None:
            test_df = _encode_labels(self.label_encoder, test_df, target)
        self.test_source_ids = _read_test_source_ids(
            self.paths, self.seed, self.dataset_name, len(test_df)
        )
        return self._load_data_in_batches(test_df, shuffle=False)

    def _load_data_in_batches(self, df, shuffle=False):
        target = self.paths["target_name"]
        X = df.reindex(columns=self.train_columns).to_numpy()
        y = df[target].to_numpy()
        if X.shape[1] != self.expected_features:
            raise ValueError(f"Expected {self.expected_features} features, got {X.shape[1]}.")
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_dtype = torch.float32 if not self.multi_y else torch.long
        y_tensor = torch.as_tensor(y, dtype=y_dtype)
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=shuffle)

    def split_data(self, df, stratify_column=None, validation=0.2):
        return data_loader.split_data(self, df, stratify_column, validation)

    def concat(self, df1, df2, concat_ratio=-1, n_sample=-1):
        return data_loader.concat(self, df1, df2, concat_ratio=concat_ratio, n_sample=n_sample)
