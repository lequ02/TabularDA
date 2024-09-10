import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from synthesize_data.onehot import onehot
from datasets import load_adult, load_news, load_census
from sklearn.preprocessing import StandardScaler

class data_loader:
    def __init__(self, file_name, dataset_name, train_option, test_option,
                 test_ratio, batch_size, numerical_columns=[]):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.numerical_columns = numerical_columns
        self.file_name = file_name
        self.train_option = train_option
        self.test_option = test_option
        self.test_ratio = test_ratio

        # Load and print shapes before batching
        train_data, test_data = self.load_datasets(self.train_option)
        print(f"Total train samples: {train_data.shape[0]}")
        print(f"Total test samples: {test_data.shape[0]}")

        self.test_data = self.load_data_in_batches(test_data, is_test=True)
        self.train_data = self.load_data_in_batches(train_data)

    def load_data_in_batches(self, ds, is_test=False):
        ds = self.standardize(ds, self.numerical_columns)
        x = ds.iloc[:, :-1]  # all columns except the last one
        y = ds.iloc[:, -1]  # the last column
        
        if is_test:
            ds_sampled = ds.sample(n=self.test_ratio, random_state=42).reset_index(drop=True)
        else:
            ds_sampled = ds
        
        return self.distribute_in_batches(ds_sampled.iloc[:, :-1], ds_sampled.iloc[:, -1])
    
    def distribute_in_batches(self, X, y):
        num_batch = int(len(X) / self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = torch.tensor(X.iloc[start:end].values, dtype=torch.float)
            batch_y = torch.tensor(y.iloc[start:end].values, dtype=torch.float)  # Assuming y is numerical for regression

            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)

        return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=self.batch_size)

    def train_test_split(self, ds, test_size, random_seed=42):
        random.seed(random_seed)
        ds = ds.sample(frac=1, random_state=random_seed)
        if test_size < 1:
            test_size = int(len(ds) * test_size)
        ds_train = ds.iloc[test_size:]
        ds_test = ds.iloc[:test_size]
        return ds_train, ds_test

    def load_datasets(self, option):
        if option == 'original':
            x, y = self.load_clean_ori_data()
            ds_ori = pd.concat([x, y], axis=1)
            ds_train, ds_test = self.train_test_split(ds_ori, self.test_ratio)
            print(f"Original dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
            return ds_train, ds_test

        elif option == 'synthetic':
            ds_synth = pd.read_csv(self.file_name, index_col=0)
            ds_train, ds_test = self.train_test_split(ds_synth, self.test_ratio)
            print(f"Synthetic dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
            return ds_train, ds_test

        elif option == 'mix':
            x_ori, y_ori = self.load_clean_ori_data()
            ds_ori = pd.concat([x_ori, y_ori], axis=1)
            ds_synth = pd.read_csv(self.file_name, index_col=0)
            ds_concat = pd.concat([ds_ori, ds_synth], axis=0)
            if self.dataset_name != 'news':
                ds_balanced = self.balance_classes(ds_concat.iloc[:, :-1], ds_concat.iloc[:, -1])
                # If the dataset is census, take only 100k data points
                if self.dataset_name == 'census':
                    ds_balanced = ds_balanced.sample(n=100000, random_state=42).reset_index(drop=True)
                
                ds_train, ds_test = self.train_test_split(ds_balanced, self.test_ratio)
            else:
                ds_train, ds_test = self.train_test_split(ds_concat, self.test_ratio)
            
            print(f"Mixed dataset - Train shape: {ds_train.shape}, Test shape: {ds_test.shape}")
            return ds_train, ds_test
        
    def balance_classes(self, X, y):
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        majority_count = int(class_counts[minority_class] * (76 / 24))
        minority_count = class_counts[minority_class]

        # Ensure majority_count does not exceed the available samples
        available_majority_count = class_counts[majority_class]
        if majority_count > available_majority_count:
            majority_count = available_majority_count

        majority_samples = X[y == majority_class].sample(majority_count, random_state=42).reset_index(drop=True)
        minority_samples = X[y == minority_class].reset_index(drop=True)

        X_balanced = pd.concat([majority_samples, minority_samples]).reset_index(drop=True)
        y_balanced = pd.concat([pd.Series([majority_class] * majority_count), pd.Series([minority_class] * minority_count)]).reset_index(drop=True)

        balanced_data = pd.concat([X_balanced, y_balanced], axis=1)
        return balanced_data

    def load_clean_ori_data(self):
        if self.dataset_name == 'adult':
            x, y = load_adult()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            x_onehot, _ = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
                                               'relationship', 'race', 'sex', 'native-country'], verbose=False)
            x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
            print("Adult dataset columns:", x_onehot.columns)
            print("Sample labels:", y['income'].unique())  # Verify labels
            return x_onehot, y

        elif self.dataset_name == 'census':
            x, y = load_census()
            y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})
            x_onehot, _ = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
                                               'relationship', 'race', 'sex', 'native-country'], verbose=False)
            x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
            print("Census dataset columns:", x_onehot.columns)
            print("Sample labels:", y['income'].unique())  # Verify labels
            return x_onehot, y

        elif self.dataset_name == 'news':
            x, y = load_news()
            # Assuming y is a numerical column in regression
            print("News dataset columns:", x.columns)
            print("Sample labels summary:", y.describe())  # Verify numerical values
            print("shape of the columns:", x.shape)
            return x, y
        

    def load_test_data(self):
        _, ds_test = self.load_datasets(self.test_option)
        return ds_test

    def load_train_data(self):
        if self.train_option == self.test_option:
            ds_train, _ = self.load_datasets(self.train_option)
        else:
            ds1, ds2 = self.load_datasets(self.train_option)
            ds_train = pd.concat([ds1, ds2], axis=0)
        return ds_train

    def standardize(self, df, numerical_cols):
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df
    
    def print_sample_data(self):
        print("\nFirst 5 samples from training set:")
        for i, (inputs, labels) in enumerate(self.train_data):
            if i < 5:
                print(f"Sample {i+1}:")
                print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
                print(f"Label: {labels[0].item()}")
            else:
                break

        print("\nFirst 5 samples from test set:")
        for i, (inputs, labels) in enumerate(self.test_data):
            if i < 5:
                print(f"Sample {i+1}:")
                print(f"Input shape: {inputs.shape}, Input data: {inputs.numpy()}")
                print(f"Label: {labels[0].item()}")
            else:
                break
