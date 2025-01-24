import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Constants
import constants

class data_loader:
    def __init__(self, dataset_name, batch_size, multi_y = True, problem_type = 'classification'):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.paths = constants.IN_DATA_PATHS[dataset_name]
        self.train_columns = None  # Store train columns for alignment
        self.multi_y = multi_y
        self.problem_type = problem_type

    def drop_index_col(self, df):
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
        return df
    
    def load_train_augment_data(self, train_option, augment_option, mix_ratio = -1, n_sample = -1, validation = 0):
        if train_option not in ['original', 'synthetic', 'mix'] or augment_option not in [None, 'ctgan', 'categorical', 'gaussian']:
            raise ValueError("Invalid train_option or augment_option")
        if train_option == 'original':
            train_df = pd.read_csv(self.paths['train_original'])
            train_df = self.drop_index_col(train_df)
        elif train_option == 'synthetic':
            train_df = pd.read_csv(self.paths[train_option][augment_option])
            train_df = train_df[sorted(train_df.columns)]
            train_df = self.drop_index_col(train_df)
        else:  # 'mix' option
            original_df = pd.read_csv(self.paths['train_original'])
            original_df = self.drop_index_col(original_df)
            # sort original and synthetic alphabetically so the columns are in the same order
            original_df = original_df.reindex(sorted(original_df.columns), axis=1)

            print("\n\noriginal columns")
            print(original_df.columns)

            # print(self.paths)
            # print(self.paths['synthetic'])
            # print(self.paths['synthetic'][augment_option])

            synthetic_df = pd.read_csv(self.paths['synthetic'][augment_option])
            synthetic_df = self.drop_index_col(synthetic_df)
            # sort original and synthetic alphabetically so the columns are in the same order
            synthetic_df = synthetic_df.reindex(sorted(synthetic_df.columns), axis=1)

            print("\n\nsynthetic columns")
            print(synthetic_df.columns)


            train_df = self.concat(original_df, synthetic_df, axis = 0, concat_ratio = mix_ratio, n_sample = n_sample) #pd.concat([original_df, synthetic_df], axis=0)
        
        # sort train and test alphabetically so the columns are in the same order
        train_df = train_df.reindex(sorted(train_df.columns), axis=1)

        # train_df.to_csv("dummy_train_df.csv", index=False)

        print("\n\ntrain columns")
        print(train_df.columns)


        # print(f"Train dataset columns:\n{train_df.columns}")
        self.train_columns = train_df.columns  # Store columns for test data alignment
        
        if self.problem_type == 'classification':
            stratify_column = self.paths['target_name']
        elif self.problem_type == 'regression':
            stratify_column = None
        else:
            ValueError(f"This {self.problem_type} is not supported!")


        train_df, dev_df = self.split_data(train_df, stratify_column, validation)
        return self._load_data_in_batches(train_df), self._load_data_in_batches(dev_df)
        
    def load_test_data(self):
        # Load test data
        # print("\n\n path", self.paths['test'])
        test_df = pd.read_csv(self.paths['test'])
        
        # print(f"Test dataset preview:\n{test_df.head()}")  # Debugging
        # print(test_df.head(2))


        if self.train_columns is None:
            raise ValueError("Training data must be loaded before test data to align columns.")

        # Align test columns with training columns
        missing_cols = set(self.train_columns) - set(test_df.columns)
        print(missing_cols)
        for col in missing_cols:
            test_df[col] = 0  

        # sort train and test alphabetically so the columns are in the same order
        test_df = test_df.reindex(sorted(test_df.columns), axis=1)
        
        # Drop extra columns
        extra_cols = set(test_df.columns) - set(self.train_columns)
        
        if extra_cols:
            test_df = test_df.drop(columns=extra_cols)

        # print(f"Test dataset columns after alignment:\n{test_df.columns}")
        # print(f"Test dataset shape after alignment: {test_df.shape}")
            
        # test_df = test_df[sorted(test_df.columns)]
        print("\n\ntest columns")
        print(test_df.columns)

        # test_df.to_csv("dummy_test_df.csv", index=False)
        

        y = test_df[self.paths['target_name']]
        X = test_df.drop(self.paths['target_name'], axis=1)


        # Standardize features
        X = self._standardize(X)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float)
        
        if (self.multi_y != True):
                y_tensor = torch.tensor(y.values, dtype=torch.float)
        else:
                y_tensor = torch.tensor(y.values, dtype=torch.long)
        
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def _load_data_in_batches(self, df):
        

        y = df[self.paths['target_name']]
        X = df.drop(self.paths['target_name'], axis=1)
        X = self._standardize(X)
        return self._distribute_in_batches(X.values, y.values)
        
    def _standardize(self, df):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
    def _distribute_in_batches(self, X, y):
        print("Type X", type(X))
        print("Type y", type(y))
        print("Shape 0 X", len(X))
        print("Shape 1 X", len(X[0]))
        
        num_batch = int(len(X) / self.batch_size)
        batches = []

        for i in range(num_batch):
            start = i * self.batch_size
            end = start + self.batch_size

            batch_X = torch.tensor(X[start:end], dtype=torch.float)
            
            if (self.multi_y != True):
                batch_y = torch.tensor(y[start:end], dtype=torch.float)
            else:
                batch_y = torch.tensor(y[start:end], dtype=torch.long)
            
            batch = TensorDataset(batch_X, batch_y)
            batches.append(batch)
        
        return DataLoader(ConcatDataset(batches), batch_size=self.batch_size)

    def split_data(self, df, stratify_column = None, validation = 0):
        
        if validation > 1:
            validation = int(validation)
            
        if stratify_column != None:
            return train_test_split(df, test_size = validation, 
            stratify = df[stratify_column], random_state=42)
        else:
            return train_test_split(df, test_size = validation, random_state=42)

    def concat(self, df1, df2, axis = 0, concat_ratio = -1, n_sample = -1):
        #concat_ratio: percentage of df1 => 1-concat_ratio  = percentage of df2
        #df1 is for original data and df2 is for augmented data
        
        if (n_sample == -1) or (concat_ratio == -1): #take all original samples and all augmented samples
            return pd.concat([df1, df2], axis = 0, join = 'inner')
        
        df1_sample, df2_sample = df1.shape[0], df2.shape[0]

        if n_sample <= 0:
            ValueError("Number of sample must be an integer greater than 0 or equal to -1")

        if (concat_ratio >= 0) and (concat_ratio<=1):

            df1_sample = int(n_sample * concat_ratio)
            
        elif int(concat_ratio) == concat_ratio:
            df1_sample = concat_ratio
        else:
            ValueError("Concat_ratio must be a positive integer or in the range [0..1]!")
        
        df2_sample = n_sample - df1_sample

        if self.problem_type == 'classification':
            stratify_column = self.paths['target_name']
        elif self.problem_type == 'regression':
            stratify_column = None
        else:
            ValueError(f"This {self.problem_type} is not supported!")

        if (df1_sample <df1.shape[0]):
            _, df1 = self.split_data(df1, stratify_column, df1_sample)
        if df1_sample < df2.shape[0]:
            _, df2 = self.split_data(df2, stratify_column, df2_sample)
        
        return pd.concat([df1, df2], axis=0, join = 'inner')

            
#        return pd.concat([df1.sample(n = df1_sample, random_state = 42, axis = 'index'), 
#            df2.sample(n = df2_sample, random_state=42, axis = 'index', )], axis=0)




