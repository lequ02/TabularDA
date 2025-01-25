from pca import pca_df, auto_pca
import pandas as pd
import numpy as np
from GaussMix_nb import GMMNaiveBayes
import sklearn
from sklearn.preprocessing import MinMaxScaler

class PCA_GMM:
    def __init__(self, X_original, y_original, X_synthesized, numerical_cols, target_name, 
                 pca_n_components=0.99, gmm_n_components=10, verbose=True,
                 filename = None):
        self.X_original = X_original
        self.y_original = y_original
        self.X_synthesized = X_synthesized
        self.numerical_cols = numerical_cols
        self.target_name = target_name
        self.pca_n_components = pca_n_components
        self.gmm_n_components = gmm_n_components
        self.verbose = verbose
        self.filename = filename

    def fit(self):
        """
        Normalize the original and synthesized data. (Because PCA is sensitive to the scale of the data)
        Perform PCA on the original and synthetic data numerical columns.
        These numerical columns are now independent. They are assume to have a distribution that can be modeled by a Gaussian Mixture Model.

        # Concat the PCA components of the original and synthetic data with the categorical column.

        Fit a Gaussian Mixture Model to the new data above.
        Use the GMM to predict the target (y_hat) of the synthetic data.


        Parameters:
        X_original (DataFrame): Original dataset.
        y_original (Series or Dataframe): Target column of the original dataset.
        X_synthesized (DataFrame): Synthesized dataset.
        target_name (str): Name of the target column.
        n_components (int, optional): Maximum number of components for each Bayesian Gaussian Mixture Model

        Returns:
        Training results (accuracy, f1).
        DataFrame: Synthetic data with target column.
        """
        X_original_backup = self.X_original.copy()
        X_synthesized_backup = self.X_synthesized.copy()

        # Normalize the original and synthesized data
        if self.verbose:
            print('Normalizing data...')
        scaler = MinMaxScaler()
        self.X_original[self.numerical_cols] = scaler.fit_transform(self.X_original[self.numerical_cols])
        self.X_synthesized[self.numerical_cols] = scaler.transform(self.X_synthesized[self.numerical_cols])

        # print("\n\nBefore PCA: NaN values in data: ", np.isnan(X_original_backup).sum())



        # Perform PCA on the original and synthetic data numerical columns
        if self.verbose:
            print('Performing PCA...')
        pca_X_original, pca_synthesized_df= self.X_original.copy(), self.X_synthesized.copy()

        # after pca, number of numerical columns may have changed
        pca_numeric_original, pca_numeric_synthesized, _ = pca_df(self.X_original[self.numerical_cols], self.X_synthesized[self.numerical_cols], self.target_name, n_components=self.pca_n_components)

        pca_X_original.drop(columns=self.numerical_cols, inplace=True)
        pca_synthesized_df.drop(columns=self.numerical_cols, inplace=True)

        pca_X_original = pd.concat([pca_X_original, pca_numeric_original], axis=1)
        pca_synthesized_df = pd.concat([pca_synthesized_df, pca_numeric_synthesized], axis=1)

        pca_numeric_cols = pca_numeric_original.columns

        print("\n\n")
        print(pca_X_original.head())
        print(pca_numeric_cols)

        # pca_X_original[self.numerical_cols], pca_synthesized_df[self.numerical_cols], pca = pca_df(self.X_original[self.numerical_cols], self.X_synthesized[self.numerical_cols], 
        #                                                   self.target_name, n_components=self.pca_n_components)
        

        # print("After PCA: NaN values in data: ", np.isnan(pca_X_original).sum())


        # Fit GMM to the new data
        # X = pca_X_original.drop(columns=[self.target_name])
        # y = pca_X_original[self.target_name]

        # print("X shape: ", X.shape)
        # print(X.head())

        if self.verbose:
            print('Fitting GMM...')
        gmm = GMMNaiveBayes(n_components=self.gmm_n_components)
        gmm.fit(pca_X_original, self.y_original, numeric_cols=pca_numeric_cols) # use pca_numeric_cols instead of self.numerical_cols because after pca, number of numerical columns may have changed

        # Train results
        # y_train = pca_X_original[self.target_name]
        y_train = self.y_original
        y_hat_train = gmm.predict(pca_X_original)
        train_accuracy = np.mean(y_hat_train == y_train)
        train_f1 = {}
        train_f1['weighted'] = sklearn.metrics.f1_score(y_train, y_hat_train, average='weighted')
        train_f1['macro'] = sklearn.metrics.f1_score(y_train, y_hat_train, average='macro')
        train_f1['micro'] = sklearn.metrics.f1_score(y_train, y_hat_train, average='micro')
        if self.verbose:
            print('Train Accuracy: ', train_accuracy)
            print('Train F1', train_f1)


        # Predict target of synthesized data
        # if synthesized data has target column, drop it
        if self.target_name in pca_synthesized_df.columns:
            X_synthesized = pca_synthesized_df.drop(columns=[self.target_name])
            self.X_synthesized = self.X_synthesized.drop(columns=[self.target_name])
        else:
            X_synthesized = pca_synthesized_df
        y_hat = gmm.predict(X_synthesized)
        y_hat = pd.DataFrame(y_hat, columns=[self.target_name])

        # Concatenate synthesized data (not normalized or pca) with target column
        synthesized_df = pd.concat([X_synthesized_backup, y_hat], axis=1)

        if self.filename:
            synthesized_df.to_csv(self.filename, index=False)

        return (train_accuracy, train_f1), synthesized_df
