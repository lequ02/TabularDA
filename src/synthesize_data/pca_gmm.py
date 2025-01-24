from pca import pca_df, auto_pca
import pandas as pd
import numpy as np

def pca_gmm(df_original, df_synthesized, target_name, n_components=None, n_clusters=3, verbose=True):
    """
    Perform PCA on the original data and project the synthesized data into the same space.
    Fit a Gaussian Mixture Model on the original data and predict the clusters of the synthesized data.
    
    Parameters:
    df_original (DataFrame): Original dataset.
    df_synthesized (DataFrame): Synthesized dataset.
    target_name (str): Name of the target column.
    n_components (int, optional): Number of components to keep. If None, keep all components.
    n_clusters (int): Number of clusters for the Gaussian Mixture Model.
    
    Returns:
    DataFrames: Transformed datasets in PCA space.
    df_original_pca, df_synthesized_pca
    """