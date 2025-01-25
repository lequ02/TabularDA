from sklearn.decomposition import PCA
import pandas as pd

def pca_df(df_original, df_synthesized, target_name, verbose=True, n_components=None):
    """
    Perform PCA on the original data and project the synthesized data into the same space.
    
    Parameters:
    df_original (DataFrame): Original dataset.
    df_synthesized (DataFrame): Synthesized dataset.
    n_components (int, optional): Number of components to keep. If None, keep all components.
    
    Returns: 
    DataFrames: Transformed datasets in PCA space.
    df_original_pca, df_synthesized_pca
    """
    only_num_flag = False
    if target_name not in df_original.columns or target_name not in df_synthesized.columns:
        print(f"Warning: Target column {target_name} not found in the dataset.")
        print("Assumes the dfs only contain numerical columns.")
        only_num_flag = True
        x_original = df_original.values
        x_synthesized = df_synthesized.values

    else:
        y_original = df_original[target_name]
        y_synthesized = df_synthesized[target_name]

        x_original = df_original.drop(columns=[target_name]).values
        x_synthesized = df_synthesized.drop(columns=[target_name]).values
    
    # Initialize PCA with the specified number of components
    pca = PCA(n_components=n_components)
    
    # Fit PCA on the original data
    pca.fit(x_original)

    if verbose:
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")
        print(f"Number of components: {pca.n_components_}")

    x_original_pca = pca.transform(x_original)
    x_synthesized_pca = pca.transform(x_synthesized)

    # print("\n\n pre df")
    # print(type(x_original_pca))
    # print(x_original_pca)

    # check if x_original_pca & df_synthesized_pca is a numpy array or a dataframe. 
    # If it is a numpy array, convert it to a dataframe
    # If it is a dataframe, do nothing
    if isinstance(x_original_pca, pd.DataFrame):
        df_original_pca = x_original_pca
        df_synthesized_pca = x_synthesized_pca
    else:
        df_original_pca = pd.DataFrame(x_original_pca, columns=[f'PC{i+1}' for i in range(x_original_pca.shape[1])])
        df_synthesized_pca = pd.DataFrame(x_synthesized_pca, columns=[f'PC{i+1}' for i in range(x_synthesized_pca.shape[1])])

    # Add the target column back to the dataframes if needed
    if not only_num_flag:
        df_original_pca[target_name] = y_original.values
        df_synthesized_pca[target_name] = y_synthesized.values
    
    # print("\n\n post df")
    # print(type(df_original_pca))
    # print(df_original_pca)

    return df_original_pca, df_synthesized_pca, pca


def auto_pca(df_original, df_synthesized, target_name, verbose=True, lower_variance_threshold=0.999, min_components=10):
    """
    Perform PCA on the original data and project the synthesized data into the same space, automatically determining the number of components to retain based on a variance threshold.
    Final data must have >=10 features and >=0.999 variance explained after PCA.
    """
    # Initialize PCA and fit on the original data
    df_original_pca, df_synthesized_pca, pca = pca_df(df_original, df_synthesized, target_name, verbose=False)

    # Determine the number of components to retain based on the variance threshold
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (cumulative_variance < lower_variance_threshold).sum() + 1
    n_components = max(n_components, min_components)  # Ensure at least 10 components

    if verbose:
        print(f"Number of components selected: {n_components}")
        print(f"Cumulative explained variance: {cumulative_variance[n_components-1]}")


    explained_variance_table = pd.DataFrame(cumulative_variance, columns=['Explained Variance (cummulative)'])
    explained_variance_table.index = explained_variance_table.index + 1
    explained_variance_table.index.name = 'Component'

    # Perform PCA with the determined number of components
    df_original_pca, df_synthesized_pca, _ = pca_df(df_original, df_synthesized, target_name, n_components=n_components, verbose=False)


    return df_original_pca, df_synthesized_pca, explained_variance_table