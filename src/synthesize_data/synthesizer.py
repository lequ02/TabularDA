from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from naive_bayes import create_label_gaussianNB, create_label_categoricalNB, create_label_gmmNB
from pca_gmm import PCA_GMM
from bayes_net import create_label_BN, create_label_BN_from_trained
import pickle
import torch
import pandas as pd
import numpy as np
import os
from ensemble import *

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# print(sys.path)
from commons.onehot import onehot

def synthesize_data(x_original, y_original, categorical_columns, target_name,
                    features_synthesizer='CTGAN',
                    sample_size=100_000, return_onehot=True,
                    verbose=False, show_network=False,
                    target_synthesizer=None,
                    synthesizer_file_name='synthesizer_onlyX.pkl', 
                    numerical_columns_pca_gmm=None,
                    csv_file_name=None, BN_filename=None,
                    npz_file_name=None, is_classification=True):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by CTGAN (SDV), y' created by GaussianNB, CategoricalNB or Bayesian Network
  """
  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}.csv'


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  # check if target_name is passed as a part of x_original, if not raise an error
  if not target_synthesizer:
    print("Target synthesizer not specified")
    print("Synthesizing data using SDV only")
    print("Assume x_original contains both X and y")
    if target_name not in x_original.columns:
      raise ValueError("Target name is not in x_original.")

  # train synthesizer & create x'
  if features_synthesizer.lower() == 'tvae':
    print("Using TVAE synthesizer")
    synthesizer = train_tvae_synthesizer(x_original, verbose)
  elif features_synthesizer.lower() == 'ctgan':
    print("Using CTGAN synthesizer")
    synthesizer = train_synthesizer_ctgan(x_original, verbose)
  else:
    raise ValueError("features_synthesizer must be 'CTGAN' or 'TVAE'")
  x_synthesized = synthesizer.sample(sample_size)
  if verbose:
    print(f"Successfully synthesized X data with shape {x_synthesized.shape}. Here are the first 5 rows:")
    print(x_synthesized.head())

  # save synthesizer
  synthesizer_file_name = '../sdv trained model/' + synthesizer_file_name
  check_directory(synthesizer_file_name) # create directory if not exist
  synthesizer.save(synthesizer_file_name)
  if verbose:
    print(f"Synthesizer saved at {synthesizer_file_name}")

  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  if not target_synthesizer:
    synthesized_data = x_synthesized.reindex(sorted(x_synthesized.columns), axis=1)
    x_synthesized_backup.drop(columns=[target_name], inplace=True)
    # ensure that the target column is the last column
    y = synthesized_data[target_name]
    synthesized_data.drop(columns=[target_name], inplace=True)
    synthesized_data = pd.concat([synthesized_data, y], axis=1)
    target_synthesizer = 'SDV'

  # create y' using GaussianNB, CategoricalNB or gmmNB
  elif target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'pca_gmm':
    if numerical_columns_pca_gmm is None:
      numerical_columns_pca_gmm =  x_original_backup.columns.difference(categorical_columns)
    pca_gmm = PCA_GMM(x_original, y_original, x_synthesized, 
                      numerical_cols =  numerical_columns_pca_gmm,
                      pca_n_components=0.99, gmm_n_components=10, verbose=verbose,
                      target_name = target_name, filename=csv_file_name, is_classification=is_classification)
    _, synthesized_data = pca_gmm.fit()

  elif target_synthesizer in ['xgb', 'rf']:
    ensemble = Ensemble(x_original, y_original, x_synthesized, target_name=target_name, target_synthesizer=target_synthesizer,
                        filename=csv_file_name, verbose=verbose, is_classification=is_classification)
    _, synthesized_data = ensemble.fit()

  elif target_synthesizer == 'gmmNB':
    raise ValueError("gmmNB is not implemented yet")
    synthesized_data = create_label_gmmNB(x_original, y_original, x_synthesized,
                                          target_name = target_name, filename=csv_file_name)

  # create y' using a Bayesian Network model
  # train a new BN model
  elif target_synthesizer == 'BN_BE':
    synthesized_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='BE', filename=csv_file_name, BN_filename=BN_filename,
                                       verbose=show_network)
  elif target_synthesizer == 'BN_MLE':
    synthesized_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='MLE', filename=csv_file_name, BN_filename=BN_filename, 
                                       verbose=show_network)

  # check if user want to return one-hot encoded X'
  if return_onehot == False:
    x_synthesized_backup = x_synthesized_backup.reindex(sorted(x_synthesized_backup.columns), axis=1)
    synthesized_data = pd.concat([x_synthesized_backup, synthesized_data[target_name]], axis=1)
  
  # save synthesized data to csv
  check_directory(csv_file_name) # create directory if not exist
  synthesized_data.to_csv(csv_file_name, index=False)

  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')


  if npz_file_name is not None:
    # kwargs_dict = synthesized_data.to_dict('list')
    # save to npz file, exclude index column
    synthesized_data_np = synthesized_data.to_numpy()
    check_directory(npz_file_name) # create directory if not exist
    np.savez(npz_file_name, syn=synthesized_data_np)
    synthesized_data.to_csv(csv_file_name, index=False)
    print(f'Data is saved at {npz_file_name}')
    print(f'Data is saved at {csv_file_name}, excluding index column')

  return synthesized_data


def synthesize_from_trained_model(x_original, y_original, categorical_columns, target_name,
                  # features_synthesizer='CTGAN', # doesn't matter because we are loading a trained model
                  sample_size=100_000, return_onehot=True,
                  verbose=False, show_network=False,
                  target_synthesizer=None, 
                  synthesizer_file_name='synthesizer_onlyX.pkl', 
                  numerical_columns_pca_gmm=None,
                  BN_model = None, BN_filename=None,
                  csv_file_name=None, npz_file_name=None,
                  is_classification=True):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by loading a trained synthesizer, y' created by GaussianNB or CategoricalNB
  """
  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}.csv'


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  # check if target_name is passed as a part of x_original, if not raise an error
  if not target_synthesizer:
    print("Target synthesizer not specified")
    print("Synthesizing data using SDV only")
    print("Assume x_original contains both X and y")
    if target_name not in x_original.columns:
      raise ValueError("Target name is not in x_original.")


  # load synthesizer
  synthesizer_file_name = '../sdv trained model/' + synthesizer_file_name
  synthesizer = load_synthesizer(synthesizer_file_name)
  if verbose:
    print(f"Synthesizer loaded from {synthesizer_file_name}")

  # synthesize x'
  x_synthesized = synthesizer.sample(sample_size)
  if verbose:
    print(f"Successfully synthesized X data with shape {x_synthesized.shape}. Here are the first 5 rows:")
    print(x_synthesized.head())


  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)



  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  if not target_synthesizer:
    synthesized_data = x_synthesized.reindex(sorted(x_synthesized.columns), axis=1)
    x_synthesized_backup.drop(columns=[target_name], inplace=True)
    # ensure that the target column is the last column
    y = synthesized_data[target_name]
    synthesized_data.drop(columns=[target_name], inplace=True)
    synthesized_data = pd.concat([synthesized_data, y], axis=1)
    target_synthesizer = 'SDV'

  # create y' using GaussianNB, CategoricalNB, or gmmNB
  elif target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'pca_gmm':
    # print("num cols: ", x_original_backup.columns.difference(categorical_columns))
    if numerical_columns_pca_gmm is None:
      numerical_columns_pca_gmm =  x_original_backup.columns.difference(categorical_columns)

    pca_gmm = PCA_GMM(x_original, y_original, x_synthesized, 
                      numerical_cols =  numerical_columns_pca_gmm,
                      pca_n_components=0.99, gmm_n_components=10, verbose=verbose,
                      target_name = target_name, filename=csv_file_name, is_classification=is_classification)
    _, synthesized_data = pca_gmm.fit()

  elif target_synthesizer in ['xgb', 'rf']:
    ensemble = Ensemble(x_original, y_original, x_synthesized, target_name=target_name, target_synthesizer=target_synthesizer,
                        filename=csv_file_name, verbose=verbose, is_classification=is_classification)
    _, synthesized_data = ensemble.fit()
  elif target_synthesizer == 'gmmNB':
    raise ValueError("gmmNB is not implemented yet")
    synthesized_data = create_label_gmmNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

  # create y' using a Bayesian Network model
  elif BN_model is not None:
  # check if user want to create label from a pre-trained BN model
    synthesized_data = create_label_BN_from_trained(x_original, y_original, x_synthesized, target_name = target_name,
                                                   BN_model=BN_model, filename=csv_file_name, 
                                                   verbose=show_network)

  # if not, train a new BN model
  elif target_synthesizer == 'BN_BE':
    synthesized_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='BE', filename=csv_file_name, BN_filename=BN_filename,
                                       verbose=show_network)
  elif target_synthesizer == 'BN_MLE':
    synthesized_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='MLE', filename=csv_file_name, BN_filename=BN_filename, 
                                       verbose=show_network)

  # check if user want to return one-hot encoded X'
  if return_onehot == False:
    # for i in synthesized_data.columns:
    #   print(i)
    x_synthesized_backup = x_synthesized_backup.reindex(sorted(x_synthesized_backup.columns), axis=1)
    synthesized_data = pd.concat([x_synthesized_backup, synthesized_data[target_name]], axis=1)


  # # check if user want to return one-hot encoded X'
  # if return_onehot == False:
  #   # for i in synthesized_data.columns:
  #   #   print(i)
  #   x_synthesized_backup = x_synthesized_backup.reindex(sorted(x_synthesized_backup.columns), axis=1)
  #   synthesized_data = pd.concat([x_synthesized_backup, synthesized_data[target_name]], axis=1)
  
  # save synthesized data to csv
  check_directory(csv_file_name) # create directory if not exist
  synthesized_data.to_csv(csv_file_name, index=False)

  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')

  if npz_file_name is not None:
    # kwargs_dict = synthesized_data.to_dict('list')
    # save to npz file
    # np.savez(npz_file_name, **kwargs_dict)

    # save to npz file, exclude index column
    synthesized_data_np = synthesized_data.to_numpy()
    check_directory(npz_file_name) # create directory if not exist
    np.savez(npz_file_name, syn=synthesized_data_np)
    synthesized_data.to_csv(csv_file_name, index=False)
    check_directory(csv_file_name) # create directory if not exist
    print(f'Data is saved at {npz_file_name}')
    print(f'Data is saved at {csv_file_name}, excluding index column')

  return synthesized_data


def synthesize_comparison_from_trained_model(x_original, y_original, categorical_columns, target_name,
                  # features_synthesizer='CTGAN', # doesn't matter because we are loading a trained model
                  sample_size=100_000, return_onehot=True,
                  verbose=False, show_network=False,
                  target_synthesizer=None,
                  synthesizer_file_name='synthesizer_onlyX.pkl',
                  numerical_columns_pca_gmm=None,
                  BN_model = None, BN_filename=None,
                  csv_file_name=None, npz_file_name=None,
                  is_classification=True):
  
  """
  This function is going to create synthetic data using a trained synthesizer (trained model includes Y). 
  The target column (Y) is then going to be dropped and replaced with the Y predicted by the target synthesizer.
  The purpose is to compare the generated data with Y vs. without Y.
  """

  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}_compare.csv'

  if not target_synthesizer:
    raise ValueError("Target synthesizer must be specified for comparison function")

  # load synthesizer
  synthesizer_file_name = '../sdv trained model/' + synthesizer_file_name
  synthesizer = load_synthesizer(synthesizer_file_name)
  if verbose:
    print(f"Synthesizer loaded from {synthesizer_file_name}")

  # synthesize x'
  x_synthesized = synthesizer.sample(sample_size)
  # drop target column from x_synthesized
  x_synthesized.drop(columns=[target_name], inplace=True)
  if verbose:
    print("Successfully synthesized X data and dropped target column")
    print(f"Shape {x_synthesized.shape}. Here are the first 5 rows:")
    print(x_synthesized.head())

  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()

  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)

  if target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'pca_gmm':
    # print("num cols: ", x_original_backup.columns.difference(categorical_columns))
    if numerical_columns_pca_gmm is None:
      numerical_columns_pca_gmm =  x_original_backup.columns.difference(categorical_columns)

    pca_gmm = PCA_GMM(x_original, y_original, x_synthesized, 
                      numerical_cols =  numerical_columns_pca_gmm,
                      pca_n_components=0.99, gmm_n_components=10, verbose=verbose,
                      target_name = target_name, filename=csv_file_name, is_classification=is_classification)
    _, synthesized_data = pca_gmm.fit()

  elif target_synthesizer in ['xgb', 'rf']:
    ensemble = Ensemble(x_original, y_original, x_synthesized, target_name=target_name, target_synthesizer=target_synthesizer,
                        filename=csv_file_name, verbose=verbose, is_classification=is_classification)
    _, synthesized_data = ensemble.fit()

  elif target_synthesizer == 'gmmNB':
    raise ValueError("gmmNB is not implemented yet")
    synthesized_data = create_label_gmmNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  # create y' using a Bayesian Network model
  elif BN_model is not None:
    # check if user want to create label from a pre-trained BN model
    synthesized_data = create_label_BN_from_trained(x_original, y_original, x_synthesized, target_name = target_name,
                                                   BN_model=BN_model, filename=csv_file_name, 
                                                   verbose=show_network)
  # if not, train a new BN model
  elif target_synthesizer == 'BN_BE':
    synthesized_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='BE', filename=csv_file_name, BN_filename=BN_filename,
                                       verbose=show_network)
  elif target_synthesizer == 'BN_MLE':
    synthesized_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='MLE', filename=csv_file_name, BN_filename=BN_filename, 
                                       verbose=show_network)
    

  # check if user want to return one-hot encoded X'
  if return_onehot == False:
    # for i in synthesized_data.columns:
    #   print(i)
    x_synthesized_backup = x_synthesized_backup.reindex(sorted(x_synthesized_backup.columns), axis=1)
    synthesized_data = pd.concat([x_synthesized_backup, synthesized_data[target_name]], axis=1)

  # save synthesized data to csv
  check_directory(csv_file_name) # create directory if not exist
  synthesized_data.to_csv(csv_file_name, index=False)
  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')

  if npz_file_name is not None:
    # kwargs_dict = synthesized_data.to_dict('list')
    # save to npz file
    # np.savez(npz_file_name, **kwargs_dict)

    # save to npz file, exclude index column
    synthesized_data_np = synthesized_data.to_numpy()
    check_directory(npz_file_name) # create directory if not exist
    np.savez(npz_file_name, syn=synthesized_data_np)
    synthesized_data.to_csv(csv_file_name, index=False)
    print(f'Data is saved at {npz_file_name}')
    print(f'Data is saved at {csv_file_name}, excluding index column')
  
  
  return synthesized_data

    


def load_synthesizer(link):
  with open(link, 'rb') as file:
    model = pickle.load(file)
  return model

def train_synthesizer_ctgan(data, verbose=False):
  metadata = get_metadata(data)
  synthesizer = create_synthesizer_ctgan(metadata)
  synthesizer.auto_assign_transformers(data)
  synthesizer.fit(data)
  if verbose:
    fig = synthesizer.get_loss_values_plot()
    fig.show()
  return synthesizer

def get_metadata(data, verbose=False):
  metadata = SingleTableMetadata()
  metadata.detect_from_dataframe(data)
  metadata_dict = metadata.to_dict()
  if verbose:
    print(metadata_dict)
    metadata.visualize(
        show_table_details='summarized',
        output_filepath='my_metadata.png'
    )
  return metadata

def create_synthesizer_ctgan(metadata):
  synthesizer = CTGANSynthesizer(
      metadata, # required
      # enforce_rounding=True,
      # epochs=500,
      verbose=True,
      cuda=True
  )
  return synthesizer


def check_directory(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")


def train_tvae_synthesizer(data, verbose=False):
  metadata = get_metadata(data)
  synthesizer = create_synthesizer_tvae(metadata)
  synthesizer.auto_assign_transformers(data)
  synthesizer.fit(data)
  ## doesn't work for tvae
  # if verbose:
  #   fig = synthesizer.get_loss_values_plot()
  #   fig.show()
  return synthesizer

def create_synthesizer_tvae(metadata):
  synthesizer = TVAESynthesizer(
      metadata, # required
      # enforce_rounding=True,
      # epochs=500,
      verbose=True,
      cuda=True
  )
  return synthesizer