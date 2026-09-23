from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
from naive_bayes import create_label_gaussianNB, create_label_categoricalNB, create_label_gmmNB
from pca_gmm import PCA_GMM
from bayes_net import create_label_BN, create_label_BN_from_trained
import pickle
import torch
import pandas as pd
import numpy as np
import os
import json
import hashlib
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
                    npz_file_name=None, is_classification=True, seed=42):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by CTGAN (SDV), y' created by GaussianNB, CategoricalNB or Bayesian Network
  """
  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}.csv'

  _validate_labeler_task(target_synthesizer, is_classification)


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
    synthesizer = train_tvae_synthesizer(
        x_original, verbose, categorical_columns=categorical_columns,
        target_name=target_name if (not target_synthesizer and is_classification) else None, seed=seed)
  elif features_synthesizer.lower() == 'ctgan':
    print("Using CTGAN synthesizer")
    synthesizer = train_synthesizer_ctgan(
        x_original, verbose, categorical_columns=categorical_columns,
        target_name=target_name if (not target_synthesizer and is_classification) else None, seed=seed)
  else:
    raise ValueError("features_synthesizer must be 'CTGAN' or 'TVAE'")
  x_synthesized = synthesizer.sample(num_rows=sample_size)
  if verbose:
    print(f"Successfully synthesized {len(x_synthesized)} rows with {x_synthesized.shape[1]} columns")

  # save synthesizer
  check_directory(synthesizer_file_name) # create directory if not exist
  synthesizer.save(synthesizer_file_name)
  _save_synthesis_provenance(synthesizer, x_original, sample_size, synthesizer_file_name, seed)
  if verbose:
    print(f"Synthesizer saved at {synthesizer_file_name}")

  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)


    # safe guard
  if not x_original.columns.equals(x_synthesized.columns):
    print("The columns are not the same.")
    print("x_original columns: ")
    print(x_original.columns)
    print("x_synthesized columns: ")
    print(x_synthesized.columns)
    raise ValueError("x_original and x_synthesized have different columns or the columns are in different orders")


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
  _save_synthetic_quality(
      synthesized_data, x_original, y_original, target_name, sample_size, csv_file_name,
      is_classification
  )

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
                  is_classification=True, seed=42):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by loading a trained synthesizer, y' created by GaussianNB or CategoricalNB
  """
  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}.csv'

  _validate_labeler_task(target_synthesizer, is_classification)


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  # check if target_name is passed as a part of x_original, if not raise an error
  if not target_synthesizer:
    print("Target synthesizer not specified")
    print("Synthesizing data using SDV only")
    print("Assume x_original contains both X and y")
    if target_name not in x_original.columns:
      raise ValueError("Target name is not in x_original.")


  # load synthesizer
  synthesizer = load_synthesizer(
      synthesizer_file_name, expected_data=x_original, expected_seed=seed
  )
  if verbose:
    print(f"Synthesizer loaded from {synthesizer_file_name}")

  # synthesize x'
  torch.manual_seed(seed)
  np.random.seed(seed)
  x_synthesized = synthesizer.sample(num_rows=sample_size)
  if verbose:
    print(f"Successfully synthesized {len(x_synthesized)} rows with {x_synthesized.shape[1]} columns")


  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)

  # safe guard
  if not x_original.columns.equals(x_synthesized.columns):
    print("The columns are not the same.")
    print("x_original columns: ")
    print(x_original.columns)
    print("x_synthesized columns: ")
    print(x_synthesized.columns)
    raise ValueError("x_original and x_synthesized have different columns or the columns are in different orders")

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

  # save synthesized data to csv
  check_directory(csv_file_name) # create directory if not exist
  synthesized_data.to_csv(csv_file_name, index=False)
  _save_synthetic_quality(
      synthesized_data, x_original, y_original, target_name, sample_size, csv_file_name,
      is_classification
  )

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
                  is_classification=True, seed=42):

  """
  This function is going to create synthetic data using a trained synthesizer (trained model includes Y).
  The target column (Y) is then going to be dropped and replaced with the Y predicted by the target synthesizer.
  The purpose is to compare the generated data with Y vs. without Y.
  """

  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}_compare.csv'

  if not target_synthesizer:
    raise ValueError("Target synthesizer must be specified for comparison function")
  _validate_labeler_task(target_synthesizer, is_classification)

  # load synthesizer
  synthesizer = load_synthesizer(
      synthesizer_file_name, expected_data=x_original, expected_seed=seed
  )
  if verbose:
    print(f"Synthesizer loaded from {synthesizer_file_name}")

  # synthesize x'
  torch.manual_seed(seed)
  np.random.seed(seed)
  x_synthesized = synthesizer.sample(num_rows=sample_size)
  if verbose:
    print("Successfully synthesized X data")
    print(f"Shape {x_synthesized.shape}")

  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()

  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)

      # safe guard
  if not x_original.columns.equals(x_synthesized.columns):
    print("The columns are not the same.")
    print("x_original columns: ")
    print(x_original.columns)
    print("x_synthesized columns: ")
    print(x_synthesized.columns)
    raise ValueError("x_original and x_synthesized have different columns or the columns are in different orders")

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
  _save_synthetic_quality(
      synthesized_data, x_original, y_original, target_name, sample_size, csv_file_name,
      is_classification
  )
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




def load_synthesizer(link, expected_data=None, expected_seed=None):
  if expected_data is not None or expected_seed is not None:
    provenance_path = os.path.splitext(link)[0] + '.provenance.json'
    if not os.path.isfile(provenance_path):
      raise FileNotFoundError(f'Missing synthesizer provenance: {provenance_path}')
    with open(provenance_path, 'r', encoding='utf-8') as file:
      provenance = json.load(file)
    if expected_seed is not None and provenance.get('seed') != expected_seed:
      raise ValueError(f'Synthesizer seed does not match requested seed {expected_seed}')
    if expected_data is not None:
      expected_hash = hashlib.sha256(
          expected_data.to_csv(index=False, lineterminator='\n').encode('utf-8')
      ).hexdigest()
      if provenance.get('fit_table_sha256') != expected_hash:
        raise ValueError('Synthesizer provenance does not match the requested fit table')
  with open(link, 'rb') as file:
    model = pickle.load(file)
  return model

def train_synthesizer_ctgan(data, verbose=False, categorical_columns=None, target_name=None, seed=42):
  torch.manual_seed(seed)
  np.random.seed(seed)
  metadata = get_metadata(data, categorical_columns=categorical_columns, target_name=target_name)
  synthesizer = create_synthesizer_ctgan(metadata)
  synthesizer.fit(data)
  if verbose:
    fig = synthesizer.get_loss_values_plot()
    fig.show()
  return synthesizer

def get_metadata(data, verbose=False, categorical_columns=None, target_name=None):
  metadata = Metadata.detect_from_dataframe(data)
  categorical = set(categorical_columns or [])
  if target_name is not None:
    categorical.add(target_name)
  for column_name in categorical:
    if column_name not in data.columns:
      raise ValueError(f"Categorical metadata column {column_name!r} is not in training data")
    metadata.update_column(table_name='table', column_name=column_name, sdtype='categorical')
  metadata.validate()
  metadata.validate_data({'table': data})
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
      epochs=500,
      batch_size=500,
      verbose=True,
      cuda=True
  )
  return synthesizer


def check_directory(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")


def train_tvae_synthesizer(data, verbose=False, categorical_columns=None, target_name=None, seed=42):
  torch.manual_seed(seed)
  np.random.seed(seed)
  metadata = get_metadata(data, categorical_columns=categorical_columns, target_name=target_name)
  synthesizer = create_synthesizer_tvae(metadata)
  synthesizer.fit(data)
  ## doesn't work for tvae
  # if verbose:
  #   fig = synthesizer.get_loss_values_plot()
  #   fig.show()
  return synthesizer

def create_synthesizer_tvae(metadata):
  synthesizer = TVAESynthesizer(
      metadata, # required
      epochs=500,
      batch_size=500,
      verbose=True,
      cuda=True
  )
  return synthesizer


def _save_synthesis_provenance(synthesizer, data, sample_size, synthesizer_file_name, seed):
  """Save the fitted SDV configuration and table metadata next to the model."""
  provenance = {
      'sdv_version': __import__('sdv').__version__,
      'sample_size': sample_size,
      'seed': seed,
      'training_rows': len(data),
      'training_columns': list(data.columns),
      'fit_table_sha256': hashlib.sha256(
          data.to_csv(index=False, lineterminator='\n').encode('utf-8')
      ).hexdigest(),
      'parameters': synthesizer.get_parameters(),
      'metadata': synthesizer.get_metadata().to_dict(),
  }
  provenance_path = os.path.splitext(synthesizer_file_name)[0] + '.provenance.json'
  with open(provenance_path, 'w', encoding='utf-8') as file:
    json.dump(provenance, file, indent=2, default=str)


def _save_synthetic_quality(synthetic, training_data, training_labels, target_name,
                            sample_size, csv_file_name, is_classification):
  if len(synthetic) != sample_size:
    raise ValueError(f'Expected {sample_size} synthetic rows, got {len(synthetic)}')
  training_features = training_data.drop(columns=[target_name]) if target_name in training_data.columns else training_data
  feature_columns = sorted(training_features.columns)
  expected_columns = feature_columns + [target_name]
  if set(synthetic.columns) != set(expected_columns):
    raise ValueError('Synthetic table columns do not match the expected feature and target schema')
  synthetic = synthetic.loc[:, expected_columns]
  training_features = training_features.loc[:, feature_columns]
  numeric = synthetic.select_dtypes(include=[np.number])
  if not np.isfinite(numeric.to_numpy()).all():
    raise ValueError('Synthetic table contains non-finite numeric values')
  if synthetic[target_name].isna().any():
    raise ValueError('Synthetic table contains missing target values')
  if is_classification:
    real_labels = (training_data[target_name] if target_name in training_data.columns
                   else training_labels)
    unexpected_labels = set(synthetic[target_name].dropna()) - set(real_labels.dropna())
    if unexpected_labels:
      raise ValueError(f'Synthetic table contains unseen target classes: {unexpected_labels}')
  train_hashes = set(pd.util.hash_pandas_object(training_features, index=False).tolist())
  synthetic_features = synthetic.drop(columns=[target_name])
  overlap = int(pd.util.hash_pandas_object(synthetic_features, index=False).isin(train_hashes).sum())
  if target_name in training_data.columns:
    training_full = training_data.loc[:, expected_columns]
  else:
    training_full = pd.concat(
        [training_features.reset_index(drop=True),
         training_labels.rename(target_name).reset_index(drop=True)], axis=1
    )
  quality = {
      'rows': len(synthetic),
      'columns': [str(column) for column in synthetic.columns],
      'synthetic_label_counts': (
          {str(label): int(count)
           for label, count in synthetic[target_name].value_counts(dropna=False).items()}
          if is_classification else None
      ),
      'synthetic_target_range': (
          [float(synthetic[target_name].min()), float(synthetic[target_name].max())]
          if not is_classification else None
      ),
      'exact_training_feature_overlap': overlap,
      'exact_training_full_row_overlap': int(pd.util.hash_pandas_object(
          synthetic.loc[:, expected_columns], index=False
      ).isin(set(pd.util.hash_pandas_object(training_full, index=False).tolist())).sum()),
  }
  quality_path = os.path.splitext(csv_file_name)[0] + '.quality.json'
  with open(quality_path, 'w', encoding='utf-8') as file:
    json.dump(quality, file, indent=2, default=str)


def _validate_labeler_task(target_synthesizer, is_classification):
  classification_only = {'gaussianNB', 'categoricalNB', 'BN_BE', 'BN_MLE'}
  if not is_classification and target_synthesizer in classification_only:
    raise ValueError(f"{target_synthesizer} is classification-only and cannot label a regression target")
