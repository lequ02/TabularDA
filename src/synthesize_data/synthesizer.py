from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from onehot import onehot
from naive_bayes import create_label_gaussianNB, create_label_categoricalNB
from bayes_net import create_label_BN, create_label_BN_from_trained
from bayes_net import create_label_BN, create_label_BN_from_trained
import pickle
import torch
import pandas as pd
import numpy as np

def synthesize_data(x_original, y_original, categorical_columns, target_name,
                    sample_size=100_000, return_onehot=True,
                    verbose=False, show_network=False,
                    target_synthesizer=None,
                    synthesizer_file_name='synthesizer_onlyX.pkl', 
                    csv_file_name=None, BN_filename=None,
                    npz_file_name=None):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by CTGAN (SDV), y' created by GaussianNB, CategoricalNB or Bayesian Network
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


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  # check if target_name is passed as a part of x_original, if not raise an error
  if not target_synthesizer:
    print("Target synthesizer not specified")
    print("Synthesizing data using SDV only")
    print("Assume x_original contains both X and y")
    if target_name not in x_original.columns:
      raise ValueError("Target name is not in x_original.")

  # train synthesizer & create x'
  synthesizer = train_synthesizer(x_original, verbose)
  x_synthesized = synthesizer.sample(sample_size)
  if verbose:
    print(f"Successfully synthesized X data with shape {x_synthesized.shape}. Here are the first 5 rows:")
    print(x_synthesized.head())

  # save synthesizer
  synthesizer_file_name = '../sdv trained model/' + synthesizer_file_name
  synthesizer.save(synthesizer_file_name)
  if verbose:
    print(f"Synthesizer saved at {synthesizer_file_name}")

  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  if not target_synthesizer:
    # print("Target synthesizer not specified")
    # print("Synthesizing data using SDV only")
    # print("Assume x_original contains both X and y")
    synthesized_data = x_synthesized.reindex(sorted(x_synthesized.columns), axis=1)
    x_synthesized_backup.drop(columns=[target_name], inplace=True)
    # ensure that the target column is the last column
    y = synthesized_data[target_name]
    synthesized_data.drop(columns=[target_name], inplace=True)
    synthesized_data = pd.concat([synthesized_data, y], axis=1)
    target_synthesizer = 'SDV'


  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  if not target_synthesizer:
    # print("Target synthesizer not specified")
    # print("Synthesizing data using SDV only")
    # print("Assume x_original contains both X and y")
    synthesized_data = x_synthesized.reindex(sorted(x_synthesized.columns), axis=1)
    x_synthesized_backup.drop(columns=[target_name], inplace=True)
    # ensure that the target column is the last column
    y = synthesized_data[target_name]
    synthesized_data.drop(columns=[target_name], inplace=True)
    synthesized_data = pd.concat([synthesized_data, y], axis=1)
    target_synthesizer = 'SDV'

  # create y' using GaussianNB or CategoricalNB
  elif target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

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
  # synthesized_data.to_csv(csv_file_name)
  # synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

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
  synthesized_data.to_csv(csv_file_name)

  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')


  if npz_file_name is not None:
    # kwargs_dict = synthesized_data.to_dict('list')
    # save to npz file, exclude index column
    synthesized_data_np = synthesized_data.to_numpy()
    np.savez(npz_file_name, syn=synthesized_data_np)
    synthesized_data.to_csv(csv_file_name, index=False)
    print(f'Data is saved at {npz_file_name}')
    print(f'Data is saved at {csv_file_name}, excluding index column')

  return synthesized_data


def synthesize_from_trained_model(x_original, y_original, categorical_columns, target_name,
                  sample_size=100_000, return_onehot=True,
                  verbose=False, show_network=False,
                  target_synthesizer=None, 
                  synthesizer_file_name='synthesizer_onlyX.pkl', BN_model = None,
                  BN_filename=None, csv_file_name=None, npz_file_name=None):
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
  # pre-encode backups
  x_original_backup = x_original.copy()
  x_synthesized_backup = x_synthesized.copy()
  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)

  # for i in x_synthesized.columns:
  #   print(i)

  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  if not target_synthesizer:
    # print("Target synthesizer not specified")
    # print("Synthesizing data using SDV only")
    # print("Assume x_original contains both X and y")
    synthesized_data = x_synthesized.reindex(sorted(x_synthesized.columns), axis=1)
    x_synthesized_backup.drop(columns=[target_name], inplace=True)
    y = synthesized_data[target_name]
    synthesized_data.drop(columns=[target_name], inplace=True)
    synthesized_data = pd.concat([synthesized_data, y], axis=1)
    # print(set(synthesize_data.columns))
    target_synthesizer = 'SDV'
    # synthesized_data.to_csv(csv_file_name)

  # for i in x_synthesized.columns:
  #   print(i)

  # if a target synthesizer is not specified, assume that the user wants to synthesize X' and y' using CTGAN only
  if not target_synthesizer:
    # print("Target synthesizer not specified")
    # print("Synthesizing data using SDV only")
    # print("Assume x_original contains both X and y")
    synthesized_data = x_synthesized.reindex(sorted(x_synthesized.columns), axis=1)
    x_synthesized_backup.drop(columns=[target_name], inplace=True)
    y = synthesized_data[target_name]
    synthesized_data.drop(columns=[target_name], inplace=True)
    synthesized_data = pd.concat([synthesized_data, y], axis=1)
    # print(set(synthesize_data.columns))
    target_synthesizer = 'SDV'
    # synthesized_data.to_csv(csv_file_name)

  # create y' using GaussianNB or CategoricalNB
  elif target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'gaussianNB':
    synthesized_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

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
  # synthesized_data.to_csv(csv_file_name)
  #   synthesized_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

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
  synthesized_data.to_csv(csv_file_name)

  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')

  if npz_file_name is not None:
    # kwargs_dict = synthesized_data.to_dict('list')
    # save to npz file
    # np.savez(npz_file_name, **kwargs_dict)

    # save to npz file, exclude index column
    synthesized_data_np = synthesized_data.to_numpy()
    np.savez(npz_file_name, syn=synthesized_data_np)
    synthesized_data.to_csv(csv_file_name, index=False)
    print(f'Data is saved at {npz_file_name}')
    print(f'Data is saved at {csv_file_name}, excluding index column')

  return synthesized_data


def load_synthesizer(link):
  with open(link, 'rb') as file:
    model = pickle.load(file)
  return model

def train_synthesizer(data, verbose=False):
  metadata = get_metadata(data)
  synthesizer = create_synthesizer(metadata)
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

def create_synthesizer(metadata):
  synthesizer = CTGANSynthesizer(
      metadata, # required
      # enforce_rounding=True,
      # epochs=500,
      verbose=True,
      cuda=True
  )
  return synthesizer
