from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from onehot import onehot
from naive_bayes import create_label_gaussianNB, create_label_categoricalNB
from bayes_net import create_label_BN, create_label_BN_from_trained
import pickle
import torch

def synthesize_data(x_original, y_original, categorical_columns,
                    sample_size=100_000, verbose=False, show_network=False,
                    target_synthesizer='gaussianNB', target_name='income',
                    synthesizer_file_name='synthesizer_onlyX.pkl', csv_file_name=None, BN_filename=None):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by CTGAN (SDV), y' created by GaussianNB or CategoricalNB
  """
  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}.csv'

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

  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)

  # create y' using GaussianNB or CategoricalNB
  if target_synthesizer == 'gaussianNB':
    synthesize_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesize_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')
  return synthesized_data


def synthesize_from_trained_model(x_original, y_original, categorical_columns,
                  sample_size=100_000, verbose=False, show_network=False,
                  target_synthesizer='gaussianNB', target_name='income',
                  synthesizer_file_name='synthesizer_onlyX.pkl', BN_model = None,
                  BN_filename=None, csv_file_name=None):
  """
  input: original data
  output: synthesized data.
  X' and y' are concatenated. X' created by loading a trained synthesizer, y' created by GaussianNB or CategoricalNB
  """
  if csv_file_name is None:
    csv_file_name = f'synthesized_data_{target_synthesizer}.csv'

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

  # one-hot encode
  x_original, x_synthesized = onehot(x_original, x_synthesized, categorical_columns, verbose=verbose)

  # create y' using GaussianNB or CategoricalNB
  if target_synthesizer == 'gaussianNB':
    synthesize_data = create_label_gaussianNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)
  elif target_synthesizer == 'categoricalNB':
    synthesize_data = create_label_categoricalNB(x_original, y_original, x_synthesized, target_name = target_name, filename=csv_file_name)

  # create y' using a Bayesian Network model
  elif BN_model is not None:
  # check if user want to create label from a pre-trained BN model
    synthesize_data = create_label_BN_from_trained(x_original, y_original, x_synthesized, target_name = target_name,
                                                   BN_model=BN_model, filename=csv_file_name, 
                                                   verbose=show_network)

  # if not, train a new BN model
  elif target_synthesizer == 'BN_BE':
    synthesize_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='BE', filename=csv_file_name, BN_filename=BN_filename,
                                       verbose=show_network)
  elif target_synthesizer == 'BN_MLE':
    synthesize_data = create_label_BN(x_original, y_original, x_synthesized, target_name = target_name,
                                       BN_type='MLE', filename=csv_file_name, BN_filename=BN_filename, 
                                       verbose=show_network)

  if verbose:
    print(f"Successfully synthesized X and y data with {target_synthesizer}")
    print(f'Data is saved at {csv_file_name}')
  return synthesize_data


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
