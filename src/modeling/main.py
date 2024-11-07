import argparse
from train import train

OUT_DIR = "./output/"

NUMERICAL_COLUMNS = {
  'adult': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
  'census': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
  'news': [
      ' n_tokens_title', ' n_tokens_content', ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens',
      ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos', ' average_token_length', ' num_keywords',
      ' kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
      ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares', ' self_reference_max_shares', ' self_reference_avg_sharess',
      ' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity', ' global_sentiment_polarity',
      ' global_rate_positive_words', ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words',
      ' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', ' avg_negative_polarity',
      ' min_negative_polarity', ' max_negative_polarity', ' title_subjectivity', ' title_sentiment_polarity',
      ' abs_title_subjectivity', ' abs_title_sentiment_polarity'
  ]
}

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset-name', dest='dataset_name', help='Name of the dataset (e.g., adult, census, news)', type=str, required=True)
  parser.add_argument('--batchsize', dest='batchsize', help='Batch size', type=int, required=True)
  parser.add_argument('--lr', dest='lr', help='Learning rate', type=float, required=True)
  parser.add_argument('--global-round', dest='global_round', help='Number of epochs to train the model', type=int, required=True)
  parser.add_argument('--data-path', dest='data_path', help='Path to the dataset CSV file', type=str, required=True)
  parser.add_argument('--test-ratio', dest='test_ratio', help='Number of samples for test set', type=int, required=True)
  parser.add_argument('--train-option', dest='train_option', help='Training data option (original or mix)', type=str, required=True)
  parser.add_argument('--test-option', dest='test_option', help='Test data option (original or mix)', type=str, required=True)
  parser.add_argument('--pre-trained-weight-file', dest='pre_trained_w_file', help='File path of a trained model to load', type=str, default=None)

  return parser.parse_args()

def main(args):
  dataset_name = args.dataset_name.lower()
  if dataset_name not in NUMERICAL_COLUMNS:
      raise ValueError(f"Unknown dataset name: {dataset_name}")

  w_dir = OUT_DIR + dataset_name + "/weight/"
  acc_dir = OUT_DIR + dataset_name + "/acc/"

  train_model = train(
      dataset_name=dataset_name,
      data_path=args.data_path,
      batch_size=args.batchsize,
      learning_rate=args.lr,
      num_epochs=args.global_round,
      w_dir=w_dir,
      acc_dir=acc_dir,
      test_ratio=args.test_ratio,
      train_option=args.train_option,
      test_option=args.test_option,
      numerical_columns=NUMERICAL_COLUMNS[dataset_name],
      pre_trained_w_file=args.pre_trained_w_file
  )
  train_model.training()

if __name__ == '__main__':
  main(parse_args())