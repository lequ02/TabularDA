import argparse
from train import train
##### Doi output co san tren github thanh output1 va output luon la cai newest
OUT_DIR = "./output/"


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset-name', dest='dataset_name', help='Name of the dataset (e.g., adult, census, news)', type=str, required=True)
  parser.add_argument('--batchsize', dest='batchsize', help='Batch size', type=int, required=True)
  parser.add_argument('--lr', dest='lr', help='Learning rate', type=float, required=True)
  parser.add_argument('--global-round', dest='global_round', help='Number of epochs to train the model', type=int, required=True)
  # parser.add_argument('--test-ratio', dest='test_ratio', help='Number of samples for test set', type=int, required=True)
  parser.add_argument('--train-option', dest='train_option', help='Training data option (original, synthetic or mix)', type=str, required=True)
  parser.add_argument('--augment-option', dest='augment_option', help='Synthetic data option (ctgan or gaussian or categorical)', type=str, default=None)
  parser.add_argument('--test-option', dest='test_option', help='Test data option (original or mix)', type=str, required=True)
  parser.add_argument('--pre-trained-weight-file', dest='pre_trained_w_file', help='File path of a trained model to load', type=str, default=None)

  return parser.parse_args()

def main(args):
  dataset_name = args.dataset_name.lower()

  w_dir = OUT_DIR + dataset_name + "/weight/"
  acc_dir = OUT_DIR + dataset_name + "/acc/"

  train_model = train(
      dataset_name=dataset_name,
      batch_size=args.batchsize,
      learning_rate=args.lr,
      num_epochs=args.global_round,
      w_dir=w_dir,
      acc_dir=acc_dir,
      # test_ratio=args.test_ratio,
      train_option=args.train_option,
      augment_option=args.augment_option,
      test_option=args.test_option,
      pre_trained_w_file=args.pre_trained_w_file
  )
  train_model.training()

if __name__ == '__main__':
  main(parse_args())