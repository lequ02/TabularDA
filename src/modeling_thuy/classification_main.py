import argparse
from classification_train import train
##### Doi output co san tren github thanh output1 va output luon la cai newest
OUT_DIR = "./output/"


def parse_args():
  parser = argparse.ArgumentParser()

#data
  parser.add_argument('--dataset-name', dest='dataset_name', help='Name of the dataset (e.g., adult, census, intrusion)', type=str, required=True)
  parser.add_argument('--train-option', dest='train_option', help='Training data option (original, synthetic or mix)', type=str, required=True)
  parser.add_argument('--augment-option', dest='augment_option', help='Synthetic data option (ctgan or gaussian or categorical)', type=str, default=None)
  parser.add_argument('--test-option', dest='test_option', help='Test data option (original or mix)', type=str, required=True)
  parser.add_argument('--validation', dest='validation', help='validation percentage', type=float, required=True, default = 0.2)

#training  
  parser.add_argument('--batchsize', dest='batchsize', help='Batch size', type=int, required=True)
  parser.add_argument('--lr', dest='lr', help='Learning rate', type=float, required=True)
  parser.add_argument('--global-round', dest='global_round', help='Number of epochs to train the model', type=int, required=True)
  parser.add_argument('--patience', dest='patience', help='Patience for early stopping, -1 means no early stopping', type=int, default=30)
  parser.add_argument('--early-stop-crit', dest='early_stop_criterion', help='Early stopping criterion (any metrics including (f1, accuracy, recall, precision for classification. If f1/recall/precision, need to include type = micro/macro/weighted/binary/samples. Example f1_micro; Example accuracy), (mae, mape, r2 for regression) or loss)', type=str, default='loss')

#pre trained momels if any
  parser.add_argument('--pre-trained-weight-file', dest='pre_trained_w_file', help='File path of a trained model to load', type=str, default=None)
  
  return parser.parse_args()

def main(args):
  dataset_name = args.dataset_name.lower()

  w_dir = OUT_DIR + dataset_name + "/weight/"
  acc_dir = OUT_DIR + dataset_name + "/acc/"
  eval_metrics = {"accuracy": None, "f1": ['macro', 'micro', 'weighted','binary']} # add 'binary' to the list for binary classification (i.e: adult, census)
  metric_to_plot = "f1_micro"


  train_model = train(
      dataset_name=dataset_name,
      train_option=args.train_option,
      augment_option=args.augment_option,
      test_option=args.test_option,
      validation = args.validation,
      
      batch_size=args.batchsize,
      learning_rate=args.lr,
      num_epochs=args.global_round,
      patience=args.patience,
      early_stop_criterion=args.early_stop_criterion,

      eval_metrics = eval_metrics,
      metric_to_plot = metric_to_plot,

      pre_trained_w_file=args.pre_trained_w_file,
      
      w_dir=w_dir,
      acc_dir=acc_dir

  )
  train_model.training()

if __name__ == '__main__':
  main(parse_args())