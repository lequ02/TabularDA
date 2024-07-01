import argparse
from train import train

DATASET_NAME = "adult"
OUT_DIR = "./output/"
W_DIR = OUT_DIR + DATASET_NAME + "/weight/"
ACC_DIR = OUT_DIR + DATASET_NAME + "/acc/"

NUMERICAL_COLUMNS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 64)', type=int, default=64)
    parser.add_argument('--lr', dest='lr', help='LEARNING RATE', type=float, default=0.001)
    parser.add_argument('--lr-decay', dest='lr_decay', help='LEARNING RATE DECAY', type=float, default=0.995)
    parser.add_argument('--global-round', dest='global_round', help='Number of epochs to train the model (default 10)', type=int, default=10)
    parser.add_argument('--data-path', dest='data_path', help='Path to the dataset CSV file', type=str, default='./data/adult/onehot_adult_sdv_gaussian_100k.csv')
    parser.add_argument('--test-ratio', dest='test_ratio', help='Number of samples for test set', type=int, default=10000)
    parser.add_argument('--train-option', dest='train_option', help='Training data option (original or mix)', type=str, default='mix')
    parser.add_argument('--test-option', dest='test_option', help='Test data option (original or mix)', type=str, default='mix')
    parser.add_argument('--pre-trained-weight-file', dest='pre_trained_w_file', help='File path of a trained model to load (default None)', type=str, default=None)

    return parser.parse_args()

def main(args):
    train_model = train(
        dataset_name=DATASET_NAME,
        data_path=args.data_path,
        batch_size=args.batchsize,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        num_epochs=args.global_round,
        w_dir=W_DIR,
        acc_dir=ACC_DIR,
        test_ratio=args.test_ratio,
        train_option=args.train_option,
        test_option=args.test_option,
        numerical_columns=NUMERICAL_COLUMNS,
        pre_trained_w_file=args.pre_trained_w_file
    )
    train_model.training()

if __name__ == '__main__':
    main(parse_args())
    
    
    #running the code in the command line:
    #python ./src/main.py --batchsize 64 --lr 0.001 --lr-decay 0.995 --global-round 10 --data-path './data/adult/onehot_adult_sdv_categorical_100k.csv' --test-ratio 10000 --train-option 'mix' --test-option 'original'
    
    
