import argparse

from train import train


DATASET_NAME = "mnist"

OUT_DIR = "/content/drive/MyDrive/eFL/output/" #gg drive

OUT_DIR = "./output/" #on local

w_DIR = OUT_DIR + DATASET_NAME + "/weight/"
acc_DIR = OUT_DIR + DATASET_NAME + "/acc/"

#def __init__(self, dataset_name, batch_size, learning_rate, lr_decay, num_epochs, w_dir, acc_dir, pre_trained_w_file = None):


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batchsize', dest='batchsize', help='Batch size (default 64)', type=int, default=64)
    parser.add_argument('--lr', dest='lr', help='LEARNING RATE', type=float, default=0.01)
    parser.add_argument('--lr-decay', dest='lr_decay', help='LEARNING RATE DECAY', type=float, default=0.995)
    parser.add_argument('--global-round', dest='global_round', help='Number of epoches to train the model (default 10)', type=int, default=10)
    
    parser.add_argument('--pre-trained-weight-file', dest='pre_trained_w_file', help='file path of a trained model to load (default None)', type = str, default=None)

    return parser.parse_args()

def main(args):
    train_model = train(DATASET_NAME, args.batchsize, args.lr, args.lr_decay, args.global_round, w_DIR, acc_DIR, args.pre_trained_w_file)
    train_model.training()

if __name__ == '__main__':
    main(parse_args())


