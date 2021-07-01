"""
Train Output ANN model
"""
import os
import argparse
import warnings

import numpy as np
import torch

from data_loader import LzoLoader

from processor import AnnProcessor, GanProcessor

from glove import Glove

project_path = os.path.dirname(os.path.abspath("__file__"))
input_path = os.path.join(project_path, "input")
model_path = os.path.join(input_path, "model")
data_path = os.path.join(input_path, "data")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_glove(model_path):
    glove_model = f"glove_10_25_0.05_20.model"

    glove = None

    try:
        glove = Glove.load(os.path.join(model_path, glove_model))
    except FileNotFoundError:
        print(f"There is no pretrained glove model at {model_path}")

    return glove


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="execute pretraining for input features")

    parser.add_argument("--epochs", type=int, default=None,
                        help="the number of epochs for training model")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="the number of batch_size for training model")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="learning rate for training model")
    parser.add_argument("--patience", type=int, default=30,
                        help="patience of EarlyStopping for training model")
    parser.add_argument("--token_model", type=str, default=None,
                        help="pass target model be pretrained [tfidf-glove | SIF | DAN]")
    parser.add_argument("--gan_ratio", type=float, default=0.5, help="condition to turn on | off gan inference")

    parser.add_argument("--lzo_numbers", nargs="*", default=None, help="lzo number list be trained")
    parser.add_argument("--target_column", type=str, default=None, help="training target column [reply | retweet | retweet_c | like]")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="train ratio of training neural network")

    parser.add_argument("--global_seed", type=int, default=42, help="global seed for fixing training process")

    parser.add_argument("--verbose", type=str2bool, default=False, help="condition to print out training process")

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    patience = args.patience

    token_model = args.token_model
    train_ratio = args.train_ratio
    gan_ratio = args.gan_ratio

    lzo_numbers = args.lzo_numbers
    lzo_numbers = list(map(lambda x: int(x), lzo_numbers[0].split(",")))

    target_column = args.target_column

    global_seed = args.global_seed

    verbose = args.verbose

    lzo_loader = LzoLoader(data_path, verbose)
    df_train = lzo_loader.load_data(lzo_numbers)

    if train_ratio <= 0.0 and train_ratio > 1.0:
        raise argparse.ArgumentError(train_ratio, "train_ratio must be (0.0, 1.0] at pretrain phase")

    target_col = None

    if target_column is None:
        raise argparse.ArgumentTypeError(target_column, f"ANN needs to define target column")

    else:
        if target_column == "reply":
            target_col = "reply_timestamp"
        elif target_column == "retweet":
            target_col = "retweet_timestamp"
        elif target_column == "retweet_c":
            target_col = "retweet_with_comment_timestamp"
        elif target_column == "like":
            target_col = "like_timestamp"
        else:
            raise argparse.ArgumentTypeError(target_column,
                                             f"target column needs to define among [reply | retweet | retweet_c | like] not {target_column}")

    glove = load_glove(model_path)

    ann_processor = AnnProcessor(model_path, global_seed, verbose)

    gan_processor = None
    if gan_ratio > 0.0:
        gan_processor = GanProcessor(model_path, global_seed, verbose)

    torch.backends.cudnn.deterministic = False

    datas = ann_processor.preprocess(df_train, train_ratio, glove, token_model, target_col)


    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if train_ratio == 1.0:
        X_train, y_train = datas
    else:
        X_train, y_train, X_test, y_test = datas

    if gan_ratio > 0.0:
        true_num = (y_train == 1).sum()
        X_fake_trn, y_fake_trn = gan_processor.predict(token_model, target_column, model_path, true_num, gan_ratio)
        X_train = np.vstack([X_train, X_fake_trn])
        y_train = np.append(y_train, y_fake_trn)

    torch.backends.cudnn.deterministic = True

    ann_processor.train((X_train, y_train, X_test, y_test), target_column, epochs, batch_size, learning_rate, patience, token_model, gan_ratio)

if __name__ == '__main__':
        main()