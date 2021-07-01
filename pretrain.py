"""
Pretrain deep learning based model for input features.
"""
import os
import argparse
import warnings

from data_loader import LzoLoader

from models import GLOVE
from processor import DanProcessor, DecProcessor, GanProcessor

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


def train_glove(df_train, window_size, n_components, epochs, learning_rate, model_path, verbose=False):
    if epochs == None:
        epochs = 25
    if learning_rate == None:
        learning_rate = 0.05

    glove = GLOVE(df_train, model_path, window_size, n_components, epochs, learning_rate, verbose)
    glove.train()


def load_glove(model_path):
    glove_model = f"glove_{10}_{25}_{0.05}_{20}.model"

    try:
        glove = Glove.load(os.path.join(model_path, glove_model))
    except FileNotFoundError:
        print(f"There is no pretrained glove model({glove_model}) at {model_path}")

    return glove


def dan(df_train, target_column, train_ratio, glove, epochs, batch_size, learning_rate, patience, model_path, verbose,
        GLOBAL_SEED):
    dan_processor = DanProcessor(df_train, glove, model_path, verbose)
    dataloader = dan_processor.preprocess(target_column, batch_size, train_ratio, GLOBAL_SEED)

    dan_processor.train(dataloader, target_column, epochs, learning_rate, model_path, patience)


def dec(df_train, train_ratio, glove, epochs, batch_size, learning_rate, layers, patience, model_path, verbose,
        GLOBAL_SEED, is_tying=True):
    dec_processor = DecProcessor(df_train, glove, model_path, verbose)

    scaled_embedded_mean_token, np_result_labels = dec_processor.preprocess(GLOBAL_SEED)

    if not dec_processor.check_pretrain(is_tying):
        dec_processor.pretrain(scaled_embedded_mean_token, np_result_labels, layers, model_path, data_path, is_tying)

    else:
        dec_processor.train(scaled_embedded_mean_token, np_result_labels, train_ratio, epochs, batch_size, learning_rate, layers,
                            patience, model_path, is_tying)


def gan(df_train, target_column, train_ratio, glove, token_model, epochs, learning_rate, model_path,
        verbose, global_seed):
    gan_processor = GanProcessor(model_path, global_seed, verbose)
    datas = gan_processor.preprocess(df_train, train_ratio, glove, target_column, token_model)
    gan_processor.train(datas, epochs, learning_rate, token_model, target_column)


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="execute pretraining for input features")
    parser.add_argument("--model_type", type=str, default=None,
                        help="target training model be pretrained [GloVe | DAN | DEC | GAN]")
    parser.add_argument("--epochs", type=int, default=None,
                        help="the number of epochs for training model")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="the number of batch_size for training model")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="learning rate for training model")

    # Train Data
    parser.add_argument("--lzo_numbers", nargs="*", default=None, help="lzo number list be trained")

    # GloVe
    parser.add_argument("--window_size", type=int, default=10,
                        help="[GloVe] window size of co-occurrence matrix for GloVe vector")
    parser.add_argument("--n_components", type=int, default=20,
                        help="[GloVe] the number of dimensionality for GloVe vector")

    # DAN | GAN
    parser.add_argument("--target_column", type=str, default=None,
                        help="[DAN | GAN] training target column [reply | retweet | retweet_c | like]")

    # DEC
    parser.add_argument("--is_tying", type=str2bool, default=True,
                        help="[DEC] condition tying weights of autoenocoder")
    parser.add_argument("--layers", nargs="*", default=["20,64,64,256,16"],
                        help="[DEC] pass the number of dimensionality for GloVe vector")

    # DAN | DEC
    parser.add_argument("--patience", type=int, default=None,
                        help="[DAN | DEC] patience of EarlyStopping for training model")

    # GAN
    parser.add_argument("--token_model", type=str, default=None,
                        help="[GAN] text token model for generating fake data / label [TFIDF | SIF | DAN]")
    # parser.add_argument("--gan_ratio", type=float, default=0.5,
    #                     help="[GAN] Generation Ratio by GAN")

    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="train ratio of training neural network")

    parser.add_argument("--global_seed", type=int, default=42, help="global seed for fixing training process")

    parser.add_argument("--verbose", type=str2bool, default=False, help="condition to print out training process")

    args = parser.parse_args()

    model_type = args.model_type

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    patience = args.patience

    lzo_numbers = args.lzo_numbers
    lzo_numbers = list(map(lambda x: int(x), lzo_numbers[0].split(",")))

    window_size = args.window_size
    n_components = args.n_components

    is_tying = args.is_tying
    layers = args.layers
    layers = list(map(lambda x: int(x), layers[0].split(",")))

    target_column = args.target_column

    token_model = args.token_model
    # gan_ratio = args.gan_ratio

    train_ratio = args.train_ratio
    global_seed = args.global_seed

    verbose = args.verbose

    if train_ratio <= 0.0 and train_ratio > 1.0:
        raise argparse.ArgumentError(train_ratio, "train_ratio must be (0.0, 1.0] at pretrain phase")

    if learning_rate is None:
        raise argparse.ArgumentTypeError(learning_rate, f"define learning_rate for your {model_type} model")

    if lzo_numbers is None:
        raise argparse.ArgumentTypeError(lzo_numbers, "lzo_numbers need to be list of lzo file number")

    lzo_loader = LzoLoader(data_path, verbose)
    df_train = lzo_loader.load_data(lzo_numbers)

    if model_type == "GloVe":
        train_glove(df_train, window_size, n_components, epochs, learning_rate, model_path, verbose)

    elif model_type == "DAN":
        if target_column == None:
            raise argparse.ArgumentTypeError(target_column, f"{model_type} needs to define target column")

        else:
            if target_column == "reply":
                target_column = "reply_timestamp"
            elif target_column == "retweet":
                target_column = "retweet_timestamp"
            elif target_column == "retweet_c":
                target_column = "retweet_with_comment_timestamp"
            elif target_column == "like":
                target_column = "like_timestamp"
            else:
                raise argparse.ArgumentTypeError(target_column,
                                                 f"target column needs to define among [reply | retweet | retweet_c | like] not {target_column}")

        glove = load_glove(model_path)

        dan(df_train, target_column, train_ratio, glove, epochs, batch_size, learning_rate, patience, model_path,
            verbose, global_seed)

        # if train_ratio == 0.0:
        #     return results


    elif model_type == "DEC":
        glove = load_glove(model_path)

        dec(df_train, train_ratio, glove, epochs, batch_size, learning_rate, layers, patience, model_path, verbose,
            global_seed, is_tying)

    elif model_type == "GAN":
        if target_column == None:
            raise argparse.ArgumentTypeError(target_column, f"{model_type} needs to define target column")

        else:
            if target_column == "reply":
                target_column = "reply_timestamp"
            elif target_column == "retweet":
                target_column = "retweet_timestamp"
            elif target_column == "retweet_c":
                target_column = "retweet_with_comment_timestamp"
            elif target_column == "like":
                target_column = "like_timestamp"
            else:
                raise argparse.ArgumentTypeError(target_column,
                                                 f"target column needs to define among [reply | retweet | retweet_c | like] not {target_column}")

        glove = load_glove(model_path)

        gan(df_train, target_column, train_ratio, glove, token_model, epochs, learning_rate, model_path,
            verbose, global_seed)


if __name__ == '__main__':
    main()
