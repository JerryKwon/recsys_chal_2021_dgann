"""
Train Output ANN model
"""
import os
import argparse
import warnings

from data_loader import LzoLoader

from processor import AnnProcessor

from glove import Glove

project_path = os.path.dirname(os.path.abspath("__file__"))
input_path = os.path.join(project_path, "input")
output_path = os.path.join(project_path, "output")

model_path = os.path.join(input_path, "model")
data_path = os.path.join(input_path, "data")
result_path = os.path.join(output_path, "result")


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

    # parser.add_argument("--epochs", type=int, default=None,
    #                     help="the number of epochs for training model")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="the number of batch_size for training model")
    # parser.add_argument("--learning_rate", type=float, default=None,
    #                     help="learning rate for training model")
    parser.add_argument("--token_model", type=str, default=None,
                        help="pass target model be pretrained [tfidf-glove | SIF | DAN]")
    parser.add_argument("--is_gan", type=str2bool, default=True, help="condition to turn on | off gan inference")

    parser.add_argument("--lzo_numbers", type=list, default=None, help="lzo number list be trained")
    parser.add_argument("--target_column", type=str, default=None, help="training target column [reply | retweet | retweet_c | like]")
    # parser.add_argument("--train_ratio", type=float, default=0.8,
    #                     help="train ratio of training neural network")

    parser.add_argument("--global_seed", type=int, default=42, help="global seed for fixing training process")

    parser.add_argument("--verbose", type=str2bool, default=False, help="condition to print out training process")

    args = parser.parse_args()

    batch_size = args.batch_size
    token_model = args.token_model
    is_gan = args.is_gan

    lzo_numbers = args.lzo_numbers
    lzo_numbers = list(map(lambda x: int(x), lzo_numbers[0].split(",")))

    target_column = args.target_column

    global_seed = args.global_seed

    verbose = args.verbose

    lzo_loader = LzoLoader(data_path, verbose)
    df_train = lzo_loader.load_data(lzo_numbers)

    glove = load_glove(model_path)

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


    ann_processor = AnnProcessor(model_path, global_seed, verbose)

    df_indexes, (X, y) = ann_processor.preprocess(df_train, 0.0, glove, token_model, target_col)

    lr_probs, y_true = ann_processor.predict((X, y), target_column, batch_size, token_model, is_gan, model_path)

    result_col = None
    if target_column == "reply_timestamp":
        result_col = "prediction_reply"
    elif target_column == "retweet_timestamp":
        result_col = "prediction_retweet"
    elif target_column == "retweet_with_comment_timestamp":
        result_col = "prediction_quote"
    elif target_column == "like_timestamp":
        result_col = "prediction_fav"

    df_indexes = df_indexes.rename(columns={"tweet_id": "Tweet_Id", "engaging_user_id": "User_ID"})
    df_indexes[result_col] = lr_probs

    f_name = f"{result_col}_{token_model.lower()}_gan.csv" if is_gan else f"{result_col}_{token_model.lower()}_not_gan.csv"

    df_indexes.to_csv(os.path.join(result_path, f_name), index=False)


if __name__ == '__main__':
    main()