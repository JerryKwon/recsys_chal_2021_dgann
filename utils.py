import os
from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment

import torch

from sklearn.metrics import log_loss


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


class EarlyStopping:
    def __init__(self, model_path, patience=7, modes=("max","max"), delta=0.001):
        self.model_path = model_path
        self.patience = patience
        self.counter = 0
        self.modes = modes
        self.dict_best_score = None
        self.early_stop = False
        self.delta = delta

        self.val_scores = [-np.Inf if mode == "max" else np.Inf for mode in modes]
        self.dict_val_scores = {idx: val_score for idx, val_score in enumerate(self.val_scores)}

        self.names = None

    def __call__(self, epoch_scores, model, file_name):

        dict_epoch_scores = dict()

        list_epoch_scores = list(zip(self.modes, epoch_scores))

        for idx, epoch_scores in enumerate(list_epoch_scores):
            mode, epoch_score = epoch_scores
            if mode == "min":
                score = -1.0 * epoch_score
            else:
                score = np.copy(epoch_score)

            dict_epoch_scores[idx] = score

        if self.dict_best_score is None:
            self.dict_best_score = dict_epoch_scores
            self.save_checkpoint(dict_epoch_scores, model, file_name)

        condition = [True if value_best < dict_epoch_scores[key_best] else False for key_best, value_best in self.dict_best_score.items()]

        # dict_best_score 갱신 및 모델 저장
        if sum(condition) == len(self.dict_best_score):
            self.dict_best_score = dict_epoch_scores
            self.save_checkpoint(dict_epoch_scores, model, file_name)
            self.counter = 0

        # patience 증가
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model_state, file_name):
        model_state_dict = deepcopy(model_state)
        total_path = os.path.join(self.model_path, file_name)
        torch.save(model_state_dict, total_path)

    def save_checkpoint(self, dict_epoch_scores, model, file_name):
        condition = [ True if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan] else False for epoch_score in dict_epoch_scores.values() ]

        if sum(condition) == len(condition):
            print("")
            for key, value in self.dict_val_scores.items():
                if self.names is None:
                    print(
                        f"Validation score improved ({value:.4f} --> {dict_epoch_scores[key]:.4f}). Saving model & encoded data")
                else:
                    print(
                        f"{self.names[key]} improved ({value:.4f} --> {dict_epoch_scores[key]:.4f}). Saving model & encoded data")
                # if not DEBUG:
                self.save_model(model.state_dict(), file_name)
                # torch.save(model.state_dict(), model_path)
        self.dict_val_scores = dict_epoch_scores

def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
