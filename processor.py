import os
from copy import deepcopy
import random
from tqdm import tqdm
from datetime import datetime

import pickle

from collections import Counter

from scipy import sparse as spr
import numpy as np
import pandas as pd
from dask import array as da

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

from glove import Glove

# custom package
from utils import EarlyStopping, compute_rce, cluster_accuracy, target_distribution
from models import GLOVE, CustomDAN, Pretrain_AutoEncoder, Tying_AutoEncoder, CustomDEC, CustomGenerator, \
    CustomDiscriminator, CustomANN
from custom_dataset import DANDataset, DecPretrainDataset, DecDataset, GanDataset, AnnDataset


# class EarlyStopping:
#     def __init__(self, model_path, patience=7, mode="max", delta=0.001):
#         self.model_path = model_path
#         self.patience = patience
#         self.counter = 0
#         self.mode = mode
#         self.best_score = None
#         self.early_stop = False
#         self.delta = delta
#         if self.mode == "min":
#             self.val_score = np.Inf
#         else:
#             self.val_score = -np.Inf
#
#     def __call__(self, epoch_score, model, file_name):
#
#         if self.mode == "min":
#             score = -1.0 * epoch_score
#         else:
#             score = np.copy(epoch_score)
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model, file_name)
#
#         elif score <= self.best_score:  # + self.delta
#             self.counter += 1
#             # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             # ema.apply_shadow()
#             self.save_checkpoint(epoch_score, model, file_name)
#             # ema.restore()
#             self.counter = 0
#
#     def save_model(self, model_state, file_name):
#         model_state_dict = deepcopy(model_state)
#         total_path = os.path.join(self.model_path, file_name)
#         torch.save(model_state_dict, total_path)
#
#     def save_checkpoint(self, epoch_score, model, file_name):
#         if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
#             print(
#                 f"Validation score improved ({self.val_score:.4f} --> {epoch_score:.4f}). Saving model & encoded data")
#             # if not DEBUG:
#             self.save_model(model.state_dict(), file_name)
#             # torch.save(model.state_dict(), model_path)
#         self.val_score = epoch_score


class TfidfSifProcessor:
    def __init__(self, df_train, glove, model_path, verbose=False):
        self.df_train = df_train
        self.glove = glove
        self.model_path = model_path
        self.verbose = verbose

        self.embeddings = None
        self.token_to_tid = None
        self.tid_col = None

    def preprocess(self):

        window_size = 10
        epochs = 25
        n_components = 20
        learning_rate = 0.05

        merged_tokens = " ".join(self.df_train["text_ tokens"])
        col = merged_tokens.split(" ")

        if self.glove is not None:

            self.embeddings = self.glove.word_vectors

            self.token_to_tid = self.glove.dictionary

            self.tid_col = [self.token_to_tid[column] for column in col if column in self.token_to_tid.keys()]

        else:

            glove = GLOVE(self.df_train, window_size, n_components, epochs, learning_rate, self.model_path,
                          self.verbose)
            glove.train()

            glove_model = f"glove_{window_size}_{epochs}_{learning_rate}_{n_components}.model"

            try:
                glove = Glove.load(os.path.join(self.model_path, glove_model))
            except FileNotFoundError:
                print(f"There is no pretrained glove model at {self.model_path}")

            self.embeddings = glove.word_vectors

            self.token_to_tid = glove.dictionary

            self.tid_col = col

    def predict(self, predict_type="TFIDF"):

        if predict_type == "TFIDF":
            def text_concatenation(target_str, d_token_to_tid):
                target = [f"{self.token_to_tid[t] + 10}" for t in target_str.split(" ") if t in d_token_to_tid.keys()]

                return " ".join(target)

            # 100만 row당 30초 소요
            tfidf_target = self.df_train["text_ tokens"].apply(lambda x: text_concatenation(x, self.token_to_tid))

            tfidf = TfidfVectorizer()

            tfidf_csr = tfidf.fit_transform(tfidf_target.values)

            result_list = list()

            for row, sentence in (enumerate(tqdm(self.df_train["text_ tokens"], desc="TFIDF processing")) if self.verbose else enumerate(self.df_train["text_ tokens"])):
                sentence = sentence.split(" ")
                sentence = [element for element in sentence if element in self.token_to_tid.keys()]

                glv_converted_token = [self.token_to_tid[token] for token in sentence]
                word_vectors = self.embeddings[glv_converted_token]

                tfidf_converted_token = [tfidf.vocabulary_[str(self.token_to_tid[token] + 10)] for token in sentence]
                word_weights = tfidf_csr[row, tfidf_converted_token]

                weighted_avg = np.average(word_vectors * word_weights.toarray().reshape(-1, 1), axis=0)
                result_list.append(weighted_avg)

            emb_result = np.vstack(result_list)

            return emb_result

        elif predict_type == "SIF":

            a = 0.0001
            entire_corpus = np.array(self.tid_col)
            counter = Counter(entire_corpus)

            matrix = list()

            for row in (tqdm(self.df_train["text_ tokens"].values, desc="SIF processing") if self.verbose else self.df_train["text_ tokens"].values):
                row = row.split(" ")
                row = [element for element in row if element in self.token_to_tid.keys()]

                w_v = [self.embeddings[self.token_to_tid[element]] for element in row]
                weights_w_v = [a / (a + (counter[self.token_to_tid[element]] / len(entire_corpus))) for element in row]
                v_s = np.average(np.array(w_v) * np.array(weights_w_v).reshape(-1, 1), axis=0)

                matrix.append(v_s)

            np_matrix = np.array(matrix)

            da_matrix = da.from_array(np_matrix, chunks=(10000, 10000))

            u, s, v = da.linalg.svd(da_matrix)

            np_sg_vector = u[:, 0].compute()

            np_sg_vector_T = np_sg_vector.reshape(1, -1)

            np_sif_latter = np_sg_vector_T.dot(np_matrix)

            emb_result = np_sg_vector.reshape(-1, 1).dot(np_sif_latter)

            return emb_result

class DanProcessor:
    def __init__(self, df_train, glove, model_path, verbose=False):
        self.df_train = df_train
        self.glove = glove
        self.model_path = model_path
        self.verbose = verbose

        self.embeddings = glove.word_vectors
        self.embeddings_w0 = self.get_embeddings_w0()
        self.token_to_tid = glove.dictionary

    def seed_initialization(self, GLOBAL_SEED):
        self.GLOBAL_SEED = GLOBAL_SEED

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

    def get_embeddings_w0(self):
        return np.vstack([np.zeros(self.embeddings.shape[1]), self.embeddings])

    def token_to_tid_count(self, target_str):
        result = 0
        for token in target_str.split(" "):
            if token in self.token_to_tid.keys():
                result += 1
        return result

    def preprocess(self, target_column, batch_size, train_ratio=0.8, GLOBAL_SEED=None):
        if GLOBAL_SEED is not None:
            self.seed_initialization(GLOBAL_SEED)

        df_train = self.df_train

        self.np_tid_token_len = df_train["text_ tokens"].apply(self.token_to_tid_count).values

        added_values = np.fromiter(self.token_to_tid.values(), dtype=np.long) + 1
        token_to_tid_add = dict(zip(list(self.token_to_tid.keys()), added_values))

        rows = list()
        cols = list()
        datas = list()

        for idx, target_str in (enumerate(tqdm(df_train["text_ tokens"].values, desc="csr matrix processing")) if self.verbose else enumerate(
                df_train["text_ tokens"].values)):
            mid_datas = [token_to_tid_add[token] for token in target_str.split(" ") if
                         token in self.token_to_tid.keys()]
            mid_rows = np.repeat(idx, len(mid_datas))
            mid_cols = np.arange(len(mid_datas))

            rows.append(mid_rows)
            cols.append(mid_cols)
            datas.append(mid_datas)

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(datas)

        csr_matrix = spr.csr_matrix((data, (row, col)), shape=(df_train.shape[0], self.np_tid_token_len.max()))

        np_token = csr_matrix.toarray()

        target_label = np.fromiter(map(lambda x: 0 if x == "" else 1, df_train[target_column].values), dtype=np.int)

        oversample = None

        if target_column == "like_timestamp":
            oversample = False
        else:
            oversample = True

        if train_ratio == 0.0:
            dataloader = self.make_loader(np_token, target_label, train_ratio, batch_size, False)
        else:
            dataloader = self.make_loader(np_token, target_label, train_ratio, batch_size, oversample)

        return dataloader

    def make_loader(self, np_token, target_label, train_ratio,
                    BATCH_SIZE=50000, oversample=True):

        dan_dataset = DANDataset(np_token, self.np_tid_token_len, target_label)

        if train_ratio == 1.0 or train_ratio == 0.0:
            if oversample:
                target_count = np.array([len(np.where(target_label == t)[0]) for t in [0, 1]])
                target_weight = 1. / target_count
                target_weight = np.array([target_weight[0], target_weight[1]])
                target_weight = np.array([target_weight[t] for t in target_label])
                target_weight = torch.from_numpy(target_weight).double()
                target_sampler = WeightedRandomSampler(target_weight, len(target_weight))

                if train_ratio == 1.0:
                    trn_dataloader = torch.utils.data.DataLoader(
                        dan_dataset,
                        batch_size=BATCH_SIZE,
                        sampler=target_sampler
                    )

                return trn_dataloader

            else:
                val_dataloader = torch.utils.data.DataLoader(
                    dan_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False
                )
                return val_dataloader

        else:
            trn_indices, val_indices = train_test_split(np.arange(np_token.shape[0]),
                                                                    train_size=train_ratio,
                                                                    stratify=target_label, random_state=self.GLOBAL_SEED)
            trn_dataset = Subset(dan_dataset, indices=trn_indices)
            val_dataset = Subset(dan_dataset, indices=val_indices)

            if oversample:
                trn_label = target_label[trn_indices]
                target_count = np.array([len(np.where(trn_label == t)[0]) for t in [0, 1]])
                target_weight = 1. / target_count
                target_weight = np.array([target_weight[0], target_weight[1]])
                target_weight = np.array([target_weight[t] for t in trn_label])
                target_weight = torch.from_numpy(target_weight).double()
                target_sampler = WeightedRandomSampler(target_weight, len(target_weight))

                trn_dataloader = torch.utils.data.DataLoader(
                    trn_dataset,
                    batch_size=BATCH_SIZE,
                    sampler=target_sampler
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False
                )

                return (trn_dataloader, val_dataloader)

            else:
                trn_dataset = Subset(dan_dataset, indices=trn_indices)
                val_dataset = Subset(dan_dataset, indices=val_indices)

                trn_dataloader = torch.utils.data.DataLoader(
                    trn_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False
                )

                return (trn_dataloader, val_dataloader)

    def train(self, dataloaders, target_column, epochs, learning_rate, model_path, patience=None):

        f_name = None

        if target_column == "reply_timestamp":
            f_name = "dan_reply_over.pth"
        elif target_column == "retweet_timestamp":
            f_name = "dan_retweet_over.pth"
        elif target_column == "retweet_with_comment_timestamp":
            f_name = "dan_retweet_c_over.pth"
        elif target_column == "like_timestamp":
            f_name = "dan_like.pth"

        is_valid = len(dataloaders) == 2

        if is_valid:
            trn_dataloader, val_dataloader = dataloaders

        else:
            trn_dataloader = dataloaders

        section = 3
        batch_prints = [int(len(trn_dataloader) / section * idx) for idx in range(1, section + 1)][:-1]

        es = None
        print(patience)
        if patience > 0:
            es = EarlyStopping(model_path, patience, "max")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        dan = CustomDAN(self.embeddings_w0, DEVICE)

        optimizer = optim.SGD(dan.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        if DEVICE == "cuda":
            dan = dan.to(DEVICE)

        for epoch in (tqdm(range(epochs), desc="Dan train processing") if self.verbose else range(epochs)):

            dan.train()

            trn_loss = 0.0
            trn_rce = 0.0
            trn_ap = 0.0

            for batch_idx, (inputs, len_idx, labels) in enumerate(trn_dataloader):
                dan.zero_grad()

                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                hidden, output = dan((inputs, len_idx))

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                trn_rce += compute_rce(output.detach().cpu().numpy()[:,1], labels.detach().cpu().numpy())
                trn_ap += average_precision_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy()[:,1])
                trn_loss += loss.item()

                if self.verbose:
                    if (batch_idx + 1) in batch_prints:
                        print(
                            f"BATCH:{batch_idx + 1}:{len(trn_dataloader)}; loss:{trn_loss / (batch_idx + 1):.4f}; ap:{trn_ap / (batch_idx + 1):.4f}; rce:{trn_rce / (batch_idx + 1):.4f}")

            if not is_valid:
                dan_state_dict = deepcopy(dan.state_dict())
                torch.save(dan_state_dict, os.path.join(model_path, f_name))
                if self.verbose:
                    print(
                        f"EPOCH:{epoch + 1}|{epochs}; loss:{trn_loss / len(trn_dataloader):.4f}; ap:{trn_ap / len(trn_dataloader):.4f}; rce:{trn_rce / len(trn_dataloader):.4f}")

            else:
                val_loss = 0.0
                val_rce = 0.0
                val_ap = 0.0

                dan.eval()

                with torch.no_grad():
                    for batch_idx, (inputs, len_idx, labels) in enumerate(val_dataloader):

                        if DEVICE == "cuda":
                            inputs = inputs.to(DEVICE)
                            labels = labels.to(DEVICE)

                        hidden, output = dan((inputs, len_idx))
                        loss = criterion(output, labels)

                        val_rce += compute_rce(output.detach().cpu().numpy()[:,1], labels.detach().cpu().numpy())
                        val_ap += average_precision_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy()[:,1])
                        val_loss += loss.item()

                        break

                if self.verbose:
                    print(
                        f"EPOCH:{epoch + 1}|{epochs}; loss:{trn_loss / len(trn_dataloader):.4f}/{val_loss:.4f}; ap:{trn_ap / len(trn_dataloader):.4f}/{val_ap}; rce:{trn_rce / len(trn_dataloader):.4f}/{val_rce:.4f}")

                if patience > 0:
                    es((val_ap), dan, f_name)

                if es.early_stop:
                    print("early_stopping")
                    break

    def predict(self, dataloader, target_column, model_path):

        result = []

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.CrossEntropyLoss()

        dan = CustomDAN(self.embeddings_w0, DEVICE)

        f_name = None

        if target_column == "reply_timestamp":
            f_name = "dan_reply_over.pth"
        elif target_column == "retweet_timestamp":
            f_name = "dan_retweet_over.pth"
        elif target_column == "retweet_with_comment_timestamp":
            f_name = "dan_retweet_c_over.pth"
        elif target_column == "like_timestamp":
            f_name = "dan_like.pth"

        dan.load_state_dict(torch.load(os.path.join(model_path,f_name)))

        dan = dan.to(DEVICE)

        dan.eval()

        val_loss = 0.0
        val_rce = 0.0
        val_ap = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, len_idx, labels) in (
                    enumerate(tqdm(dataloader, desc="Dan inference processing")) if self.verbose else enumerate(dataloader)):

                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                hidden, output = dan((inputs, len_idx))

                result.append(hidden.detach().cpu().numpy())

                loss = criterion(output, labels)

                val_rce += compute_rce(output.detach().cpu().numpy()[:,1], labels.detach().cpu().numpy())
                val_ap += average_precision_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy()[:,1])
                val_loss += loss.item()

        if self.verbose:
            print(
                f"loss:{val_loss / len(dataloader):.4f}; ap:{val_ap / len(dataloader):.4f}; rce:{val_rce / len(dataloader):.4f}")

        result = np.vstack(result)

        return result


class DecProcessor:
    def __init__(self, df_train, glove, model_path, verbose=False):
        self.df_train = df_train
        self.glove = glove
        self.model_path = model_path
        self.verbose = verbose

        self.embeddings = glove.word_vectors
        self.embeddings_w0 = self.get_embeddings_w0()
        self.token_to_tid = glove.dictionary

    def seed_initialization(self, GLOBAL_SEED):
        self.GLOBAL_SEED = GLOBAL_SEED

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

    def get_embeddings_w0(self):
        return np.vstack([np.zeros(self.embeddings.shape[1]), self.embeddings])

    def check_pretrain(self, is_tying):

        f_models = None

        if is_tying:
            f_models = ["t_h1_rec.pth", "t_h2_rec.pth", "t_h3_rec.pth", "t_h4_rec.pth", "t_total_rec.pth"]
        else:
            f_models = ["nt_h1_rec.pth", "nt_h2_rec.pth", "nt_h3_rec.pth", "nt_h4_rec.pth", "nt_total_rec.pth"]

        f_models_ind = [os.path.isfile(os.path.join(self.model_path, f_model)) for f_model in f_models]

        if sum(f_models_ind) == len(f_models):
            return True
        else:
            return False

    def token_to_tid_count(self, target_str):
        result = 0
        for token in target_str.split(" "):
            if token in self.token_to_tid.keys():
                result += 1
        return result

    def preprocess(self, GLOBAL_SEED=None):
        if GLOBAL_SEED is not None:
            self.seed_initialization(GLOBAL_SEED)

        df_train = self.df_train

        self.np_tid_token_len = df_train["text_ tokens"].apply(self.token_to_tid_count).values

        added_values = np.fromiter(self.token_to_tid.values(), dtype=np.long) + 1
        token_to_tid_add = dict(zip(list(self.token_to_tid.keys()), added_values))

        rows = list()
        cols = list()
        datas = list()

        for idx, target_str in (enumerate(tqdm(df_train["text_ tokens"].values,desc="csr matrix processing")) if self.verbose else enumerate(
                df_train["text_ tokens"].values)):
            mid_datas = [token_to_tid_add[token] for token in target_str.split(" ") if
                         token in self.token_to_tid.keys()]
            mid_rows = np.repeat(idx, len(mid_datas))
            mid_cols = np.arange(len(mid_datas))

            rows.append(mid_rows)
            cols.append(mid_cols)
            datas.append(mid_datas)

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.concatenate(datas)

        csr_matrix = spr.csr_matrix((data, (row, col)), shape=(df_train.shape[0], self.np_tid_token_len.max()))

        np_token = csr_matrix.toarray()

        embedded_mean_token = list()

        for idx, target in (enumerate(tqdm(np_token, desc="concat word_emb by mean")) if self.verbose else enumerate(np_token)):
            len_idx = self.np_tid_token_len[idx]
            embedded_mean_token.append(self.embeddings_w0[target[:len_idx]].mean(axis=0))

        embedded_mean_token = np.vstack(embedded_mean_token)

        scaler = MinMaxScaler()
        scaled_embedded_mean_token = scaler.fit_transform(embedded_mean_token)

        list_labels = [tuple(list(map(lambda x: int(x), list(format(idx, "04b"))))) for idx in range(2 ** 4)]

        dict_labels_to_idx = {idx: values for idx, values in enumerate(list_labels)}
        dict_idx_to_labels = {values: idx for idx, values in dict_labels_to_idx.items()}

        np_labels = df_train.loc[:, df_train.columns[-4:]].values
        np_labels = np.apply_along_axis(lambda x: x != "", 1, np_labels).astype(np.int)
        list_target_labels = list(map(lambda x: tuple(x), np_labels))
        np_result_labels = np.array([dict_idx_to_labels[labels] for labels in list_target_labels])

        return scaled_embedded_mean_token, np_result_labels

    def pretrain(self, np_token, np_labels, layers, model_path, data_path, is_tying):

        if self.verbose:
            print("Pretraining AutoEncoder...")

        epochs = 100
        batch_size = 50000
        train_ratio = 0.8
        learning_rate = 0.1

        dec_p_dataset = DecPretrainDataset(np_token)

        dec_trn_indices, dec_val_indices = train_test_split(np.arange(np_token.shape[0]), train_size=train_ratio,
                                                            stratify=np_labels, random_state=self.GLOBAL_SEED)

        dec_trn_dataset = Subset(dec_p_dataset, indices=dec_trn_indices)
        dec_val_dataset = Subset(dec_p_dataset, indices=dec_val_indices)

        h1_ae = Pretrain_AutoEncoder(layers[0], layers[1], is_first=True, is_last=True)
        h2_ae = Pretrain_AutoEncoder(layers[1], layers[2], is_first=True, is_last=True)
        h3_ae = Pretrain_AutoEncoder(layers[2], layers[3], is_first=True, is_last=True)
        h4_ae = Pretrain_AutoEncoder(layers[3], layers[4], is_first=True, is_last=True)

        if is_tying:
            h1_ae = Tying_AutoEncoder(h1_ae, is_first=True, is_last=True)
            h2_ae = Tying_AutoEncoder(h2_ae, is_last=True)
            h3_ae = Tying_AutoEncoder(h3_ae, is_last=True)
            h4_ae = Tying_AutoEncoder(h4_ae, is_last=True)

        datasets = self.pretrain_greedy_layer(h1_ae, (dec_trn_dataset, dec_val_dataset), epochs, batch_size,
                                              learning_rate, model_path, data_path, 1, is_tying)
        datasets = self.pretrain_greedy_layer(h2_ae, datasets, epochs, batch_size, learning_rate,
                                              model_path, data_path, 2, is_tying)
        datasets = self.pretrain_greedy_layer(h3_ae, datasets, epochs, batch_size, learning_rate,
                                              model_path, data_path, 3, is_tying)
        datasets = self.pretrain_greedy_layer(h4_ae, datasets, epochs, batch_size, learning_rate,
                                              model_path, data_path, 4, is_tying)

        self.pretrain_whole_layer(h4_ae, layers, (dec_trn_dataset, dec_val_dataset), epochs, batch_size, learning_rate,
                                  model_path, 4, is_tying)

    def make_dataloader(self, dataset, batch_size, is_train):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train
        )

        return dataloader

    def pretrain_greedy_layer(self, ae, datasets, epochs, batch_size, learning_rate, model_path, data_path, place,
                              is_tying):
        trn_encoded = None
        val_encoded = None

        section = 5 if place == 5 else 3
        fname = f"{'t' if is_tying else 'nt'}_h{place}_rec.pth"

        epoch_decreases = [int(epochs / section * idx) for idx in range(1, section + 1)][:-1]

        trn_dataset, val_dataset = datasets

        trn_dataloader = self.make_dataloader(trn_dataset, batch_size, True)
        val_dataloader = self.make_dataloader(val_dataset, batch_size, False)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        lr = learning_rate
        optimizer = optim.SGD(ae.parameters(), lr=lr, momentum=0.9)
        criterion = nn.MSELoss()

        for epoch in (tqdm(range(epochs), desc="greedy-wise pretrain processing") if self.verbose else range(epochs)):

            if epoch in epoch_decreases:
                lr /= 10
                optimizer = optim.SGD(ae.parameters(), lr=lr, momentum=0.9)

            if place < 4:
                list_trn_encoded = list()
                list_val_encoded = list()

            ae.train()

            trn_loss = 0.0

            for batch_idx, inputs in enumerate(trn_dataloader):
                ae.zero_grad()

                inputs = inputs.to(DEVICE)

                encoded, decoded = ae(inputs)
                if place < 4:
                    list_trn_encoded.append(encoded.detach().cpu().numpy())
                loss = criterion(decoded, inputs.detach())
                loss.backward()
                optimizer.step()

                trn_loss += loss

            if place < 4:
                trn_encoded = list_trn_encoded

            val_loss = 0.0

            ae.eval()

            with torch.no_grad():
                for batch_idx, inputs in enumerate(val_dataloader):
                    inputs = inputs.to(DEVICE)

                    encoded, decoded = ae(inputs)
                    if place < 4:
                        list_val_encoded.append(encoded.detach().cpu().numpy())
                    loss = criterion(decoded, inputs.detach())

                    val_loss += loss

            if place < 4:
                val_encoded = list_val_encoded

            print(
                f"EPOCH:{epoch + 1}|{epochs}; loss:{trn_loss / len(trn_dataloader):.4f}/{val_loss / len(val_dataloader):.4f}")

            torch.save(deepcopy(ae.state_dict()), os.path.join(model_path, fname))

        torch.cuda.empty_cache()

        if place < 4:

            trn_encoded_fname = f"{'t' if is_tying else 'nt'}_h{place}_rec_trn_encoded.npy"
            val_encoded_fname = f"{'t' if is_tying else 'nt'}_h{place}_rec_val_encoded.npy"

            np_train = np.vstack(trn_encoded)
            np_valid = np.vstack(val_encoded)

            np.save(os.path.join(data_path, trn_encoded_fname), np_train)
            np.save(os.path.join(data_path, val_encoded_fname), np_valid)

            trn_encoded_dataset = DecPretrainDataset(np_train)
            val_encoded_dataset = DecPretrainDataset(np_valid)

            return (trn_encoded_dataset, val_encoded_dataset)

        else:
            return None

    def pretrain_whole_layer(self, aes, layers, datasets, epochs, batch_size, learning_rate, model_path,
                             is_tying):

        trn_dataset, val_dataset = datasets

        trn_dataloader = self.make_dataloader(trn_dataset, batch_size, True)
        val_dataloader = self.make_dataloader(val_dataset, batch_size, False)

        total_ae = Pretrain_AutoEncoder(layers, is_first=True, is_last=True)

        model_h1, model_h2, model_h3, model_h4 = aes

        h1_file = os.path.join(model_path, f"{'t' if is_tying else 'nt'}_h1_rec.pth")
        h2_file = os.path.join(model_path, f"{'t' if is_tying else 'nt'}_h2_rec.pth")
        h3_file = os.path.join(model_path, f"{'t' if is_tying else 'nt'}_h3_rec.pth")
        h4_file = os.path.join(model_path, f"{'t' if is_tying else 'nt'}_h4_rec.pth")

        model_h1.load_state_dict(torch.load(h1_file))
        model_h2.load_state_dict(torch.load(h2_file))
        model_h3.load_state_dict(torch.load(h3_file))
        model_h4.load_state_dict(torch.load(h4_file))

        total_ae.encoder[0].block[0].weight = model_h1.encoder[0].block[0].weight

        total_ae.encoder[0].block[2].weight = model_h1.encoder[0].block[2].weight
        total_ae.encoder[0].block[3].weight = model_h1.encoder[0].block[3].weight

        total_ae.encoder[1].block[0].weight = model_h2.encoder[0].block[0].weight
        total_ae.encoder[1].block[1].weight = model_h2.encoder[0].block[1].weight

        total_ae.encoder[2].block[0].weight = model_h3.encoder[0].block[0].weight
        total_ae.encoder[2].block[1].weight = model_h3.encoder[0].block[1].weight

        total_ae.encoder[3].block[0].weight = model_h4.encoder[0].block[0].weight
        total_ae.encoder[3].block[1].weight = model_h4.encoder[0].block[1].weight

        if is_tying:
            total_ae = Tying_AutoEncoder(total_ae, is_first=True)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # lr = learning_rate
        optimizer = optim.SGD(total_ae.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.MSELoss()

        if DEVICE == "cuda":
            total_ae = total_ae.to(DEVICE)

        total_path = os.path.join(model_path, f"{'t' if is_tying else 'nt'}_total_rec.pth")

        for epoch in (tqdm(range(epochs), desc="whole ae pretrain processing") if self.verbose else range(epochs)):

            if epoch % 40 == 0:
                learning_rate /= 10
                optimizer = optim.SGD(total_ae.parameters(), lr=learning_rate, momentum=0.9)

            total_ae.train()

            trn_loss = 0.0

            for batch_idx, inputs in enumerate(trn_dataloader):
                total_ae.zero_grad()

                inputs = inputs.to(DEVICE)

                encoded, decoded = total_ae(inputs)
                loss = criterion(decoded, inputs.detach())
                loss.backward()
                optimizer.step()

                trn_loss += loss

            val_loss = 0.0

            total_ae.eval()

            with torch.no_grad():
                for batch_idx, inputs in enumerate(val_dataloader):
                    inputs = inputs.to(DEVICE)

                    encoded, decoded = total_ae(inputs)
                    loss = criterion(decoded, inputs.detach())

                    val_loss += loss

            print(
                f"EPOCH:{epoch + 1}|{epochs}; loss:{trn_loss / len(trn_dataloader):.4f}/{val_loss / len(val_dataloader):.4f}")

            torch.save(deepcopy(total_ae.state_dict()), total_path)

    def train(self, np_token, np_labels, train_ratio, epochs, batch_size, learning_rate, layers, patience, model_path, is_tying):

        if self.verbose:
            print("Training DEC...")

        dec_dataset = DecDataset(np_token, np_labels)

        if train_ratio == 1.0:
            dec_trn_dataset = dec_dataset
            # dec_val_dataset = None

        else:
            dec_trn_indices, dec_val_indices = train_test_split(np.arange(np_token.shape[0]), train_size=train_ratio,
                                                            stratify=np_labels, random_state=self.GLOBAL_SEED)

            dec_trn_dataset = Subset(dec_dataset, indices=dec_trn_indices)
            # dec_val_dataset = Subset(dec_dataset, indices=dec_val_indices)

        train_dataloader = self.make_dataloader(dec_trn_dataset, batch_size, True)
        # valid_dataloader = self.make_dataloader(dec_val_dataset, batch_size, True)

        total_ae = Pretrain_AutoEncoder(layers, is_first=True, is_last=True)

        if is_tying:
            total_ae = Tying_AutoEncoder(total_ae, is_first=True, is_last=True)

        total_ae_fname = f"{'t' if is_tying else 'nt'}_total_rec.pth"
        total_ae.load_state_dict(torch.load(os.path.join(model_path, total_ae_fname)))

        dec_encoder = total_ae.encoder

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        dec = CustomDEC(16, 16, dec_encoder)

        # stopping_delta = 0.1

        kmeans = KMeans(n_clusters=16, n_init=20)

        if DEVICE == "cuda":
            dec = dec.to(DEVICE)

        es = None
        if patience > 0:
            es = EarlyStopping(model_path, patience, "max")

        dec.train()

        features = []
        actual = []

        for batch_num, (inputs, labels) in (enumerate(tqdm(train_dataloader, desc="train dec_encoder processing")) if self.verbose else enumerate(train_dataloader)):
            if DEVICE == "cuda":
                inputs = inputs.to(DEVICE, non_blocking=True)

            features.append(dec.encoder(inputs).detach().cpu().numpy())
            actual.append(labels.detach().cpu().numpy())

        actual = np.hstack(actual)
        features = np.vstack(features)
        predicted = kmeans.fit_predict(features)
        predicted_previous = predicted
        _, accuracy = cluster_accuracy(actual, predicted)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)

        if DEVICE == "cuda":
            cluster_centers = cluster_centers.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            dec.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
        criterion = nn.KLDivLoss(size_average=False)
        optimizer = optim.SGD(dec.parameters(), lr=learning_rate, momentum=0.9)
        # delta_label = None

        for epoch in (tqdm(range(epochs), desc="train dec processing") if self.verbose else range(epochs)):

            running_loss = 0.0

            dec.train()
            for batch_num, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()

                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE, non_blocking=True)

                output = dec(inputs)
                target = target_distribution(output).detach()
                loss = criterion(output.log(), target) / output.shape[0]
                running_loss += loss.detach().cpu().item()

                loss.backward()
                optimizer.step(closure=None)

            dec.eval()
            with torch.no_grad():
                trn_predicted = []
                trn_predicted_actual = []

                for batch_num, (inputs, labels) in enumerate(train_dataloader):
                    if DEVICE == "cuda":
                        inputs = inputs.to(DEVICE, non_blocking=True)
                        # labels = labels.to(DEVICE)

                    trn_predicted.append(dec(inputs).detach().cpu().numpy().argmax(axis=1))
                    trn_predicted_actual.append(labels.detach().numpy())

                trn_predicted = np.hstack(trn_predicted)
                trn_predicted_actual = np.hstack(trn_predicted_actual)

            # delta_label = (
            #         float((trn_predicted != predicted_previous).sum()) / predicted_previous.shape[0]
            # )
            #
            # if stopping_delta is not None and delta_label < stopping_delta:
            #     print(
            #     )
            #         'Early stopping as label delta "%1.5f" less than "%1.5f".'
            #         % (delta_label, stopping_delta)

            predicted_previous = trn_predicted

            trn_, trn_accuracy = cluster_accuracy(trn_predicted_actual, trn_predicted)

            if self.verbose:
                print(
                    f"EPOCH:{epoch + 1}|{epochs}; loss:{running_loss / len(train_dataloader):.4f}; CLUSTER_ACC:{trn_accuracy:.4f}")

            if patience > 0:
                es((trn_accuracy), dec, "t_dec_rec.pth")

            if es.early_stop:
                print("early_stopping")
                break

    def predict(self, np_token, batch_size, layers, model_path, is_tying):
        dec_dataset = DecDataset(np_token, None)

        dataloader = self.make_dataloader(dec_dataset, batch_size, True)

        total_ae = Pretrain_AutoEncoder(layers, is_first=True, is_last=True)

        if is_tying:
            total_ae = Tying_AutoEncoder(total_ae, is_first=True, is_last=True)

        dec_encoder = total_ae.encoder
        dec = CustomDEC(16, 16, dec_encoder)

        dec.load_state_dict(torch.load(os.path.join(model_path, "t_dec_rec.pth")))

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        if DEVICE == "cuda":
            dec.to(DEVICE)

        dec.eval()
        with torch.no_grad():
            predicted = []

            for batch_num, inputs in (enumerate(tqdm(dataloader, desc="inference dec processing")) if self.verbose else enumerate(dataloader)):
                if DEVICE == "cuda":
                    inputs = inputs.to(DEVICE, non_blocking=True)

                predicted.append(dec(inputs).detach().cpu().numpy().argmax(axis=1))

            predicted = np.hstack(predicted)
            return predicted

class GanProcessor:
    def __init__(self, model_path, GLOBAL_SEED, verbose=False):
        self.model_path = model_path
        self.verbose = verbose

        self.seed_initialization(GLOBAL_SEED)

    def seed_initialization(self, GLOBAL_SEED):

        self.GLOBAL_SEED = GLOBAL_SEED

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

    def get_token_feature(self, df_train, glove, target_col, token_model):
        if token_model == "TFIDF":
            processor = TfidfSifProcessor(df_train, glove, self.model_path, self.verbose)
            processor.preprocess()
            emb_result = processor.predict(token_model)

        elif token_model == "SIF":
            processor = TfidfSifProcessor(df_train, glove, self.model_path, self.verbose)
            processor.preprocess()
            emb_result = processor.predict(token_model)

        elif token_model == "DAN":
            processor = DanProcessor(df_train, glove, self.model_path, self.verbose)
            dataloader = processor.preprocess(target_col, 5000, 0.0, self.GLOBAL_SEED)
            emb_result = processor.predict(dataloader, target_col, self.model_path)
            torch.cuda.empty_cache()

        return emb_result

    def preprocess(self, df_train, train_ratio, glove, target_col, token_model="TFIDF"):

        self.df_train = df_train
        self.glove = glove
        # self.token_model = token_model

        # self.text_processor = TfidfSifProcessor(df_train, glove, self.model_path, self.verbose)

        data = df_train.copy()

        tweet_hour = []

        for tweet_time in data['tweet_timestamp'].astype(int).values:
            tweet_hour.append(datetime.fromtimestamp(tweet_time).strftime('%H'))

        data['engaged_time > engaging_time'] = ((pd.to_numeric(
            data['engaged_with_user_account_creation']) - pd.to_numeric(
            data['engaging_user_account_creation'])) / 1000000).astype(int)
        data['tweet_time > engaging_time'] = ((pd.to_numeric(data['tweet_timestamp']) - pd.to_numeric(
            data['engaging_user_account_creation'])) / 1000000).astype(int)
        data['tweet_hour'] = tweet_hour
        data['tweet_hour'] = data['tweet_hour'].astype(int)

        data.loc[data["engaged_time > engaging_time"] >= 0, "bool_(engaged_time > engaging_time)"] = 1
        data.loc[data["engaged_time > engaging_time"] < 0, "bool_(engaged_time > engaging_time)"] = 0

        data['num_GIF'] = data['present_media'].str.count('GIF')
        data['num_Video'] = data['present_media'].str.count('Video')
        data['num_Photo'] = data['present_media'].str.count('Photo')

        data['num_GIF'] = data['num_GIF'].fillna(0)
        data['num_Video'] = data['num_Video'].fillna(0)
        data['num_Photo'] = data['num_Photo'].fillna(0)

        data.loc[data["like_timestamp"].str.len() >= 1, "like_timestamp"] = 1
        data.loc[data["like_timestamp"] != 1, "like_timestamp"] = 0

        data.loc[data["retweet_timestamp"].str.len() >= 1, "retweet_timestamp"] = 1
        data.loc[data["retweet_timestamp"] != 1, "retweet_timestamp"] = 0

        data.loc[data["retweet_with_comment_timestamp"].str.len() >= 1, "retweet_with_comment_timestamp"] = 1
        data.loc[data["retweet_with_comment_timestamp"] != 1, "retweet_with_comment_timestamp"] = 0

        data.loc[data["reply_timestamp"].str.len() >= 1, "reply_timestamp"] = 1
        data.loc[data["reply_timestamp"] != 1, "reply_timestamp"] = 0

        data.loc[data["present_links"] != 0, "present_links"] = 1

        data_X = data.loc[:, ['tweet_type', 'engaged_with_user_is_verified', 'engaging_user_is_verified'
                                 , 'engagee_follows_engager', 'engaged_with_user_following_count',
                              'engaged_with_user_follower_count'
                                 , 'engaging_user_follower_count', 'engaging_user_following_count', 'present_links',
                              'engaged_time > engaging_time', 'tweet_time > engaging_time', 'tweet_hour', 'language',
                              'bool_(engaged_time > engaging_time)', 'num_GIF', 'num_Video', 'num_Photo']]

        data_X.loc[data["engaged_with_user_is_verified"] == 'false', "engaged_with_user_is_verified"] = 0
        data_X.loc[data["engaged_with_user_is_verified"] == 'true', "engaged_with_user_is_verified"] = 1
        data_X.loc[data["engaging_user_is_verified"] == 'false', "engaging_user_is_verified"] = 0
        data_X.loc[data["engaging_user_is_verified"] == 'true', "engaging_user_is_verified"] = 1
        data_X.loc[data["engagee_follows_engager"] == 'false', "engagee_follows_engager"] = 0
        data_X.loc[data["engagee_follows_engager"] == 'true', "engagee_follows_engager"] = 1

        data_X.loc[data_X["language"].str.contains('488B32D24BD4BB44172EB981C1BCA6FA', na=False), "language"] = '10'
        data_X.loc[data_X["language"].str.contains('E7F038DE3EAD397AEC9193686C911677', na=False), "language"] = '9'
        data_X.loc[data_X["language"].str.contains('B0FA488F2911701DD8EC5B1EA5E322D8', na=False), "language"] = '8'
        data_X.loc[data_X["language"].str.contains('B8B04128918BBF54E2E178BFF1ABA833', na=False), "language"] = '7'
        data_X.loc[data_X["language"].str.contains('313ECD3A1E5BB07406E4249475C2D6D6', na=False), "language"] = '6'
        data_X.loc[data_X["language"].str.contains('1F73BB863A39DB62B4A55B7E558DB1E8', na=False), "language"] = '5'
        data_X.loc[data_X["language"].str.contains('9FCF19233EAD65EA6E32C2E6DC03A444', na=False), "language"] = '4'
        data_X.loc[data_X["language"].str.contains('9A78FC330083E72BE0DD1EA92656F3B5', na=False), "language"] = '3'
        data_X.loc[data_X["language"].str.contains('8729EBF694C3DAF61208A209C2A542C8', na=False), "language"] = '2'
        data_X.loc[data_X["language"].str.contains('E6936751CBF4F921F7DE1AEF33A16ED0', na=False), "language"] = '1'
        data_X.loc[data_X["language"].str.len() > 3, "language"] = '0'

        data_X['engaged_with_user_following_count'] = data_X['engaged_with_user_following_count'].astype(int)
        data_X['engaged_with_user_follower_count'] = data_X['engaged_with_user_follower_count'].astype(int)
        data_X['engaging_user_follower_count'] = data_X['engaging_user_follower_count'].astype(int)
        data_X['engaging_user_following_count'] = data_X['engaging_user_following_count'].astype(int)

        data_X['engaged_time'] = (pd.to_numeric(data['engaged_with_user_account_creation']) / 1000000)

        data_X['engaging_time'] = (pd.to_numeric(data['engaging_user_account_creation']) / 1000000)
        data_X['tweet_timestamp'] = (pd.to_numeric(data['tweet_timestamp']) / 1000000)

        emb_result = self.get_token_feature(df_train, glove, target_col, token_model)

        sentence_vectors = ['sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2', 'sentenc_vector_3',
                            'sentenc_vector_4', 'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7'
            , 'sentenc_vector_8', 'sentenc_vector_9', 'sentenc_vector_10', 'sentenc_vector_11'
            , 'sentenc_vector_12', 'sentenc_vector_13', 'sentenc_vector_14', 'sentenc_vector_15'
            , 'sentenc_vector_16', 'sentenc_vector_17', 'sentenc_vector_18', 'sentenc_vector_19']

        data_X[sentence_vectors] = emb_result

        data_X['engaging_time'] = (pd.to_numeric(data['engaging_user_account_creation']) / 1000000)
        data_X['tweet_timestamp'] = (pd.to_numeric(data['tweet_timestamp']) / 1000000)

        # data_X[data_X['engaging_user_following_count'] == 0] = 1
        # data_X[data_X['engaged_with_user_following_count'] == 0] = 1
        #
        # data_X[data_X['engaging_user_follower_count'] == 0] = 1
        # data_X[data_X['engaged_with_user_follower_count'] == 0] = 1

        data_X["engaging_user_following_count"][data_X['engaging_user_following_count'] == 0] = 1
        data_X["engaged_with_user_following_count"][data_X['engaged_with_user_following_count'] == 0] = 1

        data_X["engaging_user_follower_count"][data_X['engaging_user_follower_count'] == 0] = 1
        data_X["engaged_with_user_follower_count"][data_X['engaged_with_user_follower_count'] == 0] = 1

        data_X['ratio_enaging_follow'] = data_X['engaging_user_follower_count'] / data_X[
            'engaging_user_following_count']
        data_X['ratio_enagaged_follow'] = data_X['engaged_with_user_follower_count'] / data_X[
            'engaged_with_user_following_count']

        for i in range(0, 1):
            le = LabelEncoder()
            y = list(data_X.iloc[:, i])

            le.fit(y)
            y2 = le.transform(y)

            data_X.iloc[:, i] = y2

        dict_label_encoder = dict(zip(y,y2))

        if target_col == "reply_timestamp":
            f_name = "gan_dict_label_encoder_reply.pkl"
        elif target_col == "retweet_timestamp":
            f_name = "gan_dict_label_encoder_retweet.pkl"
        elif target_col == "retweet_with_comment_timestamp":
            f_name = "gan_dict_label_encoder_retweet_c.pkl"
        elif target_col == "like_timestamp":
            f_name = "gan_dict_label_encoder_like.pkl"

        with open(os.path.join(self.model_path, f_name), "wb") as f:
            pickle.dump(dict_label_encoder, f)

        data_X = pd.get_dummies(data_X, columns=['tweet_type'])
        data_X = data_X.astype('float')

        X = data_X.values
        y = data[target_col].values

        if train_ratio > 0.0 and train_ratio < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=self.GLOBAL_SEED)

            scaler = RobustScaler()
            X_train = pd.DataFrame(X_train, columns=data_X.columns)
            X_subset = scaler.fit_transform(
                X_train.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                                'engaging_user_follower_count', 'engaging_user_following_count',
                                'engaged_time > engaging_time',
                                'tweet_time > engaging_time', 'tweet_hour', 'language',
                                'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                                'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset = pd.DataFrame(X_subset)
            X_last_column = X_train.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                            'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                            'engagee_follows_engager', 'present_links',
                                            'bool_(engaged_time > engaging_time)',
                                            'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2', 'sentenc_vector_3',
                                            'sentenc_vector_4',
                                            'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7', 'sentenc_vector_8',
                                            'sentenc_vector_9',
                                            'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                            'sentenc_vector_13', 'sentenc_vector_14',
                                            'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                            'sentenc_vector_18', 'sentenc_vector_19']]
            X_train = pd.concat((X_subset, X_last_column), axis=1)
            # X_train = data_X2.values

            X_test = pd.DataFrame(X_test, columns=data_X.columns)
            X_subset_test = scaler.transform(
                X_test.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                               'engaging_user_follower_count', 'engaging_user_following_count',
                               'engaged_time > engaging_time',
                               'tweet_time > engaging_time', 'tweet_hour', 'language',
                               'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                               'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset_test = pd.DataFrame(X_subset_test)
            X_last_column_test = X_test.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                                'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                                'engagee_follows_engager', 'present_links',
                                                'bool_(engaged_time > engaging_time)',
                                                'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2',
                                                'sentenc_vector_3', 'sentenc_vector_4',
                                                'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7',
                                                'sentenc_vector_8', 'sentenc_vector_9',
                                                'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                                'sentenc_vector_13', 'sentenc_vector_14',
                                                'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                                'sentenc_vector_18', 'sentenc_vector_19']]
            X_test = pd.concat((X_subset_test, X_last_column_test), axis=1)
            # X_test = data_X2_test.values

            y_train = y_train.astype('int32')

            if target_col == "reply_timestamp":
                f_name = "gan_robust_scaler_reply.pkl"
            elif target_col == "retweet_timestamp":
                f_name = "gan_robust_scaler_retweet.pkl"
            elif target_col == "retweet_with_comment_timestamp":
                f_name = "gan_robust_scaler_retweet_c.pkl"
            elif target_col == "like_timestamp":
                f_name = "gan_robust_scaler_like.pkl"

            with open(os.path.join(self.model_path, f_name), "wb") as f:
                pickle.dump(scaler, f)

            return (X_train, y_train, X_test, y_test)

        else:
            scaler = RobustScaler()
            X = pd.DataFrame(X, columns=data_X.columns)
            X_subset = scaler.fit_transform(
                X.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                          'engaging_user_follower_count', 'engaging_user_following_count',
                          'engaged_time > engaging_time',
                          'tweet_time > engaging_time', 'tweet_hour', 'language',
                          'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                          'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset = pd.DataFrame(X_subset)
            X_last_column = X.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                      'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                      'engagee_follows_engager', 'present_links',
                                      'bool_(engaged_time > engaging_time)',
                                      'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2',
                                      'sentenc_vector_3',
                                      'sentenc_vector_4',
                                      'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7',
                                      'sentenc_vector_8',
                                      'sentenc_vector_9',
                                      'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                      'sentenc_vector_13', 'sentenc_vector_14',
                                      'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                      'sentenc_vector_18', 'sentenc_vector_19']]
            X = pd.concat((X_subset, X_last_column), axis=1)

            y = y.astype('int32')

            return (X, y)

    def train(self, datas, epochs, leanring_rate, token_model, target_column):

        X_train = None
        y_train = None
        X_test = None
        y_test = None

        if token_model.lower() == "dan":
            if target_column == "reply_timestamp":
                col_name = "reply"
            elif target_column == "retweet_timestamp":
                col_name = "retweet"
            elif target_column == "retweet_with_comment_timestamp":
                col_name = "retweet_c"
            elif target_column == "like_timestamp":
                col_name = "like"

            f_Dname = f"D_{token_model.lower()}_{col_name}.pth"
            f_Gname = f"G_{token_model.lower()}_{col_name}.pth"

        else:
            f_Dname = f"D_{token_model.lower()}.pth"
            f_Gname = f"G_{token_model.lower()}.pth"

        if len(datas) == 4:
            X_train, y_train, X_test, y_test = datas
        elif len(datas) == 2:
            X_train, y_train = datas

        X_train["label"] = y_train

        X_train_True = X_train[X_train['label'] == 1]

        X_train_gan = X_train_True.loc[:, X_train_True.columns[:-1]].values
        y_train_gan = X_train_True.loc[:, ['label']].values

        y_train_gan = y_train_gan.astype('int32')

        # 위에서 설명한 데이터 텐서화
        X_train_gan_torch = torch.FloatTensor(X_train_gan)
        y_train_gan_torch = torch.LongTensor(y_train_gan)

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        EPOCHS = epochs
        # BATCH_SIZE = int(X_train_True.shape[0] * 0.1)
        BATCH_SIZE = int(X_train_True.shape[0] * 0.0125)
        print(BATCH_SIZE)

        train_dataset = GanDataset(X_train_gan_torch, y_train_gan_torch)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                   drop_last=True)

        D = CustomDiscriminator()
        G = CustomGenerator()

        D = D.to(DEVICE)
        G = G.to(DEVICE)

        criterion = nn.BCELoss()
        d_optimizer = optim.Adam(D.parameters(), lr=leanring_rate, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(G.parameters(), lr=leanring_rate, betas=(0.5, 0.999))

        # for epoch in (tqdm(range(EPOCHS),desc="train gan processing") if self.verbose else range(EPOCHS)):
        for epoch in range(EPOCHS):
            for i, (images, _) in enumerate(train_loader):
                # images = images.reshape(BATCH_SIZE, -1).to(DEVICE)

                # '진짜'와 '가짜' 레이블 생성
                real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE)
                fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

                # 판별자가 진짜 데이터를 진짜로 인식하는 오차를 예산
                outputs = D(images.to(DEVICE))
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs.detach().cpu().numpy()

                # 무작위 텐서로 가짜 데이터 생성
                z = torch.randn(BATCH_SIZE, 100, device=DEVICE)
                fake_images = G(z)

                # 판별자가 가짜 데이터를 가짜로 인식하는 오차를 계산
                outputs = D(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs.detach().cpu().numpy()

                # 진짜와 가짜 데이터를 갖고 낸 오차를 더해서 판별자의 오차 계산
                d_loss = d_loss_real + d_loss_fake

                # 역전파 알고리즘으로 판별자 모델의 학습을 진행
                if epoch % 5 == 0:
                    d_optimizer.zero_grad()
                    g_optimizer.zero_grad()
                    d_loss.backward()
                    d_optimizer.step()

                # 생성자가 판별자를 속였는지에 대한 오차를 계산
                fake_images = G(z)
                outputs = D(fake_images)
                g_loss = criterion(outputs, real_labels)

                # 역전파 알고리즘으로 생성자 모델의 학습을 진행
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # 학습 진행 알아보기
            print('\nEpoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, EPOCHS, d_loss.item(), g_loss.item(),
                          real_score.mean(), fake_score.mean()))

            torch.save(D.state_dict(), os.path.join(self.model_path, f_Dname))
            torch.save(G.state_dict(), os.path.join(self.model_path, f_Gname))

    def predict(self, token_model, target_column, model_path, true_num, gen_ratio=0.5):
        if token_model.lower() == "dan":
            d_file = f"D_{token_model.lower()}_{target_column}.pth"
            g_file = f"G_{token_model.lower()}_{target_column}.pth"
        else:
            d_file = f"D_{token_model.lower()}.pth"
            g_file = f"G_{token_model.lower()}.pth"

        D = CustomDiscriminator()
        G = CustomGenerator()

        D.load_state_dict(torch.load(os.path.join(model_path, d_file)))
        G.load_state_dict(torch.load(os.path.join(model_path, g_file)))


        # D.D.load_state_dict(torch.load(os.path.join(model_path, d_file)).state_dict())
        # G.G.load_state_dict(torch.load(os.path.join(model_path, g_file)).state_dict())

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        G = G.to(DEVICE)

        G.eval()

        # BATCH_SIZE = int(true_num * gen_ratio)

        gan_size = int(true_num * gen_ratio)
        batch_size = min(10000, int(gan_size*0.1))

        # print(batch_size)

        batch_num = (gan_size // batch_size) + 1

        result_fake_data = []

        for idx in (tqdm(range(batch_num), desc="inference gan processing") if self.verbose else range(batch_num)):
            if idx == batch_num-1:
                remainder = gan_size % batch_size
                z = torch.randn(remainder, 100, device=DEVICE)
            else:
                z = torch.randn(batch_size, 100, device=DEVICE)

            fake_datas = G(z)
            fake_datas_data = fake_datas.detach().cpu().numpy()
            result_fake_data.append(fake_datas_data)

        result_fake_data = np.vstack(result_fake_data)

        # for i in (tqdm(range(BATCH_SIZE),desc="inference gan processing") if self.verbose else range(BATCH_SIZE)):
        #     fake_datas = G(z)
        #     fake_datas_data = fake_datas.detach().cpu().numpy()
        #     result_fake_data.append(fake_datas_data)
        #
        # result_fake_data = np.vstack(result_fake_data)
        # y_fake_data = np.array([1 for i in range(result_fake_data.shape[0])])
        #
        # if self.verbose:
        #     print("inference gan processing...")

        # fake_datas = G(z)
        # fake_datas_data = fake_datas.detach().cpu().numpy()

        y_fake_data = np.array([1 for i in range(result_fake_data.shape[0])])

        return result_fake_data, y_fake_data


class AnnProcessor:
    def __init__(self, model_path, global_seed, verbose=False):
        # self.df_train = df_train
        # self.glove = glove
        self.model_path = model_path
        self.verbose = verbose
        self.seed_initialization(global_seed)

    def seed_initialization(self, GLOBAL_SEED):

        self.GLOBAL_SEED = GLOBAL_SEED

        torch.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(GLOBAL_SEED)
        random.seed(GLOBAL_SEED)

    def get_token_feature(self, df_train, glove, target_col, token_model):
        if token_model == "TFIDF":
            processor = TfidfSifProcessor(df_train, glove, self.model_path, self.verbose)
            processor.preprocess()
            emb_result = processor.predict(token_model)

        elif token_model == "SIF":
            processor = TfidfSifProcessor(df_train, glove, self.model_path, self.verbose)
            processor.preprocess()
            emb_result = processor.predict(token_model)

        elif token_model == "DAN":
            processor = DanProcessor(df_train, glove, self.model_path, self.verbose)
            dataloader = processor.preprocess(target_col, 10000, 0.0, self.GLOBAL_SEED)
            emb_result = processor.predict(dataloader, target_col, self.model_path)
            torch.cuda.empty_cache()

        return emb_result

    def preprocess(self, df_train, train_ratio, glove, token_model, target_col):
        self.df_train = df_train
        self.glove = glove
        self.token_model = token_model

        data = df_train.copy()

        df_indexes = data[["tweet_id","engaging_user_id"]]

        tweet_hour = []

        for tweet_time in data['tweet_timestamp'].astype(int).values:
            tweet_hour.append(datetime.fromtimestamp(tweet_time).strftime('%H'))

        data['engaged_time > engaging_time'] = ((pd.to_numeric(
            data['engaged_with_user_account_creation']) - pd.to_numeric(
            data['engaging_user_account_creation'])) / 1000000).astype(int)
        data['tweet_time > engaging_time'] = ((pd.to_numeric(data['tweet_timestamp']) - pd.to_numeric(
            data['engaging_user_account_creation'])) / 1000000).astype(int)
        data['tweet_hour'] = tweet_hour
        data['tweet_hour'] = data['tweet_hour'].astype(int)

        data.loc[data["engaged_time > engaging_time"] >= 0, "bool_(engaged_time > engaging_time)"] = 1
        data.loc[data["engaged_time > engaging_time"] < 0, "bool_(engaged_time > engaging_time)"] = 0

        data['num_GIF'] = data['present_media'].str.count('GIF')
        data['num_Video'] = data['present_media'].str.count('Video')
        data['num_Photo'] = data['present_media'].str.count('Photo')

        data['num_GIF'] = data['num_GIF'].fillna(0)
        data['num_Video'] = data['num_Video'].fillna(0)
        data['num_Photo'] = data['num_Photo'].fillna(0)

        data.loc[data["like_timestamp"].str.len() >= 1, "like_timestamp"] = 1
        data.loc[data["like_timestamp"] != 1, "like_timestamp"] = 0

        data.loc[data["retweet_timestamp"].str.len() >= 1, "retweet_timestamp"] = 1
        data.loc[data["retweet_timestamp"] != 1, "retweet_timestamp"] = 0

        data.loc[data["retweet_with_comment_timestamp"].str.len() >= 1, "retweet_with_comment_timestamp"] = 1
        data.loc[data["retweet_with_comment_timestamp"] != 1, "retweet_with_comment_timestamp"] = 0

        data.loc[data["reply_timestamp"].str.len() >= 1, "reply_timestamp"] = 1
        data.loc[data["reply_timestamp"] != 1, "reply_timestamp"] = 0

        data.loc[data["present_links"] != 0, "present_links"] = 1

        data_X = data.loc[:, ['tweet_type', 'engaged_with_user_is_verified', 'engaging_user_is_verified'
                                 , 'engagee_follows_engager', 'engaged_with_user_following_count',
                              'engaged_with_user_follower_count'
                                 , 'engaging_user_follower_count', 'engaging_user_following_count', 'present_links',
                              'engaged_time > engaging_time', 'tweet_time > engaging_time', 'tweet_hour', 'language',
                              'bool_(engaged_time > engaging_time)', 'num_GIF', 'num_Video', 'num_Photo']]

        data_X.loc[data["engaged_with_user_is_verified"] == 'false', "engaged_with_user_is_verified"] = 0
        data_X.loc[data["engaged_with_user_is_verified"] == 'true', "engaged_with_user_is_verified"] = 1
        data_X.loc[data["engaging_user_is_verified"] == 'false', "engaging_user_is_verified"] = 0
        data_X.loc[data["engaging_user_is_verified"] == 'true', "engaging_user_is_verified"] = 1
        data_X.loc[data["engagee_follows_engager"] == 'false', "engagee_follows_engager"] = 0
        data_X.loc[data["engagee_follows_engager"] == 'true', "engagee_follows_engager"] = 1

        data_X.loc[data_X["language"].str.contains('488B32D24BD4BB44172EB981C1BCA6FA', na=False), "language"] = '10'
        data_X.loc[data_X["language"].str.contains('E7F038DE3EAD397AEC9193686C911677', na=False), "language"] = '9'
        data_X.loc[data_X["language"].str.contains('B0FA488F2911701DD8EC5B1EA5E322D8', na=False), "language"] = '8'
        data_X.loc[data_X["language"].str.contains('B8B04128918BBF54E2E178BFF1ABA833', na=False), "language"] = '7'
        data_X.loc[data_X["language"].str.contains('313ECD3A1E5BB07406E4249475C2D6D6', na=False), "language"] = '6'
        data_X.loc[data_X["language"].str.contains('1F73BB863A39DB62B4A55B7E558DB1E8', na=False), "language"] = '5'
        data_X.loc[data_X["language"].str.contains('9FCF19233EAD65EA6E32C2E6DC03A444', na=False), "language"] = '4'
        data_X.loc[data_X["language"].str.contains('9A78FC330083E72BE0DD1EA92656F3B5', na=False), "language"] = '3'
        data_X.loc[data_X["language"].str.contains('8729EBF694C3DAF61208A209C2A542C8', na=False), "language"] = '2'
        data_X.loc[data_X["language"].str.contains('E6936751CBF4F921F7DE1AEF33A16ED0', na=False), "language"] = '1'
        data_X.loc[data_X["language"].str.len() > 3, "language"] = '0'

        data_X['engaged_with_user_following_count'] = data_X['engaged_with_user_following_count'].astype(int)
        data_X['engaged_with_user_follower_count'] = data_X['engaged_with_user_follower_count'].astype(int)
        data_X['engaging_user_follower_count'] = data_X['engaging_user_follower_count'].astype(int)
        data_X['engaging_user_following_count'] = data_X['engaging_user_following_count'].astype(int)

        data_X['engaged_time'] = (pd.to_numeric(data['engaged_with_user_account_creation']) / 1000000)

        data_X['engaging_time'] = (pd.to_numeric(data['engaging_user_account_creation']) / 1000000)
        data_X['tweet_timestamp'] = (pd.to_numeric(data['tweet_timestamp']) / 1000000)

        emb_result = self.get_token_feature(df_train, glove, target_col, token_model)

        sentence_vectors = ['sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2', 'sentenc_vector_3',
                            'sentenc_vector_4', 'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7'
            , 'sentenc_vector_8', 'sentenc_vector_9', 'sentenc_vector_10', 'sentenc_vector_11'
            , 'sentenc_vector_12', 'sentenc_vector_13', 'sentenc_vector_14', 'sentenc_vector_15'
            , 'sentenc_vector_16', 'sentenc_vector_17', 'sentenc_vector_18', 'sentenc_vector_19']

        data_X[sentence_vectors] = emb_result

        # data_X['engaging_time'] = (pd.to_numeric(data['engaging_user_account_creation']) / 1000000)
        # data_X['tweet_timestamp'] = (pd.to_numeric(data['tweet_timestamp']) / 1000000)

        data_X["engaging_user_following_count"][data_X['engaging_user_following_count'] == 0] = 1
        data_X["engaged_with_user_following_count"][data_X['engaged_with_user_following_count'] == 0] = 1

        data_X["engaging_user_follower_count"][data_X['engaging_user_follower_count'] == 0] = 1
        data_X["engaged_with_user_follower_count"][data_X['engaged_with_user_follower_count'] == 0] = 1

        data_X['ratio_enaging_follow'] = data_X['engaging_user_follower_count'] / data_X[
            'engaging_user_following_count']
        data_X['ratio_enagaged_follow'] = data_X['engaged_with_user_follower_count'] / data_X[
            'engaged_with_user_following_count']

        for i in range(0, 1):
            le = LabelEncoder()
            y = list(data_X.iloc[:, i])

            le.fit(y)
            y2 = le.transform(y)

            data_X.iloc[:, i] = y2

        dict_label_encoder = dict(zip(y,y2))

        if target_col == "reply_timestamp":
            f_name = "ann_dict_label_encoder_reply.pkl"
        elif target_col == "retweet_timestamp":
            f_name = "ann_dict_label_encoder_retweet.pkl"
        elif target_col == "retweet_with_comment_timestamp":
            f_name = "ann_dict_label_encoder_retweet_c.pkl"
        elif target_col == "like_timestamp":
            f_name = "ann_dict_label_encoder_like.pkl"

        with open(os.path.join(self.model_path, f_name), "wb") as f:
            pickle.dump(dict_label_encoder, f)

        data_X = pd.get_dummies(data_X, columns=['tweet_type'])
        data_X = data_X.astype('float')

        X = data_X.values
        y = data[target_col].values

        if train_ratio > 0.0 and train_ratio < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=self.GLOBAL_SEED)

            scaler = RobustScaler()
            X_train = pd.DataFrame(X_train, columns=data_X.columns)
            X_subset = scaler.fit_transform(
                X_train.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                                'engaging_user_follower_count', 'engaging_user_following_count',
                                'engaged_time > engaging_time',
                                'tweet_time > engaging_time', 'tweet_hour', 'language',
                                'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                                'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset = pd.DataFrame(X_subset)
            X_last_column = X_train.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                            'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                            'engagee_follows_engager', 'present_links',
                                            'bool_(engaged_time > engaging_time)',
                                            'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2', 'sentenc_vector_3',
                                            'sentenc_vector_4',
                                            'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7', 'sentenc_vector_8',
                                            'sentenc_vector_9',
                                            'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                            'sentenc_vector_13', 'sentenc_vector_14',
                                            'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                            'sentenc_vector_18', 'sentenc_vector_19']]
            X_train = pd.concat((X_subset, X_last_column), axis=1)
            # X_train = data_X2.values

            X_test = pd.DataFrame(X_test, columns=data_X.columns)
            X_subset_test = scaler.transform(
                X_test.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                               'engaging_user_follower_count', 'engaging_user_following_count',
                               'engaged_time > engaging_time',
                               'tweet_time > engaging_time', 'tweet_hour', 'language',
                               'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                               'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset_test = pd.DataFrame(X_subset_test)
            X_last_column_test = X_test.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                                'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                                'engagee_follows_engager', 'present_links',
                                                'bool_(engaged_time > engaging_time)',
                                                'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2',
                                                'sentenc_vector_3', 'sentenc_vector_4',
                                                'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7',
                                                'sentenc_vector_8', 'sentenc_vector_9',
                                                'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                                'sentenc_vector_13', 'sentenc_vector_14',
                                                'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                                'sentenc_vector_18', 'sentenc_vector_19']]
            X_test = pd.concat((X_subset_test, X_last_column_test), axis=1)
            # X_test = data_X2_test.values

            y_train = y_train.astype('int32')
            y_test = y_test.astype('int32')

            if target_col == "reply_timestamp":
                f_name = "ann_robust_scaler_reply.pkl"
            elif target_col == "retweet_timestamp":
                f_name = "ann_robust_scaler_retweet.pkl"
            elif target_col == "retweet_with_comment_timestamp":
                f_name = "ann_robust_scaler_retweet_c.pkl"
            elif target_col == "like_timestamp":
                f_name = "ann_robust_scaler_like.pkl"

            with open(os.path.join(self.model_path, f_name), "wb") as f:
                pickle.dump(scaler, f)

            return (X_train, y_train, X_test, y_test)

        else:
            scaler = RobustScaler()
            X = pd.DataFrame(X, columns=data_X.columns)
            X_subset = scaler.fit_transform(
                X.loc[:, ['engaged_with_user_following_count', 'engaged_with_user_follower_count',
                                'engaging_user_follower_count', 'engaging_user_following_count',
                                'engaged_time > engaging_time',
                                'tweet_time > engaging_time', 'tweet_hour', 'language',
                                'num_GIF', 'num_Video', 'num_Photo', 'engaged_time', 'engaging_time', 'tweet_timestamp',
                                'ratio_enaging_follow', 'ratio_enagaged_follow']])
            X_subset = pd.DataFrame(X_subset)
            X_last_column = X.loc[:, ['tweet_type_0', 'tweet_type_1', 'tweet_type_2',
                                            'engaged_with_user_is_verified', 'engaging_user_is_verified',
                                            'engagee_follows_engager', 'present_links',
                                            'bool_(engaged_time > engaging_time)',
                                            'sentenc_vector_0', 'sentenc_vector_1', 'sentenc_vector_2',
                                            'sentenc_vector_3',
                                            'sentenc_vector_4',
                                            'sentenc_vector_5', 'sentenc_vector_6', 'sentenc_vector_7',
                                            'sentenc_vector_8',
                                            'sentenc_vector_9',
                                            'sentenc_vector_10', 'sentenc_vector_11', 'sentenc_vector_12',
                                            'sentenc_vector_13', 'sentenc_vector_14',
                                            'sentenc_vector_15', 'sentenc_vector_16', 'sentenc_vector_17',
                                            'sentenc_vector_18', 'sentenc_vector_19']]
            X = pd.concat((X_subset, X_last_column), axis=1)

            y = y.astype('int32')

            if train_ratio == 0.0:
                return df_indexes, (X,y)

            else:
                return (X, y)

    def train(self, datas, target_column, epochs, batch_size, learning_rate, patience, token_model, gan_ratio):

        if token_model.lower() == "dan":
            f_name = f"ANN_{target_column.lower()}_{token_model.lower()}_gan.pth" if gan_ratio > 0.0 else f"ANN_{target_column.lower()}_{token_model.lower()}_not_gan.pth"
        else:
            f_name = f"ANN_{token_model.lower()}_gan.pth" if gan_ratio > 0.0 else f"ANN_{token_model.lower()}_not_gan.pth"

        X_train, y_train, X_test, y_test = datas

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ann = CustomANN()
        ann = ann.to(device)

        es = None
        if patience > 0:
            es = EarlyStopping(self.model_path, patience, ("max","max"))
            es.names = ["rce", "ap"]

        X_train_torch = torch.FloatTensor(X_train if gan_ratio > 0.0 else X_train.values)
        y_train_torch = torch.LongTensor(y_train)

        train_dataset = AnnDataset(X_train_torch, y_train_torch)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if (X_test is not None) & (y_test is not None):
            X_test_torch = torch.FloatTensor(X_test.values)
            y_test_torch = torch.LongTensor(y_test)

            test_dataset = AnnDataset(X_test_torch, y_test_torch)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # loss 함수 정의
        criterion = torch.nn.SmoothL1Loss(reduction='sum')
        # 최적화 함수 정의
        optimizer = optim.Adam(ann.parameters(), lr=learning_rate)

        for epoch in (tqdm(range(epochs), desc="train dnn processing") if self.verbose else range(epochs)):
            ann.train()
            running_loss = 0.0

            for i, data in enumerate(train_dataloader):
                # [inputs, labels]의 목록인 data 로부터 입력을 받은 후;
                inputs, labels = data[0].to(device), data[1].to(device)

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = ann(inputs)
                labels = labels.type_as(outputs)
                loss = criterion(outputs, labels.reshape(-1, 1))
                loss.backward()
                optimizer.step()

                # loss 출력
                running_loss += loss.item()
                # if self.verbose:
                #     if i % 10 == 9:
                #         print('[%d, %5d] loss: %.3f' %
                #               (epoch + 1, i + 1, running_loss / 10))
                #         running_loss = 0.0

            rce = None
            ap = None

            if (X_test is not None) & (y_test is not None):
                ann.eval()
                for data in test_loader:
                    lr_probs = []
                    test_y = []
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = ann(images)
                    lr_probs = outputs.cpu().squeeze().tolist()
                    test_y = labels.cpu().squeeze().tolist()

                    rce = compute_rce(lr_probs, test_y)
                    ap = average_precision_score(test_y, lr_probs)

                    break

                if self.verbose:
                    print(
                        f"EPOCH:{epoch + 1}|{epochs}; loss:{running_loss / len(train_dataloader):.4f}; rce:{rce:.4f}; ap:{ap:.4f}")

                if patience > 0:
                    # rce = compute_rce(lr_probs, test_y)
                    # ap = average_precision_score(test_y, lr_probs)
                    es((rce,ap), ann, f_name)

                if es.early_stop:
                    print("early_stopping")
                    break
                # if self.verbose:
                #     print("rce = ", compute_rce(lr_probs, test_y), "ap = ", average_precision_score(test_y, lr_probs))

            # if epoch % 5 == 4:
                #     ann.eval()
                #     for data in test_loader:
                #         lr_probs = []
                #         test_y = []
                #         images, labels = data[0].to(device), data[1].to(device)
                #         outputs = ann(images)
                #         lr_probs = outputs.cpu().squeeze().tolist()
                #         test_y = labels.cpu().squeeze().tolist()
                #         break
                #     if self.verbose:
                #         print("rce = ", compute_rce(lr_probs, test_y), "ap = ", average_precision_score(test_y, lr_probs))


            # torch.save(ann.state_dict(), os.path.join(self.model_path, f_name))

        # if self.verbose:
        #     print('Finished Training')

    def predict(self, datas, target_column, batch_size, token_model, is_gan, model_path):


        f_name = f"ANN_{target_column.lower()}_{token_model.lower()}_gan.pth" if is_gan else f"ANN_{target_column.lower()}_{token_model.lower()}_not_gan.pth"

        lr_probs = []
        y_true = []

        X, y = datas

        np_X = X.values

        X = torch.FloatTensor(np_X)
        y = torch.LongTensor(y)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ann = CustomANN()
        ann.load_state_dict(torch.load(os.path.join(model_path, f_name)))

        ann = ann.to(device)

        test_dataset = AnnDataset(X, y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            ann.eval()
            for data in (tqdm(test_loader, desc="inference ann processing") if self.verbose else test_loader):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = ann(images)
                lr_probs.append(outputs.cpu().tolist())
                y_true.append(labels.cpu().tolist())

        lr_probs = np.concatenate(lr_probs)
        y_true = np.hstack(y_true)

        return lr_probs, y_true





