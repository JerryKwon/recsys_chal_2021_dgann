import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from glove import Glove, Corpus


class GLOVE:
    def __init__(self, df_train, model_path, window_size=10, n_components=20, epochs=25, learning_rate=0.05, verbose=False):
        self.df_train = df_train
        self.window_size = window_size
        self.n_components = n_components
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.verbose = verbose

    def make_target(self):
        corpus_target = self.df_train["text_ tokens"].apply(lambda x: x.split(" "))
        return corpus_target

    def make_corpus(self, corpus_target):
        corpus = Corpus()

        begin = datetime.now()
        begin_time = begin.time()

        if self.verbose:

            print(f"Making co-occurrence matrix for window_size:{self.window_size} at {begin_time.hour:02}:{begin_time.minute:02}:{begin_time.second:02}")

        corpus.fit(corpus_target, self.window_size)
        end = datetime.now()

        elapse_sec = (end - begin).seconds

        if self.verbose:
            print(f"Elasped {elapse_sec // 60} min {elapse_sec % 60} sec for {self.window_size} window_size")

        return corpus

    def train(self):
        corpus_target = self.make_target()
        corpus = self.make_corpus(corpus_target)

        glove = Glove(no_components=self.n_components, learning_rate=self.learning_rate)

        begin = datetime.now()
        begin_time = begin.time()
        if self.verbose:
            print(f"glove begin {begin_time.hour}:{begin_time.minute}:{begin_time.second}")
        glove.fit(corpus.matrix, epochs=self.epochs, no_threads=4, verbose=self.verbose)
        end = datetime.now()

        elapse_sec = (end - begin).seconds

        if self.verbose:
            print(f"Elasped {elapse_sec // 60} min {elapse_sec % 60} sec for {self.epochs} epochs")

        glove.add_dictionary(corpus.dictionary)

        glove.save(os.path.join(self.model_path, f"glove_{self.window_size}_{self.epochs}_{self.learning_rate}_{self.n_components}.model"))


class WordDropout(nn.Module):
    def __init__(self, embedding_layer, DEVICE, dropout=0.3):
        super(WordDropout, self).__init__()
        self.dropout = dropout
        self.DEVICE = DEVICE
        self.embedding_layer = embedding_layer

    def sampling(self, batch):
        #         new_batch = batch.detach().cpu()
        new_batches = []
        inputs, len_idx = batch
        max_size = inputs.shape[1]

        target_idx = np.arange(max_size)[np.random.binomial(1, 0.3, max_size) == 1]

        embeddings = self.embedding_layer((inputs[:, target_idx] if self.training else inputs))
        for emb in embeddings:
            target_emb = emb[emb.sum(axis=1) != 0]

            if target_emb.nelement() == 0:
                new_batches.append(torch.zeros(20, dtype=torch.float).to(self.DEVICE))
            else:
                new_batches.append(target_emb.mean(axis=0))

        result = torch.vstack(new_batches)
        result.requires_grad = True
        return result

    def forward(self, x):
        return self.sampling(x)


class CustomDAN(nn.Module):
    def __init__(self, embeddings, DEVICE, hidden_layers=2):
        super(CustomDAN, self).__init__()

        self.embedding_layer = self.make_emb_layer(embeddings)
        self.word_dropout = WordDropout(self.embedding_layer, DEVICE)

        h_layers = []
        for idx in range(hidden_layers):
            h_layers.append(self.make_hidden_layer(20, 20))

        self.hidden_layers = nn.Sequential(*h_layers)

        self.softmax = nn.Softmax()

        self.dense_out = nn.Linear(20, 2)

    def make_emb_layer(self, embeddings):
        num_embeddings, embedding_dim = embeddings.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({"weight": torch.FloatTensor(embeddings)})
        emb_layer.weight.requires_grad = False

        return emb_layer

    def make_hidden_layer(self, in_channel, out_channel, dropout=0.2):
        layers = []

        layers.append(nn.Linear(in_channel, out_channel))
        layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        wd = self.word_dropout(x)

        h = self.hidden_layers(wd)

        x = self.dense_out(h)
        x = self.softmax(x)

        return h, x


class AEBlock(nn.Module):
    def __init__(self, layers, activation=nn.ReLU(), is_first=False, is_last=False):
        super(AEBlock, self).__init__()
        self.block = self.make_layers(layers, activation, is_first, is_last)

    def make_layers(self, layers, activation, is_first, is_last, dropout=0.2):
        list_layers = []
        if is_first:
            list_layers.append(nn.BatchNorm1d(layers[0]))
            list_layers.append(nn.Dropout(dropout))

        list_layers.append(nn.Linear(layers[0], layers[1]))

        if is_last:
            list_layers.append(activation)

        else:
            list_layers.append(nn.BatchNorm1d(layers[1]))
            list_layers.append(activation)
            list_layers.append(nn.Dropout(dropout))

        return nn.Sequential(*list_layers)

    def forward(self, x):
        return self.block(x)


class Pretrain_AutoEncoder(nn.Module):
    def __init__(self, layers, activation=nn.ReLU(), is_first=False, is_last=False):
        super(Pretrain_AutoEncoder, self).__init__()
        self.encoder = self.make_encoder(layers, activation, is_first)
        self.decoder = self.make_decoder(layers, activation, is_first, is_last)

    def make_encoder(self, layers, activation, is_first):
        encoder = []
        for idx in np.arange(len(layers) - 1):
            if idx == 0:
                encoder.append(AEBlock([layers[idx], layers[idx + 1]], activation=activation, is_first=is_first))
            else:
                encoder.append(AEBlock([layers[idx], layers[idx + 1]], activation=activation))

        return nn.Sequential(*encoder)

    def make_decoder(self, layers, activation, is_first, is_last):
        decoder = []
        for idx in np.arange(len(layers))[:0:-1]:
            if idx == 1:
                if is_first:
                    decoder.append(AEBlock([layers[idx], layers[idx - 1]], activation=nn.Sigmoid(), is_last=is_last))
                else:
                    decoder.append(AEBlock([layers[idx], layers[idx - 1]], activation=activation, is_last=is_last))
            else:
                decoder.append(AEBlock([layers[idx], layers[idx - 1]], activation=activation))

        return nn.Sequential(*decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Tying_AutoEncoder(nn.Module):

    def __init__(self, autoencoder, is_first=False, is_last=False):
        super(Tying_AutoEncoder, self).__init__()

        self.encoder, self.decoder = self.tying_weights(autoencoder, is_first, is_last)

    def tying_weights(self, autoencoder, is_first, is_last):
        linear_weights = []
        batch_weights = []

        encoder = autoencoder.encoder
        decoder = autoencoder.decoder

        for idx, enc in enumerate(encoder):
            if is_first & (idx == 0):
                linear_weights.append(nn.Parameter(enc.block[2].weight.t()))
            else:
                linear_weights.append(nn.Parameter(enc.block[0].weight.t()))

        for idx, dec in enumerate(decoder[::-1]):
            dec.block[0].weight = linear_weights[idx]

        if ~is_last:
            for idx, enc in enumerate(encoder):
                if idx == 0:
                    continue

                batch_weights.append(nn.Parameter(enc.block[1].weight.t()))

            for idx, dec in enumerate(decoder[:-1:-1]):
                dec.block[1].weight = linear_weights[idx]

        return encoder, decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SoftAssignment(nn.Module):
    def __init__(self, cluster_number, hidden_dimension, alpha, cluster_centers=None):
        super(SoftAssignment, self).__init__()

        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.hidden_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers

        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class CustomDEC(nn.Module):
    def __init__(self, cluster_number, hidden_dimension, encoder, alpha=1.0):
        super(CustomDEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = SoftAssignment(self.cluster_number, self.hidden_dimension, self.alpha)

    def forward(self, x):
        return self.assignment(self.encoder(x))


class CustomGenerator(nn.Module):
    def __init__(self):
        super(CustomGenerator, self).__init__()

        self.G = nn.Sequential(
                nn.Linear(100, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 44),
                nn.Tanh())

    def forward(self, x):
        return self.G(x)


class CustomDiscriminator(nn.Module):
    def __init__(self):
        super(CustomDiscriminator, self).__init__()

        self.D = nn.Sequential(
                    nn.Linear(44, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    def forward(self, x):
        return self.D(x)


# ANN 모델 생성
class CustomANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(44, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = torch.nn.Dropout(p=0.2)

    # 순전파
    def forward(self, x):
        x = F.leaky_relu(self.bn1((self.fc1(x))))
        x = F.leaky_relu(self.bn2((self.fc2(x))))
        x = F.leaky_relu(self.bn3((self.fc3(x))))
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x
