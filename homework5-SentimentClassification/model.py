import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from sklearn.metrics import accuracy_score


class RNN(nn.Module):
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3):
        super(RNN, self).__init__()

        self.DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.hidden_size = hidden_size
        self.sup = num_layers * 2  # num_layers * num_directions

        # Loading pre-trained embeddings
        self.embeddings = nn.Embedding(emb_size, emb_dimension)

        if pretrained_emb is not None:
            # self.embeddings.weight = nn.Parameter(pretrained_emb, requires_grad=False)
            self.embeddings.weight.data.copy_(pretrained_emb)
        else:
            init_weights()

        # Bi-directional layer
        self.rnn = nn.RNN(emb_dimension,
                          hidden_size,
                          num_layers=num_layers,
                          bidirectional=True)

        # Dropout layer
        self.dropout_train = nn.Dropout(dropout)
        self.dropout_test = nn.Dropout(0.0)

        # Linear Layers
        self.fc1 = nn.Linear(self.sup * hidden_size, int(hidden_size))
        self.fc2 = nn.Linear(int(hidden_size), int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), output_size)

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input_sentences, do_train=True):
        # Get embeddings of each of the words in the sentences
        sentence_of_emb = self.embeddings(input_sentences)

        # [sentence_length, batch_size, emb_dim]
        rnn_input = sentence_of_emb.permute(1, 0, 2)
        h_0 = torch.zeros(self.sup, input_sentences.size(0),
                          self.hidden_size).to(self.DEVICE)

        # out : [sentence_length, batch_size, 2 * hidden_size]
        out, h_n = self.rnn(rnn_input, h_0)

        # h_n : [batch_size, 4 * self.hidden_size]
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.contiguous().view(h_n.size(0), -1)

        output = F.relu(self.fc1(h_n))
        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)
        output = F.relu(self.fc2(output))
        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)
        logits = F.log_softmax(self.fc3(output), dim=1)

        return logits

    def test(self, iter, batch_size):
        y_pred, y_true = [], []
        for batch in iter:
            x, y = batch.text, batch.label - 1
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE)
            if len(x) < batch_size:
                continue
            logits = self.forward(x, do_train=False)
            y_pred.extend(torch.argmax(logits, dim=1).tolist())
            y_true.extend(y.int().tolist())
        return accuracy_score(y_true, y_pred)


class CNN(nn.Module):
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 dropout=0.5):
        super(CNN, self).__init__()

        self.DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.embeddings = nn.Embedding(emb_size, emb_dimension)
        if pretrained_emb is not None:
            self.embeddings.weight.data.copy_(pretrained_emb)
        else:
            init_weights()

        # self.convs = nn.ModuleList([nn.Conv2d(1, 100, (n, pretrained_emb.shape[1])) for n in (3,4,5)])
        self.conv1 = nn.Conv2d(1, 100, (3, pretrained_emb.shape[1]))
        self.conv2 = nn.Conv2d(1, 100, (4, pretrained_emb.shape[1]))
        self.conv3 = nn.Conv2d(1, 100, (5, pretrained_emb.shape[1]))

        # Dropout layer
        self.dropout_train = nn.Dropout(dropout)
        self.dropout_test = nn.Dropout(0.0)

        self.linear = nn.Linear(in_features=emb_dimension,
                                out_features=output_size,
                                bias=True)

    def forward(self, x, do_train=True):
        embedded = self.embeddings(x)
        embedded = embedded.unsqueeze(1)

        convd1 = F.relu(self.conv1(embedded)).squeeze(3)
        pool1 = F.max_pool1d(convd1, convd1.size(2)).squeeze(2)
        convd2 = F.relu(self.conv2(embedded)).squeeze(3)
        pool2 = F.max_pool1d(convd2, convd2.size(2)).squeeze(2)
        convd3 = F.relu(self.conv3(embedded)).squeeze(3)
        pool3 = F.max_pool1d(convd3, convd3.size(2)).squeeze(2)
        output = torch.cat((pool1, pool2, pool3), 1)

        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)
        logits = F.log_softmax(self.linear(output), dim=1)

        return logits

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def test(self, iter, batch_size):
        y_pred, y_true = [], []
        for batch in iter:
            x, y = batch.text, batch.label - 1
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE)
            if len(x) < batch_size:
                continue
            logits = self.forward(x, do_train=False)
            y_pred.extend(torch.argmax(logits, dim=1).tolist())
            y_true.extend(y.int().tolist())
        return accuracy_score(y_true, y_pred)