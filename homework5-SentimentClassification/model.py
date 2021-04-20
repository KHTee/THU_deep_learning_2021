import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from sklearn.metrics import accuracy_score


class BaseModel(nn.Module):
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 dropout=0.3):
        super().__init__()

        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.pretrained_emb = pretrained_emb
        self.output_size = output_size
        self.dropout = dropout
        self.DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Loading pre-trained embeddings
        self.embedding = nn.Embedding(self.emb_size, self.emb_dimension)

        if self.pretrained_emb is not None:
            # self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=False)
            self.embedding.weight.data.copy_(self.pretrained_emb)
        else:
            init_weights()
        self.embedding.weight.requires_grad = False

        # Dropout layer
        self.dropout_train = nn.Dropout(self.dropout)
        self.dropout_test = nn.Dropout(0.0)

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.fc.weight.data.uniform_(-initrange, initrange)
        # self.fc.bias.data.zero_()

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


class RNN(BaseModel):
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3,
                 bidirectional=True):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        # Initialize attributes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_directions = 2 if self.bidirectional else 1
        self.sup = self.num_layers * self.n_directions  # num_layers * num_directions

        # Bi-directional layer
        self.rnn = nn.RNN(self.emb_dimension,
                          self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional)

        # Linear Layers
        self.fc1 = nn.Linear(self.sup * self.hidden_size,
                             int(self.hidden_size))
        self.fc2 = nn.Linear(int(self.hidden_size), int(self.hidden_size / 2))
        self.fc3 = nn.Linear(int(self.hidden_size / 2), self.output_size)

    def forward(self, x, do_train=True):
        self.batch_size, self.sent_length = x.size(0), x.size(1)
        # Get embeddings of each of the words in the sentences
        embedded = self.embedding(x)

        # [sentence_length, batch_size, emb_dim]
        rnn_embedded = embedded.permute(1, 0, 2)
        h_0 = torch.zeros(self.sup, self.batch_size,
                          self.hidden_size).to(self.DEVICE)

        # out : [sentence_length, batch_size, 2 * hidden_size]
        out, h_n = self.rnn(rnn_embedded, h_0)

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


class CNN(BaseModel):
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 dropout=0.5):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        # self.convs = nn.ModuleList([nn.Conv2d(1, 100, (n, pretrained_emb.shape[1])) for n in (3,4,5)])
        self.conv1 = nn.Conv2d(1, int(emb_dimension / 3),
                               (3, pretrained_emb.shape[1]))
        self.conv2 = nn.Conv2d(1, int(emb_dimension / 3),
                               (4, pretrained_emb.shape[1]))
        self.conv3 = nn.Conv2d(1, int(emb_dimension / 3),
                               (5, pretrained_emb.shape[1]))

        self.fc = nn.Linear(in_features=int(int(emb_dimension / 3) * 3),
                            out_features=output_size,
                            bias=True)

    def forward(self, x, do_train=True):
        embedded = self.embedding(x)
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
        logits = F.log_softmax(self.fc(output), dim=1)

        return logits


class BiGRU(BaseModel):
    """BiDirectional GRU

    Args:
        num_layers: An int for number of stacked recurrent layers. (default=2)
        hidden_size: An int for umber of features in the hidden state. (default=256)
        bidirectional: A bool whether to use the bidirectional GRU. (default=True)
        spatial_dropout: A bool whether to use the spatial dropout. (default=True)
    """
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3,
                 bidirectional=True,
                 spatial_dropout=True):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.spatial_dropout = spatial_dropout
        self.n_directions = 2 if self.bidirectional else 1
        if self.spatial_dropout:
            self.spatial_dropout1d = nn.Dropout2d(self.dropout)

        self.gru = nn.GRU(
            self.emb_dimension,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=(0 if self.num_layers == 1 else self.dropout),
            batch_first=True,
            bidirectional=self.bidirectional)

        # Linear layer embedded size = (hidden_size * 3) because of
        # concatenation of max_pooling ,avg_pooling, last hidden state
        self.linear = nn.Linear(self.hidden_size * 3, self.output_size)

    def forward(self, x, do_train=True):
        self.batch_size, self.sent_length = x.size(0), x.size(1)
        embedded_lengths = torch.LongTensor([self.sent_length] *
                                            self.batch_size)
        hidden = None
        # h_0 = torch.zeros(self.sup, self.batch_size,
        #                   self.hidden_size).to(self.DEVICE)
        # h_0 = torch.zeros(1, self.batch_size, self.hidden_size)).to(self.DEVICE)

        # x: (batch_size, sentence_length)
        # embedded: (batch_size, sentence_length, emb_dimension)
        embedded = self.embedding(x)

        if self.spatial_dropout:
            # Convert to (batch_size, emb_dimension, sentence_length)
            embedded = embedded.permute(0, 2, 1)
            embedded = self.spatial_dropout1d(embedded)
            # Convert back to (batch_size, sentence_length, emb_dimension)
            embedded = embedded.permute(0, 2, 1)
        else:
            embedded = self.droput_train(embedded)

        # Pack padded batch of sequences for RNN module
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded,
                                                       embedded_lengths,
                                                       batch_first=True)

        # GRU embedded/output shapes, if batch_first=True
        # embedded: (batch_size, seq_len, emb_dimension)
        # Output: (batch_size, seq_len, hidden_size*num_directions)
        # Number of directions = 2 when used bidirectional, otherwise 1
        # shape of hidden: (num_layers x num_directions, batch_size, hidden_size)
        # Hidden state defaults to zero if not provided
        gru_out, hidden = self.gru(packed_emb, hidden)
        # gru_out: tensor containing the output features h_t from the last layer of the GRU
        # gru_out comprises all the hidden states in the last layer ("last" depth-wise, not time-wise)
        # For biGRu gru_out is the concatenation of a forward GRU representation and a backward GRU representation
        # hidden (h_n) comprises the hidden states after the last timestep

        # Extract and sum last hidden state
        # embedded hidden shape: (num_layers x num_directions, batch_size, hidden_size)
        # Separate hidden state layers
        hidden = hidden.view(self.num_layers, self.n_directions,
                             self.batch_size, self.hidden_size)
        last_hidden = hidden[-1]
        # last hidden shape (num_directions, batch_size, hidden_size)
        # Sum the last hidden state of forward and backward layer
        last_hidden = torch.sum(last_hidden, dim=0)
        # Summed last hidden shape (batch_size, hidden_size)

        # Pad a packed batch
        # gru_out output shape: (batch_size, seq_len, hidden_size*num_directions)
        gru_out, lengths = nn.utils.rnn.pad_packed_sequence(gru_out,
                                                            batch_first=True)

        # Sum the gru_out along the num_directions
        if self.bidirectional:
            gru_out = gru_out[:, :, :self.hidden_size] + gru_out[:, :, self.
                                                                 hidden_size:]

        # Select the maximum value over each dimension of the hidden representation (max pooling)
        # Permute the embedded tensor to dimensions: (batch_size, hidden, seq_len)
        # Output dimensions: (batch_size, hidden_size)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(0, 2, 1),
                                         (1, )).view(self.batch_size, -1)

        # Consider the average of the representations (mean pooling)
        # Sum along the batch axis and divide by the corresponding lengths (FloatTensor)
        # Output shape: (batch_size, hidden_size)
        lengths = lengths.view(-1, 1).type(torch.FloatTensor).to(self.DEVICE)
        avg_pool = torch.sum(gru_out, dim=1) / lengths

        # Concatenate max_pooling, avg_pooling, hidden state and embedded_feat tensor
        concat_out = torch.cat([last_hidden, max_pool, avg_pool], dim=1)

        # concat_out = self.droput_train(concat_out)
        out = self.linear(concat_out)
        return F.log_softmax(out, dim=-1)


class LSTMAttn(BaseModel):
    """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        emb_size : Size of the vocabulary containing unique words
        emb_dimension : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe embedding which we will use to create our word_embedding look-up table 

        --------
        
    """
    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3,
                 bidirectional=True,
                 spatial_dropout=True):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.emb_dimension, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                    new hidden state.
                    
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                        
        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, x, do_train=True):
        """ 
        Parameters
        ----------
        x: embedded_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its embedded as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)
        
        """
        self.batch_size, self.sent_length = x.size(0), x.size(1)
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)
        h_0 = torch.zeros(1, self.batch_size, self.hidden_size).to(self.DEVICE)
        c_0 = torch.zeros(1, self.batch_size, self.hidden_size).to(self.DEVICE)

        output, (h_n, c_n) = self.lstm(
            embedded, (h_0, c_0))  # h_n.size() = (1, batch_size, hidden_size)
        output = output.permute(
            1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, h_n)
        logits = F.log_softmax(self.fc(attn_output), dim=1)

        return logits