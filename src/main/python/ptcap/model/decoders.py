import torch

from torch import nn
from torch.autograd import Variable


class Decoder(nn.Module):

    def forward(self, decoder_states, teacher_captions=None):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError


class FullyConnectedDecoder(Decoder):

    def __init__(self, state_dim, caption_len, vocab_size):
        super(FullyConnectedDecoder, self).__init__()
        self.caption_len, self.vocab_size = caption_len, vocab_size
        self.input_mapping = nn.Linear(state_dim,
                                       self.caption_len * self.vocab_size)
        self.caption_mapping = nn.Linear(self.caption_len * self.vocab_size,
                                         self.caption_len * self.vocab_size)

    def forward(self, decoder_states, teacher_captions=None):
        batch_size = decoder_states.size()[0]
        predictions = self.input_mapping(decoder_states)
        if teacher_captions is not None:
            predictions += self.caption_mapping(
                teacher_captions.view(batch_size, -1))
        return predictions.view(batch_size, -1, self.vocab_size)


class LSTMDecoder(nn.Module):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, num_hidden_lstm, features=None, use_cuda=False):

        super(LSTMDecoder, self).__init__()
        self.num_hidden_lstm = num_hidden_lstm

        #Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        #batch_first: whether input and output are (batch, seq, feature)

        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.use_cuda = use_cuda

    def init_hidden_decoder(self, features):
        """
        Hidden states of the LSTM are initialized with features.
        """

        c0 = features.unsqueeze(0)
        h0 = features.unsqueeze(0)
        if self.use_cuda:
            h0 = h0.cuda(self.gpus[0])
            c0 = c0.cuda(self.gpus[0])
        return h0, c0

    def forward(self, features, captions):
        h0, c0 = self.init_hidden_decoder(features)
        embedded_captions = self.embedding(captions)

        h_list, _ = self.lstm(embedded_captions, (h0, c0))

        # Project features in a 'vocab_size'-dimensional space
        h_list1 = torch.stack([self.linear(h) for h in h_list], 0)
        h_list2 = torch.stack([self.softmax(h) for h in h_list1], 0)

        return h_list2
