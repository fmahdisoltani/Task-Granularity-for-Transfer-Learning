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
        self.caption_mapping = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, decoder_states, teacher_captions=None):
        batch_size = decoder_states.size()[0]
        predictions = self.input_mapping(decoder_states)
        if teacher_captions is not None:
            predictions += self.caption_mapping(
                teacher_captions).view(batch_size, -1)
        return predictions.view(batch_size, -1, self.vocab_size)


class LSTMDecoder(Decoder):

    def __init__(self, embedding_size, hidden_size,
                 vocab_size, num_hidden_lstm, use_cuda=False, go_token=0):

        super(LSTMDecoder, self).__init__()
        self.num_hidden_lstm = num_hidden_lstm

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax()
        self.use_cuda = use_cuda
        self.go_token = go_token

    def init_hidden(self, features):
        """
        Hidden states of the LSTM are initialized with features.
        """

        c0 = features.unsqueeze(0)
        h0 = features.unsqueeze(0)
        if self.use_cuda:
            h0 = h0.cuda(self.gpus[0])
            c0 = c0.cuda(self.gpus[0])
        return h0, c0

    def forward(self, features, captions, use_teacher_forcing=False):

        batch_size, seq_len = captions.size()
        go_part = Variable(self.go_token * torch.ones(batch_size, 1).long())

        if use_teacher_forcing:
            # Add go token and remove the last token for all captions
            captions_with_go_token = torch.cat([go_part, captions[:, :-1]], 1)
            probs, _ = self.apply_lstm(features, captions_with_go_token)

        else:
            # Without teacher forcing: use its own predictions as the next input
            probs = self.predict(features, go_part)

        return probs

    def apply_lstm(self, features, captions, lstm_hidden=None):

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)
        embedded_captions = self.embedding(captions)
        lstm_output, lstm_hidden = self.lstm(embedded_captions, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_linear = torch.stack([self.linear(h) for h in lstm_output], 0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_out_linear], 0)

        return probs, lstm_hidden

    def predict(self, features, go_tokens, maxlen=10):
        lstm_input = go_tokens
        output_probs = []
        lstm_hidden = None
        for i in range(maxlen):
            probs, lstm_hidden = self.apply_lstm(features, lstm_input,
                                                 lstm_hidden)

            # Greedy decoding
            _, pred = torch.max(probs, dim=2)
            lstm_input = pred.squeeze(1)

            output_probs.append(probs)

        concatenated_probs = torch.cat(output_probs, dim=1)
        return concatenated_probs
