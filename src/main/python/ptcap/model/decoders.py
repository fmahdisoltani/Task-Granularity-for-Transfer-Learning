import torch

from torch import nn
from torch.autograd import Variable

from ptcap.tensorboardY import (update_dict, register_grad)


class Decoder(nn.Module):

    def forward(self, decoder_states, teacher_captions,
                use_teacher_forcing=False):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError


class FullyConnectedDecoder(Decoder):

    def __init__(self, state_dim, caption_len, vocab_size):
        super(FullyConnectedDecoder, self).__init__()
        self.caption_len, self.vocab_size = caption_len, vocab_size
        self.input_mapping = nn.Linear(state_dim,
                                       self.caption_len * self.vocab_size)
        self.caption_mapping = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, decoder_states, teacher_captions,
                use_teacher_forcing=False):
        batch_size = decoder_states.size()[0]
        predictions = self.input_mapping(decoder_states)
        if use_teacher_forcing:
            predictions += self.caption_mapping(
                teacher_captions).view(batch_size, -1)
        return predictions.view(batch_size, -1, self.vocab_size)


class LSTMDecoder(Decoder):

    def __init__(self, embedding_size, hidden_size, vocab_size,
                 num_hidden_lstm, go_token=0, gpus=None):

        super(LSTMDecoder, self).__init__()
        self.num_hidden_lstm = num_hidden_lstm

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax()
        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.go_token = go_token

        self.gradients = {}
        self.hidden = {}

    def init_hidden(self, features):
        """
        Hidden states of the LSTM are initialized with features.
        c0 and h0 should have the shape of 1 * batch_size * hidden_size
        """

        c0 = features.unsqueeze(0)
        h0 = features.unsqueeze(0)
        return h0, c0

    def forward(self, features, captions, use_teacher_forcing=False):
        """
        This method computes the forward pass of the decoder with or without
        teacher forcing. It should be noted that the <GO> token is
        automatically appended to the input captions.

        Args:
            features: Video features extracted by the encoder.
            captions: Video captions (required if use_teacher_forcing=True).
            use_teacher_forcing: Whether to use teacher forcing or not.

        Returns:
            The probability distribution over the vocabulary across the entire
            sequence.
        """

        batch_size, num_step = captions.size()
        go_part = Variable(self.go_token * torch.ones(batch_size, 1).long())
        if self.use_cuda:
            go_part = go_part.cuda(self.gpus[0])

        if use_teacher_forcing:
            # Add go token and remove the last token for all captions
            captions_with_go_token = torch.cat([go_part, captions[:, :-1]], 1)
            probs, lstm_out_projections, lstm_outputs = self.apply_lstm(
                features, captions_with_go_token)

        else:
            # Without teacher forcing: use its own predictions as the next input
            probs, lstm_out_projections, lstm_outputs = self.predict(
                features, go_part, num_step)

        key_list = ["decoder_lstm_output", "decoder_lstm_out_proj",
                    "token_prob"]
        var_list = [lstm_outputs, lstm_out_projections, probs]

        update_dict(self.hidden, zip(key_list, var_list), num_step)
        register_grad(self.gradients, zip(key_list, var_list), num_step)

        return probs

    def apply_lstm(self, features, captions, lstm_hidden=None):

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)
        embedded_captions = self.embedding(captions)
        lstm_output, lstm_hidden = self.lstm(embedded_captions, lstm_hidden)
        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output],
                                         0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_out_projected], 0)

        return probs, lstm_out_projected, lstm_output

    def predict(self, features, go_tokens, num_step=1):
        lstm_input = go_tokens
        lstm_hidden = None
        lstm_outputs = []
        lstm_out_projections = []
        output_probs = []

        for i in range(num_step):
            probs, lstm_out_projected, lstm_output = self.apply_lstm(
                features, lstm_input, lstm_hidden)

            lstm_outputs.append(lstm_output)
            lstm_out_projections.append(lstm_out_projected)
            output_probs.append(probs)

            # Greedy decoding
            _, preds = torch.max(probs, dim=2)
            lstm_input = preds

        concat_probs = torch.cat(output_probs, dim=1)
        concat_lstm_out_projections = torch.cat(lstm_out_projections, dim=1)
        concat_lstm_outputs = torch.cat(lstm_outputs, dim=1)

        return concat_probs, concat_lstm_out_projections, concat_lstm_outputs
