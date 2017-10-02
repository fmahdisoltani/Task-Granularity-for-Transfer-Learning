import torch

from torch import nn
from torch.autograd import Variable


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
        # self.go_part = Variable(torch.LongTensor([self.go_token]))
        # if self.use_cuda:
        #     self.go_part = self.go_part.cuda(self.gpus[0])

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
        # go_part = self.go_part.expand(batch_size, 1)

        if use_teacher_forcing:
            # Add go token and remove the last token for all captions
            captions_with_go_token = captions
            probs, _ = self.apply_lstm(features, captions_with_go_token)

        else:
            # Without teacher forcing: use its own predictions as the next input
            probs = self.predict(features, captions, num_step)

        return probs

    def apply_lstm(self, features, captions, lstm_hidden=None):

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)

        # if self.embedding.weight.get_device() == captions.get_device():
        #     print("captions on gpu {}".format(captions.get_device()))
        # else:
        #     new_gpu = self.embedding.weight.get_device()
        #     print("Putting captions on gpu {}".format(new_gpu))
        #     captions = captions.cuda(new_gpu)
        embedded_captions = self.embedding(captions)
        lstm_output, lstm_hidden = self.lstm(embedded_captions, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output], 0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_out_projected], 0)

        return probs, lstm_hidden

    def predict(self, features, go_tokens, num_step=1):
        lstm_input = go_tokens
        output_probs = []
        lstm_hidden = None

        for i in range(num_step):
            probs, lstm_hidden = self.apply_lstm(features, lstm_input,
                                                 lstm_hidden)

            output_probs.append(probs)
            # Greedy decoding
            _, preds = torch.max(probs, dim=2)

            lstm_input = preds

        concatenated_probs = torch.cat(output_probs, dim=1)
        return concatenated_probs
