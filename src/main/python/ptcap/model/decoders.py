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

    def forward(self, features, captions, use_teacher_forcing=True):
        batch_size, seq_len = captions.size()
        h0, c0 = self.init_hidden_decoder(features)


        # Add go token and remove the last token for all captions
        go_part = Variable(torch.zeros(batch_size, 1).long())
        captions_with_go_token = torch.cat([go_part, captions[:, :-1]], 1)

        embedded_captions = self.embedding(captions_with_go_token)
        #predictions = []
        lstm_hid, lstm_input = (h0, c0), embedded_captions[:,0,:].unsqueeze(1)
        use_teacher_forcing = False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(seq_len):

                 lstm_output, lstm_hid = self.lstm(lstm_input, lstm_hid)
                 #_, pi = torch.max(lstm_output, dim=2)
                 lstm_input = embedded_captions[:,di,:].unsqueeze(1)  # Teacher forcing
                 #predictions += pi

        else:
            print("YAY")
            # Without teacher forcing: use its own predictions as the next input
            for di in range(seq_len):
                lstm_output, lstm_hid = self.lstm(lstm_input, lstm_hid)

                lstm_hid_linear = torch.stack([self.linear(h) for h in lstm_output], 0)
                #print(lstm_hid_linear)
                #probs = torch.stack([self.logsoftmax(h) for h in lstm_hid_linear], 0)
                _, pi = torch.max(lstm_hid_linear, dim=2)

                topv, topi = lstm_hid_linear.data.topk(1)
                #ni = topi[0][0]

                #lstm_input = Variable(torch.LongTensor([[ni]]))
                lstm_input = self.embedding(pi.squeeze()).unsqueeze(1)

            #    lstm_input = lstm_input.cuda() if use_cuda else lstm_input


        lstm_hid, _ = self.lstm(embedded_captions, (h0, c0))

        # Project features in a 'vocab_size'-dimensional space
        lstm_hid_linear = torch.stack([self.linear(h) for h in lstm_hid], 0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_hid_linear], 0)

        return probs