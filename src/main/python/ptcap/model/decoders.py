import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from ptcap.tensorboardY import forward_hook_closure


class CoupledLSTMDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size,
                 num_lstm_layers, vocab_size, num_step=13, fc_size=37):

        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.mapping = nn.Linear(fc_size, hidden_size) #TODO: Fix this number
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.num_step = num_step
        self.lstm = nn.LSTM(self.embedding_size+self.hidden_size,
                            self.hidden_size, self.num_lstm_layers,
                            batch_first=True)

    def init_hidden(self, features):

        augmented_features = features.unsqueeze(0)
        expansion_size = features.size()
        c0 = h0 = augmented_features.expand(self.num_lstm_layers, *expansion_size)

        return h0.contiguous() , c0.contiguous()

    def forward(self, features, captions, use_teacher_forcing=False, lstm_hidden=None):

        if use_teacher_forcing:
            probs, lstm_hidden = self.run_teacher_forced(features, captions, lstm_hidden=lstm_hidden)
        else:
            probs, lstm_hidden = self.run_non_teacher_forced(features, captions, num_step=self.num_step, lstm_hidden=lstm_hidden)

        return probs, lstm_hidden

    def run_non_teacher_forced(self, features, go_tokens, num_step=1, lstm_hidden=None):
        lstm_input = go_tokens
        output_probs = []

        for i in range(num_step):
            probs, lstm_hidden = self.run_one_step(features, lstm_input,
                                                 lstm_hidden)

            output_probs.append(probs)
            # Greedy decoding
            _, preds = torch.max(probs, dim=2)

            lstm_input = preds

        concatenated_probs = torch.cat(output_probs, dim=1)
        return concatenated_probs, lstm_hidden

    def run_teacher_forced(self, features, captions, lstm_hidden=None):
        relued_features = F.relu(self.mapping(features))
        pooled_features = relued_features.mean(dim=1)
        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(pooled_features)
        embedded_captions = self.embedding(captions)
        lstm_input = self.prepare_lstm_input(embedded_captions,
                                             pooled_features)

        self.lstm.flatten_parameters()
        lstm_output, lstm_hidden = self.lstm(lstm_input, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h)
                                          for h in lstm_output], 0)
        probs = torch.stack(
            [self.logsoftmax(h) for h in lstm_out_projected], 0)

        return probs, lstm_hidden

    def run_one_step(self, features, captions, lstm_hidden=None):
        if captions.size()[1] > 1:
            print("WARNING: I'm using only the first token of your input")
            captions = captions[:, 0:1]

        return self.run_teacher_forced(features, captions, lstm_hidden=lstm_hidden)

    def prepare_lstm_input(self, embedded_captions, features):
        batch_size, seq_len, _ = embedded_captions.size()
        unsqueezed_features = features.unsqueeze(1)
        expansion_size = [batch_size, seq_len, unsqueezed_features.size(2)]

        expanded_features = unsqueezed_features.expand(*expansion_size)
        lstm_input = torch.cat([embedded_captions, expanded_features], dim=2)
        return lstm_input

