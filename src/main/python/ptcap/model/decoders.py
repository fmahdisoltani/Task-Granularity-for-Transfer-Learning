import torch

from torch import nn

from ptcap.tensorboardY import forward_hook_closure


class Decoder(nn.Module):
    def forward(self, decoder_states, teacher_captions,
                use_teacher_forcing=False):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError


class DecoderBase(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size,
                 num_lstm_layers, num_step):

        super().__init__()
        self.num_lstm_layers = num_lstm_layers

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax()
        self.num_step = num_step

        self.activations = self.register_forward_hooks()

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
        teacher forcing. It should be noted that the <GO> token is assumed to be
        present in the input captions.
        Args:
            features: Video features extracted by the encoder.
            captions: Video captions (required if use_teacher_forcing=True).
            use_teacher_forcing: Whether to use teacher forcing or not.
        Returns:
            The probability distribution over the vocabulary across the entire
            sequence.
        """

        if use_teacher_forcing:
            probs, _ = self.apply_lstm(features, captions)

        else:
            # Without teacher forcing: use its own predictions as the next input
            probs = self.predict(features, captions, self.num_step)

        return probs

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

    def register_forward_hooks(self):
        master_dict = {}
        self.embedding.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_embedding"))
        self.linear.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_linear"))
        self.logsoftmax.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_logsoftmax"))
        return master_dict

    def apply_lstm(self):
        raise NotImplementedError("apply_lstm should be implemented")


class LSTMDecoder(DecoderBase):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_lstm_layers,
                 num_step):

        super().__init__(embedding_size, hidden_size, vocab_size,
                         num_lstm_layers, num_step)
        # batch_first: whether input and output are (batch, seq, feature)
        print (num_lstm_layers)
        print("****"*10)
        self.lstm = nn.LSTM(embedding_size, hidden_size,num_lstm_layers , batch_first=True)

    def apply_lstm(self, features, captions, lstm_hidden=None):
        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)
        embedded_captions = self.embedding(captions)
        lstm_output, lstm_hidden = self.lstm(embedded_captions, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output],
                                         0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_out_projected], 0)

        return probs, lstm_hidden


class CoupledLSTMDecoder(DecoderBase):
    def __init__(self, embedding_size, hidden_size, vocab_size,
                 num_hidden_lstm, num_step):

        super().__init__(embedding_size, hidden_size, vocab_size,
                         num_hidden_lstm, num_step)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, 1,
                            batch_first=True)

    def apply_lstm(self, features, captions, lstm_hidden=None):
        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)
        embedded_captions = self.embedding(captions)
        batch_size, seq_len, _ = embedded_captions.size()

        expansion_size = [batch_size, seq_len, features.size(2)]
        expanded_lstm_hidden = features.expand(*expansion_size)
        lstm_input = torch.cat([embedded_captions, expanded_lstm_hidden], dim=2)

        self.lstm.flatten_parameters()
        lstm_output, lstm_hidden = self.lstm(lstm_input, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output],
                                         0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_out_projected], 0)

        return probs, lstm_hidden
