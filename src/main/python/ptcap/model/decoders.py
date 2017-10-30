import torch
import torch.functional as F

from torch import nn
from torch.autograd import Variable

from ptcap.tensorboardY import forward_hook_closure


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
                 num_lstm_layers, num_step):

        super(LSTMDecoder, self).__init__()
        self.num_lstm_layers = num_lstm_layers

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size, hidden_size, 1, batch_first=True)

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

    def apply_lstm(self, features, captions, lstm_hidden=None):

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)
        embedded_captions = self.embedding(captions)

        self.lstm.flatten_parameters()
        lstm_output, lstm_hidden = self.lstm(embedded_captions, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output],
                                         0)
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

    def register_forward_hooks(self):
        master_dict = {}
        self.embedding.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_embedding"))
        self.lstm.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_lstm", 0, False))
        self.linear.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_linear"))
        self.logsoftmax.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_logsoftmax"))
        return master_dict


class CoupledLSTMDecoder(Decoder):

    def __init__(self, embedding_size, hidden_size, vocab_size,
                 num_lstm_layers, num_step):

        super(Decoder, self).__init__()

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size,
                            num_lstm_layers, batch_first=True)

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

    def apply_lstm(self, features, captions, lstm_hidden=None):

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)
        embedded_captions = self.embedding(captions)
        batch_size, seq_len, _ = embedded_captions.size()
        altered_lstm_hidden = lstm_hidden[0][0].unsqueeze(1)
        expansion_size = [batch_size, seq_len, altered_lstm_hidden.size(2)]
        expanded_lstm_hidden = altered_lstm_hidden.expand(*expansion_size)
        lstm_input = torch.cat([embedded_captions, expanded_lstm_hidden], dim=2)

        self.lstm.flatten_parameters()
        lstm_output, lstm_hidden = self.lstm(lstm_input, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output],
                                         0)
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

    def register_forward_hooks(self):
        master_dict = {}
        self.embedding.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_embedding"))
        self.lstm.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_lstm", 0, False))
        self.linear.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_linear"))
        self.logsoftmax.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_logsoftmax"))
        return master_dict


class AttentionDecoder(Decoder):

    def __init__(self, encoder_hidden_size, embedding_size,
                 hidden_size, num_step, vocab_size, num_lstm_layers, dropout=0):
        super(Decoder, self).__init__()

        cell_type = nn.LSTM

        factor = 2 if "LSTM" in str(cell_type) else 1

        self.attn_softmax = torch.nn.Softmax()
        self.dropout = nn.Dropout(dropout)

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.hidden_size = hidden_size

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax()

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = cell_type(embedding_size + encoder_hidden_size, hidden_size,
                              num_lstm_layers, batch_first=True)

        self.num_step = int(num_step)

        self.alignment = nn.Linear(self.hidden_size * factor +
                                   encoder_hidden_size, 1)

        self.activations = self.register_forward_hooks()

    def forward(self, encoder_outputs, captions, use_teacher_forcing=False):
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

        last_hidden = None

        output_probs = []
        token = captions[:, 0]

        for i in range(self.num_step):
            probs, last_hidden, _ = self.apply_attention(encoder_outputs,
                                                            token, last_hidden)
            output_probs.append(probs.unsqueeze(1))
            if use_teacher_forcing:
                token = captions[:, i]
            else:
                _, token = torch.max(probs, dim=1)

        concatenated_probs = torch.cat(output_probs, dim=1)
        return concatenated_probs

    def init_hidden(self, features):
        """
        Hidden states of the LSTM are initialized with features.
        c0 and h0 should have the shape of 1 * batch_size * hidden_size
        """

        c0 = features.unsqueeze(0)
        h0 = features.unsqueeze(0)
        return h0, c0

    def apply_attention(self, encoder_outputs, captions, last_hidden):

        embedded_captions = self.embedding(captions)
        embedded_captions = self.dropout(embedded_captions)

        if last_hidden is None:
            last_hidden = self.init_hidden(encoder_outputs[:, -1])

        output, hidden, attn_weights = self.get_attention(encoder_outputs,
                                                          embedded_captions,
                                                          last_hidden)

        return output, hidden, attn_weights

    def get_attention(self, encoder_outputs, embedding, state):
        batch_size, encoder_seq_len = encoder_outputs.size()[0:2]
        attn_scores = Variable(torch.zeros(batch_size, encoder_seq_len))
        flat_state = torch.cat(state, 2).squeeze(0)
        # if USE_CUDA: attn_energies = attn_energies.cuda()
        for i in range(encoder_seq_len):
            current_output = encoder_outputs[:, i]
            cat_states = torch.cat((flat_state, current_output), 1)
            attn_scores[:, i] = self.alignment(cat_states)
        attn_weights = self.attn_softmax(attn_scores)
        transposed_encoder_outputs = encoder_outputs.transpose(2, 1)
        context = transposed_encoder_outputs.bmm(attn_weights.unsqueeze(2))
        context_and_embedding = torch.cat([context.squeeze(2), embedding], 1)
        final_output, next_hidden = self.lstm(context_and_embedding.unsqueeze(1), state)
        output = self.logsoftmax(self.linear(final_output.squeeze(1)))
        return output, next_hidden, attn_weights

    def register_forward_hooks(self):
        master_dict = {}
        self.embedding.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_embedding"))
        self.lstm.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_lstm", 0, False))
        self.linear.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_linear"))
        self.logsoftmax.register_forward_hook(
            forward_hook_closure(master_dict, "decoder_logsoftmax"))
        return master_dict
