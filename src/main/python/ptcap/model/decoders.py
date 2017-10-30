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
                 num_hidden_lstm, num_step):

        super(Decoder, self).__init__()
        self.num_hidden_lstm = num_hidden_lstm

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, 1,
                            batch_first=True)

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

    def __init__(self, embedding_size, hidden_size, num_step, vocab_size,
                 num_lstm_layers, dropout=0.5):
        super(Decoder, self).__init__()

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, 1,
                            batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax()
        self.num_step = num_step

        self.activations = self.register_forward_hooks()
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        #####################################################

        self.attn_softmax = torch.nn.Softmax()

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

        if use_teacher_forcing:

            probs, hidden = self.apply_lstm(encoder_outputs, captions,
                                               self.num_step)

        else:
            # Without teacher forcing: use its own predictions as the next input
            probs = self.apply_lstm(encoder_outputs, captions, self.num_step)

        return probs

    def apply_lstm(self, encoder_outputs, captions, max_len):

        last_hidden = None

        for i in range(max_len):
            probs, last_hidden, _ = self.apply_attention(encoder_outputs,
                                                         captions, last_hidden)

        return probs, last_hidden

    def apply_attention(self, encoder_outputs, captions, last_hidden):

        embedded_captions = self.embedding(captions)
        embedded_captions = self.dropout(embedded_captions)

        # Calculate the attention weights and apply the to the encoder's outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_input = torch.cat((embedded_captions, context), 2)
        output, hidden = self.lstm(rnn_input, last_hidden)

        output = output.squeeze(0)
        # Try context vs rnn_input
        output = F.log_softmax(self.linear(torch.cat((output, context), 1)))
        # vs
        # output = F.log_softmax(self.linear(torch.cat((output, rnn_input), 1)))

        return output, hidden, attn_weights

    def get_attention(self, encoder_outputs, embedding, state):
        attn_scores = Variable(torch.zeros(self.encoder_seq_len))
        # if USE_CUDA: attn_energies = attn_energies.cuda()
        for i in range(self.encoder_seq_len):
            attn_scores[i] = self.alignment(state, encoder_outputs[i])
        attn_weights = self.attn_softmax(attn_scores)
        context = torch.mm(attn_weights, encoder_outputs)
        next_state = self.lstm(torch.cat([embedding, context], 1), state)
        output = g(embedding, next_state, context)


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
