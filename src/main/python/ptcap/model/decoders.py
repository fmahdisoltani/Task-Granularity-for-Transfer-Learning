import torch

from torch import nn

from ptcap.tensorboardY import forward_hook_closure


class Decoder(nn.Module):
    def forward(self, decoder_states, teacher_captions,
                use_teacher_forcing=False):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError


class DecoderBase(nn.Module):
    def __init__(self, encoder_hidden_size, embedding_size, hidden_size,
                 num_lstm_layers, vocab_size, num_step=13, dropout=0):

        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.mapper = nn.Linear(self.encoder_hidden_size, self.hidden_size * 2
                                * num_lstm_layers)
        self.num_lstm_layers = num_lstm_layers
        self.num_step = num_step
        self.relu = nn.ReLU()

        # Embed each token in vocab to a 128 dimensional vector
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.activations = self.register_forward_hooks()
        self.announce()

    def announce(self):
        print("Instantiating {}".format(self.__class__.__name__))

    def init_hidden(self, encoder_outputs):
        """
        Hidden states of the LSTM are initialized with encoder_outputs.
        c0 and h0 should have the shape of num_lstm_layers * batch_size * hidden_size
        """

        mapped_state = self.relu(self.mapper(encoder_outputs))
        init_state = mapped_state.view([self.num_lstm_layers, encoder_outputs.size(0),
                                           self.hidden_size*2])

        h_state, c_state = (init_state[:, :, :self.hidden_size],
                            init_state[:, :, self.hidden_size:])

        c0 = c_state.contiguous()
        h0 = h_state.contiguous()
        return h0, c0

    def forward(self, encoder_outputs, captions, use_teacher_forcing=False):
        """
        This method computes the forward pass of the decoder with or without
        teacher forcing. It should be noted that the <GO> token is assumed to be
        present in the input captions.
        Args:
            encoder_outputs: Video features extracted by the encoder.
            captions: Video captions (required if use_teacher_forcing=True).
            use_teacher_forcing: Whether to use teacher forcing or not.
        Returns:
            The probability distribution over the vocabulary across the entire
            sequence.
        """

        input_features = encoder_outputs.mean(dim=1)

        if use_teacher_forcing:
            probs, _ = self.apply_lstm(input_features, captions)

        else:
            # Without teacher forcing: use its own predictions as the next input
            probs = self.predict(input_features, captions, self.num_step)

        return probs

    def predict(self, features, go_tokens, num_step):
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
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,
                            self.num_lstm_layers, batch_first=True)

    def apply_lstm(self, features, captions, lstm_hidden=None):
        embedded_captions = self.embedding(captions)
        embedded_captions = self.dropout(embedded_captions)

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)

        lstm_output, lstm_hidden = self.lstm(embedded_captions, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h) for h in lstm_output],
                                         0)
        probs = torch.stack([self.logsoftmax(h) for h in lstm_out_projected], 0)

        return probs, lstm_hidden


class CoupledLSTMDecoder(DecoderBase):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(self.embedding_size+self.encoder_hidden_size,
                            self.hidden_size, self.num_lstm_layers,
                            batch_first=True)

    def apply_lstm(self, features, captions, lstm_hidden=None):
        embedded_captions = self.embedding(captions)
        embedded_captions = self.dropout(embedded_captions)

        if lstm_hidden is None:
            lstm_hidden = self.init_hidden(features)

        lstm_input = self.prepare_lstm_input(embedded_captions, features)

        self.lstm.flatten_parameters()
        lstm_output, lstm_hidden = self.lstm(lstm_input, lstm_hidden)

        # Project features in a 'vocab_size'-dimensional space
        lstm_out_projected = torch.stack([self.linear(h)
                                          for h in lstm_output], 0)
        probs = torch.stack([self.logsoftmax(h)
                             for h in lstm_out_projected], 0)

        return probs, lstm_hidden

    def prepare_lstm_input(self, embedded_captions, features):
        batch_size, seq_len, _ = embedded_captions.size()
        unsqueezed_features = features.unsqueeze(1)
        expansion_size = [batch_size, seq_len, unsqueezed_features.size(2)]
        expanded_features = unsqueezed_features.expand(*expansion_size)
        lstm_input = torch.cat([embedded_captions, expanded_features], dim=2)
        return lstm_input


class AttentionDecoder(DecoderBase):
    def __init__(self, alignment_size, *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # batch_first: whether input and output are (batch, seq, feature)
        self.lstm = nn.LSTM(self.embedding_size+self.encoder_hidden_size,
                            self.hidden_size, self.num_lstm_layers,
                            batch_first=True)
        self.tanh = nn.Tanh()

        if int(torch.__version__.split('.')[1]) < 3:
            self.attn_softmax = nn.Softmax()
        else:
            self.attn_softmax = nn.Softmax(dim=-1)

        # Alignment model that computes similarity vectors
        self.alignment = nn.Linear(self.hidden_size * 2 +
                                   self.encoder_hidden_size, alignment_size)

        # Maps the similarity vectors to a similarity score before softmax
        self.pre_attn = nn.Linear(alignment_size, 1, bias=False)

    def forward(self, encoder_outputs, captions, use_teacher_forcing=False):
        """
        This method computes the forward pass of the decoder with or without
        teacher forcing. It should be noted that the <GO> token is
        automatically appended to the input captions.
        Args:
            encoder_outputs: Video features extracted by the encoder.
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
            output_probs.append(probs)
            if use_teacher_forcing and i < self.num_step - 1:
                token = captions[:, i + 1]
            else:
                _, token = torch.max(probs, dim=1)

        concatenated_probs = torch.stack(output_probs, dim=1)
        return concatenated_probs

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
        # Compute the attention for every encoder output state
        attn_weights = self.get_attention_weights(encoder_outputs, state)

        # Get the context vector
        context = attn_weights.bmm(encoder_outputs)

        # Input context vector and embedding to LSTM
        lstm_input = torch.cat([context, embedding.unsqueeze(1)], 2)

        self.lstm.flatten_parameters()
        # lstm_input.contiguous()
        lstm_output, lstm_hidden = self.lstm(lstm_input, state)

        # Project features in a 'vocab_size'-dimensional space
        output = self.logsoftmax(self.linear(lstm_output.squeeze(1)))

        return output, lstm_hidden, attn_weights

    def get_attention_weights(self, encoder_outputs, state):
        lstm_state = state[0][0], state[1][0]
        flat_state = [torch.cat(lstm_state, 1)]
        tr_encoder_outputs = encoder_outputs.transpose(1, 0)

        alignment_scores = [[self.alignment(torch.cat((layer_state, output), 1))
                            for output in tr_encoder_outputs] for layer_state in flat_state]

        attn_scores = [[self.pre_attn(self.tanh(score)) for score in alignments]
                       for alignments in alignment_scores]

        cat_attn_scores = [torch.stack(attn_score).squeeze(2) for attn_score in attn_scores]
        cat_layer_scores = torch.stack(cat_attn_scores).transpose(1, 2)
        if int(torch.__version__.split('.')[1]) < 3:
            attn_weights = self.attn_softmax(cat_layer_scores.squeeze(0)).unsqueeze(0)
        else:
            attn_weights = self.attn_softmax(cat_layer_scores)
        return attn_weights.transpose(0, 1)
