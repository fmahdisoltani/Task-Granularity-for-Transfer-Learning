from torch import nn


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
