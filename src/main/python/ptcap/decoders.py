from torch import nn


class Decoder(nn.Module):

    def forward(self, decoder_states, teacher_targets=None):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError


class FullyConnectedDecoder(Decoder):

    def __init__(self, input_dim, output_dim):
        super(FullyConnectedDecoder, self).__init__()
        self.max_len, self.vocab_size = output_dim
        self.input_mapping = nn.Linear(input_dim,
                                       self.max_len * self.vocab_size)
        self.caption_mapping = nn.Linear(self.max_len * self.vocab_size,
                                         self.max_len * self.vocab_size)

    def forward(self, data_batch, teacher_captions=None):
        batch_size = data_batch.size()[0]
        if teacher_captions is not None:
            predictions = self.caption_mapping(
                teacher_captions.view(batch_size, -1)) + \
                          self.input_mapping(data_batch)
        else:
            predictions = self.input_mapping(data_batch)
        return predictions.view(batch_size, -1, self.vocab_size)
