from torch import nn


class Decoder(nn.Module):

    def forward(self, decoder_states, teacher_targets=None):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError


class FullyConnectedDecoder(Decoder):

    def __init__(self, mapped_dim, decoded_dim):
        super(FullyConnectedDecoder, self).__init__()
        max_len, self.vocab_size = decoded_dim
        self.linear = nn.Linear(mapped_dim, max_len*self.vocab_size)

    def forward(self, data_batch):
        batch_size = data_batch.size()[0]
        fc_layer = self.linear(data_batch)
        return fc_layer.view(batch_size, -1, self.vocab_size)
