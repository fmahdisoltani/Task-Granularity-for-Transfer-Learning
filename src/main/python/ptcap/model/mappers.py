from torch import nn


class Mapper(nn.Module):

    def forward(self, feature_batch):
        """BxM -> BxD"""
        raise NotImplementedError


class FullyConnectedMapper(Mapper):

    def __init__(self, num_features, decoder_state_dim):
        super(FullyConnectedMapper, self).__init__()
        self.linear = nn.Linear(num_features, decoder_state_dim)

    def forward(self, data_batch):
        return self.linear(data_batch)
