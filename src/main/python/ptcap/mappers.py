from torch import nn


class Mapper(nn.Module):

    def forward(self, feature_batch):
        """BxM -> BxD"""
        raise NotImplementedError


class FullyConnectedMapper(Mapper):

    def __init__(self, input_dim, output_dim):
        super(FullyConnectedMapper, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, data_batch):
        return self.linear(data_batch)
