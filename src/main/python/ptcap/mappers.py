from torch import nn


class Mapper(nn.Module):

    def forward(self, feature_batch):
        """BxM -> BxD"""
        raise NotImplementedError
