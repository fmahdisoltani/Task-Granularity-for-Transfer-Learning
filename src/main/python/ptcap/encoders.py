from torch import nn


class Encoder(nn.Module):

    def forward(self, video_batch):
        """BxCxTxWxH -> BxM"""
        raise NotImplementedError
