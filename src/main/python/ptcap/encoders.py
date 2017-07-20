from torch import nn


class Encoder(nn.Module):

    def forward(self, video_batch):
        """BxCxTxWxH -> BxM"""
        raise NotImplementedError


class FullyConnectedEncoder(Encoder):

    def __init__(self, input_dim, output_dim):
        super(FullyConnectedEncoder, self).__init__()
        C, T, W, H = input_dim
        self.linear = nn.Linear(C * T * W * H, output_dim)

    def forward(self, data_batch):
        batch_size = data_batch.size()[0]
        return self.linear(data_batch.view(batch_size, -1))
