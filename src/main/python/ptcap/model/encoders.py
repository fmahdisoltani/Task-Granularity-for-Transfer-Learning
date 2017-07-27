from torch import nn

from rtorchn.models.gestures import gesture_net_rnn


class Encoder(nn.Module):

    def forward(self, video_batch):
        """BxCxTxWxH -> BxM"""
        raise NotImplementedError


class FullyConnectedEncoder(Encoder):

    def __init__(self, video_dims, num_features):
        super(FullyConnectedEncoder, self).__init__()
        C, T, W, H = video_dims
        self.linear = nn.Linear(C * T * W * H, num_features)

    def forward(self, video_batch):
        batch_size = video_batch.size()[0]
        return self.linear(video_batch.view(batch_size, -1))


class RtorchnEncoder(Encoder):

    def __init__(self, video_dims, num_features):
        super(RtorchnEncoder).__init__()

    def forward(self, video_batch):
        return gesture_net_rnn.forward()
