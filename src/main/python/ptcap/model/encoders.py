import torch
import torch.nn as nn

from ptcap.model.layers import CNN3dLayer
from torch.autograd import Variable


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


class CNN3dEncoder(Encoder):
    def __init__(self, num_features=128):
        super(CNN3dEncoder, self).__init__()

        self.conv1 = CNN3dLayer(3, 16, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv2 = CNN3dLayer(16, 32, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv3 = CNN3dLayer(32, 64, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)

        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CNN3dLayer(64, 128, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))
        self.conv5 = CNN3dLayer(128, 128, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))
        self.conv6 = CNN3dLayer(128, num_features, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))

        self.pool4 = nn.MaxPool3d((1, 6, 6))

    def forward(self, videos):
        # Video encoding
        h = self.conv1(videos)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.pool3(h)

        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)

        h = self.pool4(h)

        h = h.squeeze().permute(2, 0, 1)  # num_step * batch_size * embed_size

        h = h.mean(0).squeeze()

        return h
