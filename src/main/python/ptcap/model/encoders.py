import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable

from ptcap.model.layers import CNN3dLayer
from ptcap.tensorboardY import register_grad, update_dict


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
    def __init__(self, num_features=128, gpus=None):
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

        self.gradients = {}
        self.hidden = {}

    def forward(self, videos):
        # Video encoding
        self.hidden["conv1_layer"] = self.conv1(videos)
        self.hidden["pool1_layer"] = self.pool1(self.hidden["conv1_layer"])

        self.hidden["conv2_layer"] = self.conv2(self.hidden["pool1_layer"])
        self.hidden["pool2_layer"] = self.pool2(self.hidden["conv2_layer"])

        self.hidden["conv3_layer"] = self.conv3(self.hidden["pool2_layer"])
        self.hidden["pool3_layer"] = self.pool3(self.hidden["conv3_layer"])

        self.hidden["conv4_layer"] = self.conv4(self.hidden["pool3_layer"])
        self.hidden["conv5_layer"] = self.conv5(self.hidden["conv4_layer"])
        self.hidden["conv6_layer"] = self.conv6(self.hidden["conv5_layer"])

        # batch_size * num_features * num_step * w * h
        self.hidden["pool4_layer"] = self.pool4(self.hidden["conv6_layer"])

        self.hidden["mean_pool"] = self.hidden["pool4_layer"].mean(2)
        self.hidden["features"] = self.hidden["mean_pool"].view(
                                self.hidden["mean_pool"].size()[0:2])

        register_grad(self.gradients, self.hidden.items())

        return self.hidden["features"]


class CNN3dLSTMEncoder(Encoder):
    def __init__(self, num_features=128, gpus=None):
        """
        num_features: defines the output size of the encoder
        """

        super(CNN3dLSTMEncoder, self).__init__()

        self.num_layers = 1
        self.num_features = num_features
        self.use_cuda = True if gpus else False
        self.gpus = gpus
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
        self.conv6 = CNN3dLayer(128, 128, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=(1, 0, 0))

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        self.lstm = nn.LSTM(input_size=128, hidden_size=self.num_features,
                            num_layers=self.num_layers, batch_first=True)

        self.gradients = {}
        self.hidden = {}

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.num_features))
        c0 = Variable(torch.zeros(1, batch_size, self.num_features))
        if self.use_cuda:
            h0 = h0.cuda(self.gpus[0])
            c0 = c0.cuda(self.gpus[0])
        return (h0, c0)

    def forward(self, videos):
        # Video encoding
        self.hidden["conv1_layer"] = self.conv1(videos)
        self.hidden["pool1_layer"] = self.pool1(self.hidden["conv1_layer"])

        self.hidden["conv2_layer"] = self.conv2(self.hidden["pool1_layer"])
        self.hidden["pool2_layer"] = self.pool2(self.hidden["conv2_layer"])

        self.hidden["conv3_layer"] = self.conv3(self.hidden["pool2_layer"])
        self.hidden["pool3_layer"] = self.pool3(self.hidden["conv3_layer"])

        self.hidden["conv4_layer"] = self.conv4(self.hidden["pool3_layer"])
        self.hidden["conv5_layer"] = self.conv5(self.hidden["conv4_layer"])
        self.hidden["conv6_layer"] = self.conv6(self.hidden["conv5_layer"])

        # batch_size * num_features * num_step * w * h
        self.hidden["pool4_layer"] = self.pool4(self.hidden["conv6_layer"])

        h = self.hidden["pool4_layer"].view(
                                        self.hidden["pool4_layer"].size()[0:3])
        h = h.permute(0, 2, 1)  # batch_size * num_step * num_features

        lstm_hidden = self.init_hidden(batch_size=h.size(0))
        lstm_outputs, lstm_hidden = (self.lstm(h, lstm_hidden))

        self.hidden["features"] = torch.mean(lstm_outputs, dim=1)

        vars_tuple = [("encoder_lstm_outputs", lstm_outputs)]

        register_grad(self.gradients, self.hidden.items())
        update_dict(self.hidden, vars_tuple, h.size(1))
        register_grad(self.gradients, vars_tuple, h.size(1))

        return self.hidden["features"]
