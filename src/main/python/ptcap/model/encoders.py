import torch
import torch.nn as nn

from torch.autograd import Variable

from ptcap.model.layers import CNN3dLayer
from ptcap.tensorboardY import forward_hook_closure


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

        self.activations = self.register_forward_hooks()

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

        h = self.pool4(h)  # batch_size * num_features * num_step * w * h

        h = h.mean(2)
        h = h.view(h.size()[0:2])

        return h

    def register_forward_hooks(self):
        master_dict = {}
        self.conv1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv1"))
        self.conv2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv2"))
        self.conv3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv3"))
        self.conv4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv4"))
        self.conv5.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv5"))
        self.conv6.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv6"))
        self.pool1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool1"))
        self.pool2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool2"))
        self.pool3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool3"))
        self.pool4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool4"))
        return master_dict


class CNN3dLSTMEncoder(Encoder):
    def __init__(self, encoder_output_size=52, gpus=None):
        """
        num_features: defines the output size of the encoder
        """

        super(CNN3dLSTMEncoder, self).__init__()

        self.num_layers = 1
        self.num_features = encoder_output_size
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

        self.activations = self.register_forward_hooks()

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.num_features))
        c0 = Variable(torch.zeros(1, batch_size, self.num_features))
        if self.use_cuda:
            h0 = h0.cuda(self.gpus[0])
            c0 = c0.cuda(self.gpus[0])
        return (h0, c0)

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

        h = h.view(h.size()[0:3])
        h = h.permute(0, 2, 1)  # batch_size * num_step * num_features

        lstm_hidden = self.init_hidden(batch_size=h.size()[0])

        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(h)

        h_mean = torch.mean(lstm_outputs, dim=1)

        return h_mean

    def register_forward_hooks(self):
        master_dict = {}
        self.conv1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv1"))
        self.conv2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv2"))
        self.conv3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv3"))
        self.conv4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv4"))
        self.conv5.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv5"))
        self.conv6.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv6"))
        self.pool1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool1"))
        self.pool2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool2"))
        self.pool3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool3"))
        self.pool4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool4"))
        self.lstm.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_lstm", 0, True))
        return master_dict
