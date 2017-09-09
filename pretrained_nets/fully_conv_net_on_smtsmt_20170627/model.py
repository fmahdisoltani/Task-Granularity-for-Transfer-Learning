import torch.nn as nn

from rtorchn.core.layers import CustomConv3D, Glu1D
from rtorchn.core.abstract_nets import ClassifierNet


class FullyConvolutionalNet(ClassifierNet):
    def __init__(self, num_classes, num_features=256, **kwargs):
        super(FullyConvolutionalNet, self).__init__(num_classes, num_features, **kwargs)

        self.conv1 = CustomConv3D(3, 16, (3, 3, 3), nn.ReLU(), stride=1, padding=1, batchnorm=True)
        self.conv2 = CustomConv3D(16, 32, (3, 3, 3), nn.ReLU(), stride=1, padding=1, batchnorm=True)
        self.conv3 = CustomConv3D(32, 64, (3, 3, 3), nn.ReLU(), stride=1, padding=1, batchnorm=True)

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CustomConv3D(64, 128, (3, 3, 3), nn.ReLU(), stride=1, padding=(1, 0, 0), batchnorm=True)
        self.conv5 = CustomConv3D(128, 128, (3, 3, 3), nn.ReLU(), stride=1, padding=(1, 0, 0), batchnorm=True)
        self.conv6 = CustomConv3D(128, 128, (3, 3, 3), nn.ReLU(), stride=1, padding=(1, 0, 0), batchnorm=True)

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        self.glu_block_a = self.glu_block(128, 256, 256)
        self.glu_block_b = self.glu_block(256, 512, 256)
        self.glu_block_c = self.glu_block(256, 512, num_features)

    def glu_block(self, num_in, num_middle, num_out):
        return nn.ModuleList([Glu1D(num_in, num_middle, 3, padding=1, batchnorm=True),
                              Glu1D(num_middle, num_middle, 3, padding=2, dilation=2, batchnorm=True),
                              Glu1D(num_middle, num_out, 3, padding=1, batchnorm=True),
                              nn.MaxPool1d(2)])

    def default_input_size(self):
        return (1, 3, 36, 96, 96)

    def extract_features(self, x):
        h = self.conv1(x)
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

        for layer in self.glu_block_a:
            h = layer(h)
        for layer in self.glu_block_b:
            h = layer(h)
        for layer in self.glu_block_c:
            h = layer(h)

        return h.permute(0, 2, 1)
