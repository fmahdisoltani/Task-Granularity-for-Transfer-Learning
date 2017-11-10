import torch.nn as nn


class CNN3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                  batchnorm=True):

        super(CNN3dLayer, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        self.activation = activation

        if batchnorm:
            self.batchnorm = nn.BatchNorm3d(out_channels)
        else:
            self.batchnorm = None

    def forward(self, x):
        h = self.conv(x)
        if self.batchnorm:
            h = self.batchnorm(h)

        return self.activation(h)

