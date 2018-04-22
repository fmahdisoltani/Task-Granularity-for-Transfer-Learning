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


class C2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                  batchnorm=True):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        self.activation = activation

        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        else:
            self.batchnorm = None

    def forward(self, x):
        h = self.conv(x)
        if self.batchnorm:
            h = self.batchnorm(h)

        return self.activation(h)


class CausalC3dLayer(nn.Module):
    # video => [batch_size * num_ch * len * w * h]
    def __init__(self, in_channels, out_channels, kernel_size, activation,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                  batchnorm=True):

        super().__init__()
        t = kernel_size[0]

        self.pad = nn.ConstantPad3d((padding, padding, padding, padding,
                                     2*dilation, 0), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride, 0, dilation, groups, bias)
        self.activation = activation

        if batchnorm:
            self.batchnorm = nn.BatchNorm3d(out_channels)
        else:
            self.batchnorm = None

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.batchnorm:
            h = self.batchnorm(h)

        return self.activation(h)
