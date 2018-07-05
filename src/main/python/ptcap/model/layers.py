import torch
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
                                     (t-1)*dilation, 0), value=0)
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


class StatefulLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

        self.lstm_hidden = None

    def forward(self, lstm_input):
        lstm_output, lstm_hidden = self.lstm(lstm_input, self.lstm_hidden)
        self.lstm_hidden = lstm_hidden
        return lstm_output, lstm_hidden

    def reset(self):
        self.lstm_hidden = None

class SlantedC3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                  batchnorm=True):

        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, (3,5,5),
                              stride, padding, dilation, groups, bias)
        #self.conv.weight[:, :, 0, 0, 0] = 0.

        for dim0 in range(out_channels):
            for dim1 in range(in_channels):
                for dim2 in range(3):
                    for dim3 in (3, 4):
                        for dim4 in range(5):
                            self.conv.weight[
                                dim0, dim1, dim2, dim3, dim4].data = torch.Tensor(
                                [0])

                            #self.conv.weight[dim0,dim1,dim2, dim4, dim3] = 0
                            #var_no_grad = self.conv.weight[dim0, dim1, dim2, dim4, dim3].detach()
                           # self.conv.weight[dim0, dim1, dim2, dim3, dim4].detach()

                            #self.conv.weight[dim0, dim1, dim2, dim4, dim3].requires_grad = False





        self.activation = activation
        self.out_channels = out_channels
        self.in_channels = in_channels

        if batchnorm:
            self.batchnorm = nn.BatchNorm3d(out_channels)
        else:
            self.batchnorm = None

    def forward(self, x):
        h = self.conv(x)
        if self.batchnorm:
            h = self.batchnorm(h)

        return self.activation(h)

    def slant(self):
        for dim0 in range(self.out_channels):
            for dim1 in range(self.in_channels):
                for dim2 in range(3):
                    for dim3 in (3, 4):
                        for dim4 in range(5):
                            self.conv.weight[
                                dim0, dim1, dim2, dim3, dim4].data = torch.Tensor(
                                [0])

