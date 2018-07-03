import torch
import torch.nn as nn

from torch.autograd import Variable

from ptcap.model.layers import CNN3dLayer, CausalC3dLayer,SlantedC3dLayer
from ptcap.tensorboardY import forward_hook_closure


class C3dExtractor(nn.Module):
    def __init__(self, out_ch=32):
        super().__init__()

        self.conv1 = SlantedC3dLayer(3, out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv2 = CNN3dLayer(out_ch, 2 * out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv3 = CNN3dLayer(2 * out_ch, 4 * out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=1)

        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CNN3dLayer(4 * out_ch, 8 * out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=(1, 0, 0))
        self.conv5 = CNN3dLayer(8 * out_ch, 8 * out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))
        self.conv6 = CNN3dLayer(8 * out_ch, 8 * out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=(1, 0, 0))

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        self.activations = self.register_forward_hooks()

    def extract_features(self, videos):
        # Video encoding
        h = self.conv1(videos)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.pool3(h)  # [8,64,48,12,12]

        h = self.conv4(h)  # [8,128,48,10,10]
        h = self.conv5(h)  # [8,128,48,8,8]
        h = self.conv6(h)  # [8,128,48,6,6]

        h = self.pool4(h)  # batch_size * num_features * num_step * w * h

        h = h.view(h.size()[0:3])  # [batch_size*num_feature*num_step](8*128*48)
        h = h.permute(0, 2, 1)  # [batch_size*num_step*num_features]

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


class C2dExtractor(nn.Module):
    def __init__(self, encoder_output_size=52, gpus=None, out_ch=32,
                 bidirectional=True):
        """
        num_features: defines the output size of the encoder
        """

        super().__init__()

        self.conv1 = CNN3dLayer(3, out_ch, (1, 3, 3), nn.ReLU(),
                                stride=1, padding=(0, 1, 1))
        self.conv2 = CNN3dLayer(out_ch, 2 * out_ch, (1, 3, 3), nn.ReLU(),
                                stride=1, padding=(0, 1, 1))
        self.conv3 = CNN3dLayer(2 * out_ch, 4 * out_ch, (1, 3, 3),
                                nn.ReLU(), stride=1, padding=(0, 1, 1))

        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CNN3dLayer(4 * out_ch, 8 * out_ch, (1, 3, 3),
                                nn.ReLU(), stride=1, padding=0)
        self.conv5 = CNN3dLayer(8 * out_ch, 8 * out_ch, (1, 3, 3),
                                nn.ReLU(),
                                stride=1, padding=0)
        self.conv6 = CNN3dLayer(8 * out_ch, 8 * out_ch, (1, 3, 3),
                                nn.ReLU(), stride=1, padding=0)

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        #self.activations = self.register_forward_hooks()


    def extract_features(self, videos):
        # videos: [batch_size*num_ch*len*w*h] (8*3*48*96*96)

        h = self.conv1(videos)  # [batch_size*num_ch*len*w*h]=8*32*48*96*96
        h = self.pool1(h)  # 8*32*48*48 *48

        h = self.conv2(h)  # 8*64 *48*48*48
        h = self.pool2(h)  # 8*64 *48*24*24

        h = self.conv3(h)  # 8*128 *48*24*24
        h = self.pool3(h)  # 8*128 *48*12*12

        h = self.conv4(h)  # 8*256 *48*10*10
        h = self.conv5(h)  # 8*256 *48*8*8
        h = self.conv6(h)  # 8*256 *48*6*6
        h = self.pool4(h)  # 8*256 *48*1*1

        h = h.view(h.size()[0:3])  # [batch_size*num_feature*num_step](8*256*48)
        h = h.permute(0, 2, 1)  # [batch_size*num_step*num_features]

        return h  # 8* 48 * 256


class CausalC3dExtractor(nn.Module):
    def __init__(self, out_ch=32):
        super().__init__()

        self.conv1 = CausalC3dLayer(3, out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv2 = CausalC3dLayer(out_ch, 2 * out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv3 = CausalC3dLayer(2 * out_ch, 4 * out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=1)

        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CausalC3dLayer(4 * out_ch, 8 * out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=0)
        self.conv5 = CausalC3dLayer(8 * out_ch, 8 * out_ch, (3, 3, 3),
                                nn.ReLU(),
                                stride=1, padding=0)
        self.conv6 = CausalC3dLayer(8 * out_ch, 8 * out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=0)

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        # self.activations = self.register_forward_hooks()

    def extract_features(self, videos):
        # Video encoding
        h = self.conv1(videos)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.pool3(h)  # [8,64,48,12,12]

        h = self.conv4(h)  # [8,128,48,10,10]
        h = self.conv5(h)  # [8,128,48,8,8]
        h = self.conv6(h)  # [8,128,48,6,6]

        h = self.pool4(h)  # batch_size * num_features * num_step * w * h

        h = h.view(
            h.size()[0:3])  # [batch_size*num_feature*num_step](8*128*48)
        h = h.permute(0, 2, 1)  # [batch_size*num_step*num_features]

        return h


