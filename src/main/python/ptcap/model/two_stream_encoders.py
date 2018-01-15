import torch
import torch.nn as nn

from torch.autograd import Variable

from ptcap.model.layers import CNN3dLayer
from ptcap.model.encoders import Encoder, C2dEncoder, C3dEncoder
from ptcap.tensorboardY import forward_hook_closure


class TwoStreamEncoder(Encoder):
    def __init__(self, encoder_output_size=52, num_features=128, gpus=None):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.c3d_encoder = C3dEncoder()
        self.c2d_encoder = C2dEncoder()
        lstm_hidden_size = self.encoder_output_size

        self.lstm = nn.LSTM(input_size=384, hidden_size=lstm_hidden_size,
                            num_layers=1, batch_first=True,
                            bidirectional=True)

        self.activations = {}


    def extract_features(self, videos):
        # Video encoding
        c3d_features = self.c3d_encoder.extract_features(videos) #8*48*128
        c2d_features = self.c2d_encoder.extract_features(videos) #8*48*256
        h = torch.cat((c3d_features, c2d_features), 2) #8*48*384
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(h)  #lstm_outputs: [8*48*1024]


        return lstm_outputs

    #def register_forward_hooks(self):
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


