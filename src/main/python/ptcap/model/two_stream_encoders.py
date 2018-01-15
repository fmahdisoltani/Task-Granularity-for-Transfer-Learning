import torch
import torch.nn as nn

from torch.autograd import Variable

from ptcap.model.layers import CNN3dLayer
from ptcap.model.encoders import Encoder, C2dEncoder, C3dLSTMEncoder
from ptcap.tensorboardY import forward_hook_closure


class TwoStreamEncoder(Encoder):
    def __init__(self, encoder_output_size=52, num_features=128, gpus=None):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.c3d_encoder = C3dLSTMEncoder()
        self.c2d_encoder = C2dEncoder()

        #self.activations = self.register_forward_hooks()

    def extract_features(self, videos):
        # Video encoding
        c3d_features = self.c3d_encoder.extract_features(videos)
        c2d_features = self.c2d_encoder.extract_features(videos)
        h = c3d_features +c2d_features
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


