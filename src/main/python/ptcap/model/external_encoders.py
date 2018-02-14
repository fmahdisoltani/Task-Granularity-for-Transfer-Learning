import torch
import torch.nn as nn

from torchvision.models import vgg11_bn
from .encoders import Encoder
from rtorchn.core.networks import (FullyConvolutionalNet, JesterNet)
from rtorchn.core.networks import BiJesterNetII, JesterNetBase
from .encoders import Encoder
from rtorchn.core.networks.resnets import InflatedResNet18

import torch.nn.functional as F


class ExternalEncoder(Encoder):
    def __init__(self, encoder, encoder_args, pretrained_path=None,
                 checkpoint_key=None, freeze=False):

        super(ExternalEncoder, self).__init__()
        self.encoder = encoder(*encoder_args)

        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        self.activations = {}

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            if checkpoint_key is not None:
                self.encoder.load_state_dict(checkpoint[checkpoint_key])
            else:
                self.encoder.load_state_dict(checkpoint)

    def forward(self, video_batch):
        return self.encoder(video_batch)

    def predict_from_features(self, features):
        return self.encoder.predict_from_features(features)


class FCEncoder(ExternalEncoder):
    def __init__(self, pretrained_path=None, freeze=False):
        # Hardcoded encoder for using FullyConvolutionalNet from 20bn_rtorchn

        # it thinks it's getting num_features, but it's not. what is happening
        # is equivalent to FullyConvolutionalNet(..,
        #                                     num_features=encoder_output_size)

        encoder_output_size = 256
        num_classes = 178
        encoder_args = (num_classes, encoder_output_size)
        super(FCEncoder, self).__init__(encoder=FullyConvolutionalNet,
                                        encoder_args=encoder_args,
                                        pretrained_path=pretrained_path,
                                        freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)


class JesterEncoder(ExternalEncoder):
    """
        Hardcoded encoder for using JesterNet from 20bn_rtorchn
        num_classes = 329 means this class expects the "supermodel" version.
    """

    def __init__(self, pretrained_path=None, freeze=False):
        encoder_output_size = 256
        num_classes = 329
        encoder_args = (num_classes, encoder_output_size)
        super(JesterEncoder, self).__init__(encoder=JesterNet,
                                            encoder_args=encoder_args,
                                            pretrained_path=pretrained_path,
                                            freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)


class BIJesterEncoder_II(ExternalEncoder):

    """
            Hardcoded encoder for using BIJesterNet from 20bn_rtorchn
            num_classes = 174 means this class expects the "supermodel" version.
            Encoder output size in this version is 1024
    """

    def __init__(self, pretrained_path=None, freeze=False):
        self.encoder_output_size = 1024
        num_classes = 178
        kernel_base = 32
        bidirectional = True
        encoder_args = (num_classes, int(self.encoder_output_size/2), kernel_base,
                        bidirectional)

        # FC layer on top of features

        super().__init__(encoder=JesterNetBase,
                         encoder_args=encoder_args,
                         pretrained_path=pretrained_path,
                         freeze=freeze)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(int(self.encoder_output_size/2), self.encoder_output_size)

        # self.num_classes = 178
        # self.classif_layer = nn.Linear(1024, self.num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def extract_features(self, video_batch):

        features = self.encoder.extract_features(video_batch)
        return self.dropout(self.relu(self.fc(features)))


class BIJesterEncoder(ExternalEncoder):
    def __init__(self, pretrained_path=None, freeze=False):
        self.encoder_output_size = 1024
        relu_output_size = 1024
        num_classes = 178
        encoder_args = (num_classes, self.encoder_output_size)
        super(BIJesterEncoder, self).__init__(encoder=BiJesterNetII,
                                              encoder_args=encoder_args,
                                              pretrained_path=pretrained_path,
                                              freeze=freeze)

    def extract_features(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features



class ImagenetEncoder(ExternalEncoder):
    def __init__(self, pretrained_path=None, freeze=False):
        # self.encoder = vgg16_bn(pretrained=True)

        self.encoder_output_size = 1000

        num_classes = 1000
        pretrained = True
        encoder_args = (pretrained,)
        super().__init__(encoder=vgg11_bn,
                         encoder_args=encoder_args,
                         pretrained_path=pretrained_path,
                         freeze=freeze)

    def extract_features(self, video_batch):
        num_frames =  video_batch.size()[2] #[batch_size*num_ch*len*w*h]
        features = []
        for f in range(num_frames):
            frame_features = self.encoder(video_batch[:, :, f, :, :])
            features.append(frame_features)
        return torch.stack(features,dim=1) #len*num_features
