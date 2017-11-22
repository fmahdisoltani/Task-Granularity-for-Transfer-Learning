import torch
from .encoders import Encoder
from rtorchn.core.networks import (FullyConvolutionalNet, JesterNet, 
                                   BiJesterNetII, InflatedResNet18)
from .encoders import Encoder

class PretrainedEncoder(Encoder):
    def __init__(self, encoder, encoder_args, pretrained_path=None,
                 checkpoint_key=None, freeze=False):

        super(PretrainedEncoder, self).__init__()
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


class FCEncoder(PretrainedEncoder):
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


class JesterEncoder(PretrainedEncoder):
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


class BIJesterEncoder(PretrainedEncoder):

    """
    Hardcoded encoder for using BIJesterNet from 20bn_rtorchn
    num_classes = 174 means this class expects the "supermodel" version.
    Encoder output size in this version is 1024
    """

    def __init__(self, pretrained_path=None, freeze=False):
        encoder_output_size = 1024
        num_classes = 178
        encoder_args = (num_classes, encoder_output_size)
        super(BIJesterEncoder, self).__init__(encoder=BiJesterNetII,
                                              encoder_args=encoder_args,
                                              pretrained_path=pretrained_path,
                                              freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features
        # return features.mean(dim=1)
      
      
class Resnet18Encoder(PretrainedEncoder):
    def __init__(self, pretrained_path=None, freeze=False, ):
        num_classes = 1363
        encoder_args = (num_classes,)
        super(Resnet18Encoder, self).__init__(encoder=InflatedResNet18,
                                              encoder_args=encoder_args,
                                              pretrained_path=pretrained_path,
                                              freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features
        # return features.mean(dim=1)
