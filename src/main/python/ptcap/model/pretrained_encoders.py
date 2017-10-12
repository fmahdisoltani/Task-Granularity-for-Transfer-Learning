import torch
from .encoders import Encoder
from rtorchn.core.networks import (FullyConvolutionalNet, JesterNet)
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

    def __init__(self, pretrained_path=None,
                 freeze=False):

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

    # Hardcoded encoder for using JesterNet

    def __init__(self, pretrained_path=None,
                 freeze=False, encoder_output_size=256, num_classes=329):

        encoder_args = (num_classes, encoder_output_size)
        super(FCEncoder, self).__init__(encoder=JesterNet,
                                              encoder_args=encoder_args,
                                              pretrained_path=pretrained_path,
                                              freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)

