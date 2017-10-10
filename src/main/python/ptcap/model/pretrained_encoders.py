import torch
from .encoders import Encoder
from rtorchn.core.networks import (FullyConvolutionalNet, JesterNet)
from rtorchn.core.networks.resnets import InflatedResNet18
from .encoders import Encoder


class PretrainedEncoder(Encoder):
    def __init__(self, encoder, encoder_args, encoder_kwargs,
                 pretrained_path=None, checkpoint_key=None, freeze=False):

        encoder_args = encoder_args or ()
        encoder_kwargs = encoder_kwargs or {}
        super(PretrainedEncoder, self).__init__()
        self.encoder = encoder(*encoder_args, **encoder_kwargs)

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


class RtorchnEncoderP(PretrainedEncoder):

    def __init__(self, pretrained_path=None,
                 freeze=False, encoder_output_size=256, num_classes=178):

        # it thinks it's getting num_features, but it's not. what is happening
        # is equivalent to FullyConvolutionalNet(..,
        #                                     num_features=encoder_output_size)

        encoder_args = (num_classes, encoder_output_size)
        super(RtorchnEncoderP, self).__init__(encoder=FullyConvolutionalNet,
                                              encoder_args=encoder_args,
                                              encoder_kwargs={},
                                              pretrained_path=pretrained_path,
                                              freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)


class JesterEncoderP(PretrainedEncoder):

    def __init__(self, pretrained_path=None,
                 freeze=False, encoder_output_size=256, num_classes=329):

        # it thinks it's getting num_features, but it's not. what is happening
        # is equivalent to FullyConvolutionalNet(..,
        #                                     num_features=encoder_output_size)

        encoder_args = (num_classes, encoder_output_size)
        use_cuda = True
        gpus = [0]
        encoder_kwargs = {"use_cuda":use_cuda, "gpus":gpus}
        super(JesterEncoderP, self).__init__(encoder=JesterNet,
                                             encoder_args=encoder_args,
                                             encoder_kwargs=encoder_kwargs,
                                             pretrained_path=pretrained_path,
                                              freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)


class Resnet18EncoderP(PretrainedEncoder):

    def __init__(self, pretrained_path=None, freeze=False, ):
        num_classes = 328
        encoder_args = (num_classes,)
        super(Resnet18Encoder, self).__init__(encoder=InflatedResNet18,
                                               encoder_args=encoder_args,
                                               pretrained_path=pretrained_path,
                                               freeze=freeze)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)
