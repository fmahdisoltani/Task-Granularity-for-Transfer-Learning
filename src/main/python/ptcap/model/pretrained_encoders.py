import torch
from .encoders import Encoder
from rtorchn.core.networks import FullyConvolutionalNet
from .encoders import Encoder


class PretrainedEncoder(Encoder):
    def __init__(self, encoder, pretrained_path=None, checkpoint_key=None,
                 freeze=False, encoder_args=()):
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


class RtorchnEncoderP(PretrainedEncoder):

    def __init__(self, pretrained_path=None,
                 freeze=False, num_features=256, num_classes=178):



        pretrained_path = ('/home/farzaneh/PycharmProjects/'
                                'pretrained_nets/'
                                'fully_conv_net_on_smtsmt_20170627/model.checkpoint')
        encoder_args = (num_classes, num_features)
        super(RtorchnEncoderP, self).__init__(encoder=FullyConvolutionalNet,
                                              encoder_args=encoder_args,
                                              pretrained_path=pretrained_path,
                                              freeze=freeze)

        #self.encoder = FullyConvolutionalNet(num_classes=num_classes,
        #                                     num_features=num_features)


        # checkpoint = torch.load('/home/farzaneh/PycharmProjects/'
        #                         'pretrained_nets/'
        #                         'fully_conv_net_on_smtsmt_20170627/model.checkpoint')

        # self.encoder.load_state_dict(checkpoint)

    def forward(self, video_batch):
        features = self.encoder.extract_features(video_batch)
        return features.mean(dim=1)

