import torch
from torch import nn

from rtorchn.models.captioning.vid2caption import DeepNet
from ptcap.model.encoders import CNN3dEncoder
from ptcap.model.decoders import LSTMDecoder
class Captioner(nn.Module):

    def forward(self, video_batch):
        """BxCxTxWxH -> BxTxV"""
        raise NotImplementedError


class RtorchnCaptioner(Captioner):
    """
    This class is a wrapper for rtorchn captioning model.
    Typical kwargs: is_training=True, use_cuda=False, gpus=[0]
    """

    def __init__(self, vocab_size, go_token=0, batchnorm=True, stateful=False,
                 **kwargs):
        super(RtorchnCaptioner, self).__init__()
        self.captioner = DeepNet(vocab_size, batchnorm, stateful, **kwargs)

    def forward(self, video_batch):
        return self.captioner.forward(video_batch)


class EncoderDecoder(Captioner):
    def __init__(self, encoder=CNN3dEncoder,
                 decoder=LSTMDecoder, encoder_output_size=128):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder(encoder_output_size)
        self.decoder = decoder(embedding_size=97,
                               hidden_size=encoder_output_size,
                               vocab_size=34, num_hidden_lstm=71)

    def forward(self, video_batch):
        videos, captions = video_batch
        features = self.encoder(videos)
        h_list = self.decoder(features, captions)

        return h_list

