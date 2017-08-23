import torch
from torch import nn

#from rtorchn.core.networks import RtorchnCaptioner as RtorchnCap
from ptcap.model.encoders import (CNN3dEncoder,
                                  CNN3dLSTMEncoder)
from ptcap.model.decoders import LSTMDecoder


class Captioner(nn.Module):

    def forward(self, video_batch, use_teacher_forcing):
        """BxCxTxWxH -> BxTxV"""
        raise NotImplementedError


class RtorchnCaptioner(Captioner):
    """
    This class is a wrapper for rtorchn captioning model.
    Typical kwargs: is_training=True, use_cuda=False, gpus=[0]
    """

    def __init__(self, vocab_size):
        super(RtorchnCaptioner, self).__init__()
        self.captioner = RtorchnCap(vocab_size)

    def forward(self, video_batch, use_teacher_forcing=True):
        return self.captioner.forward(video_batch)


class EncoderDecoder(Captioner):
    def __init__(self, encoder, decoder, encoder_args=(), decoder_args=(),
                 use_cuda=False, gpus=[0]):
        super(EncoderDecoder, self).__init__()
        self.use_cuda = use_cuda
        self.gpus = gpus

        self.encoder = encoder(*encoder_args)
        self.decoder = decoder(*decoder_args)

    def forward(self, video_batch, use_teacher_forcing):
        videos, captions = video_batch
        features = self.encoder(videos)
        probs = self.decoder(features, captions)

        return probs


class CNN3dLSTM(EncoderDecoder):
    def __init__(self, encoder_output_size=128, embedding_size=31,
                 vocab_size=33, num_hidden_lstm=71, go_token=0, use_cuda=False,
                 gpus =[0]):

        decoder_args = (embedding_size, encoder_output_size,
                        vocab_size, num_hidden_lstm, go_token, use_cuda, gpus)

        encoder_args = (encoder_output_size, use_cuda, gpus)

        super(CNN3dLSTM, self).__init__(CNN3dLSTMEncoder, LSTMDecoder,
                                        encoder_args=encoder_args,
                                        decoder_args=decoder_args,
                                        use_cuda=use_cuda, gpus=gpus)

