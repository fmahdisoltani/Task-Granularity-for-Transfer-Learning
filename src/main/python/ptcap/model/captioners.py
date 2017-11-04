import torch
from torch import nn

from rtorchn.core.networks import RtorchnCaptioner as RtorchnCap
from ptcap.model.encoders import (CNN3dEncoder, CNN3dLSTMEncoder)
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
    def __init__(self, encoder, decoder, encoder_args=None, encoder_kwargs=None,
                 decoder_args=None, decoder_kwargs=None):

        super(EncoderDecoder, self).__init__()
        encoder_args = encoder_args or ()
        encoder_kwargs = encoder_kwargs or {}
        decoder_args = decoder_args or ()
        decoder_kwargs = decoder_kwargs or {}
        print("*"*100)
        print(encoder_args)
        print(encoder_kwargs)
        print(decoder_args)
        print(decoder_kwargs)
        print("&"*100)
        self.encoder = encoder(*encoder_args, **encoder_kwargs)
        self.decoder = decoder(*decoder_args, **decoder_kwargs)

        self.activations = {}
        self.register_forward_hook(self.merge_activations)

    def forward(self, video_batch, use_teacher_forcing):
        videos, captions = video_batch
        features = self.encoder(videos)
        probs = self.decoder(features, captions, use_teacher_forcing)

        return probs

    def merge_activations(self, module, input_tensor, output_tensor):
        self.activations = dict(self.encoder.activations,
                                **self.decoder.activations)


# class CNN3dLSTM(EncoderDecoder):
#     def __init__(self, encoder_output_size=256, embedding_size=31,
#                  vocab_size=33, num_hidden_lstm=71, go_token=0, gpus=None):
# 
#         decoder_args = (embedding_size, encoder_output_size,
#                         vocab_size, num_hidden_lstm, go_token, gpus)
# 
#         encoder_args = (encoder_output_size, gpus)
# 
#         super(CNN3dLSTM, self).__init__(CNN3dLSTMEncoder, LSTMDecoder,
#                                         encoder_args=encoder_args,
#                                         decoder_args=decoder_args,
#                                         gpus=gpus)

