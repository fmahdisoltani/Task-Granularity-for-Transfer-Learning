from torch import nn


class Decoder(nn.Module):

    def forward(self, decoder_states, teacher_targets=None):
        """(BxD, BxKxV) -> BxKxV"""
        raise NotImplementedError
