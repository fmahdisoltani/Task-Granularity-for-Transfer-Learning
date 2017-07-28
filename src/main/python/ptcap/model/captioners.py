from torch import nn

from rtorchn.models.captioning.vid2caption import DeepNet

class Captioner(nn.Module):

    def forward(self, video_batch):
        """BxCxTxWxH -> BxTxV"""
        raise NotImplementedError


class RtorchnCaptioner(Captioner):
    # **kwargs: is_training=True, use_cuda=False, gpus=[0]
    def __init__(self, vocab_size, batchnorm=True, stateful=False, **kwargs):
        super(RtorchnCaptioner, self).__init__()
        self.captioner = DeepNet(vocab_size, batchnorm, stateful,**kwargs)

    def forward(self, video_batch):
        return self.captioner.forward(video_batch)