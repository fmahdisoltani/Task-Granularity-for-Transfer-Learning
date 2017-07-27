from torch import nn

from rtorchn.models.captioning import vid2caption

class Captioner(nn.Module):

    def forward(self, video_batch):
        """BxCxTxWxH -> BxTxV"""
        raise NotImplementedError


class RtorchnCaptioner(Captioner):

    def __init__(self):
        super(RtorchnCaptioner, self).__init__()
        self.captioner = vid2caption()

    def forward(self, video_batch):
        self.captioner.forward(video_batch)