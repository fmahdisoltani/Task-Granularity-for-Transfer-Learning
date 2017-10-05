import torch

from .encoders import Encoder

class PretrainedEncoder(Encoder):
    def __init__(self, encoder, pretrained_path=None, checkpoint_key=None):
        super(PretrainedEncoder, self).__init__()
        self.encoder = encoder
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            if checkpoint_key is not None:
                self.encoder.load_state_dict(checkpoint[checkpoint_key])
            else:
                self.encoder.load_state_dict(checkpoint)

    def forward(self, video_batch):
        return self.encoder(video_batch)
