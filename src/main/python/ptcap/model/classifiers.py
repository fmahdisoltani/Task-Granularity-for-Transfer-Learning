import torch
from torch import nn
from ptcap.model.encoders import (CNN3dEncoder,
                                  CNN3dLSTMEncoder)


class C3dClassifier(nn.Module):
    def __init__(self, encoder=CNN3dEncoder, encoder_args=(), num_classes=7, gpus=None):
        super(C3dClassifier, self).__init__()
        self.use_cuda = True if gpus else False
        self.gpus = gpus

        #TODO: 128 is encoder num_feature, fix it to receive it
        self.linear = nn.Linear(128, num_classes)
        self.encoder = encoder(*encoder_args)
        self.logsoftmax = nn.LogSoftmax()


    def forward(self, video_batch, use_teacher_forcing=False):
        videos, captions = video_batch
        features = self.encoder(videos)
        probs = self.predict(features)
        return probs


    def predict(self, features):

        # The output should be num_classes dimensional now
        fc_output = self.linear(features)
        probs = self.logsoftmax(fc_output)

        _, preds = torch.max(probs, dim=1)
        return probs