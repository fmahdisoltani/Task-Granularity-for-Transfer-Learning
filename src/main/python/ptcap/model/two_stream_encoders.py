import torch
import torch.nn as nn

from ptcap.model.encoders import Encoder
from ptcap.model.feature_extractors import C2dFeatureExtractor, C3dFeatureExtractor


class TwoStreamEncoder(Encoder):
    def __init__(self, encoder_output_size=52, c3d_output_size=0,
                 c2d_output_size=0, rnn_output_size=512, bidirectional=True):
        super().__init__()
        self.encoder_output_size = encoder_output_size
        self.c3d_output_size = c3d_output_size
        self.c2d_output_size = c2d_output_size
        self.c3d_feature_extractor = C3dFeatureExtractor()
        self.c2d_feature_extractor = C2dFeatureExtractor()

        lstm_hidden_size = int(rnn_output_size/2 if
                               bidirectional else rnn_output_size)
        self.lstm = nn.LSTM(input_size=c3d_output_size+c2d_output_size,
                            hidden_size=lstm_hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.fc = (nn.Linear(rnn_output_size, self.encoder_output_size))
        self.dropout = nn.Dropout(p=0.5)

        self.activations = {}  # TODO:FIX Tensorboard

    def extract_features(self, videos):
        # Video encoding
        c3d_features = self.c3d_feature_extractor.extract_features(videos)  # 8*48*128
        c2d_features = self.c2d_feature_extractor.extract_features(videos)  # 8*48*256

        if self.c3d_output_size and self.c2d_output_size:
            h = torch.cat((c3d_features, c2d_features), 2)  # 8*48*384
        elif self.c3d_output_size:
            h = c3d_features
        elif self.c2d_output_size:
            h = c2d_features

        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(h)  # lstm_outputs:[8*48*512]

        return self.dropout(self.relu(self.fc(lstm_outputs)))