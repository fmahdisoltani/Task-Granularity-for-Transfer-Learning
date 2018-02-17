import torch
import torch.nn as nn

from ptcap.model.encoders import Encoder
from ptcap.model.feature_extractors import C2dExtractor, C3dExtractor


class TwoStreamEncoder(Encoder):
    def __init__(self, encoder_output_size=52, c3d_out_ch=0,
                 c2d_out_ch=0, rnn_output_size=51, bidirectional=True):
        super().__init__()
        # c3d_out_ch = 36
        # c2d_out_ch = 0
        self.encoder_output_size = encoder_output_size

        self.use_c3d = c3d_out_ch > 0
        self.c3d_extractor = C3dExtractor(out_ch=c3d_out_ch)

        self.use_c2d = c2d_out_ch > 0
        self.c2d_extractor = C2dExtractor(c2d_out_ch)

        lstm_hidden_size = int(rnn_output_size/2 if
                               bidirectional else rnn_output_size)
        self.lstm = nn.LSTM(input_size=8*c3d_out_ch + 8*c2d_out_ch,
                            hidden_size=lstm_hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.fc = (nn.Linear(rnn_output_size, self.encoder_output_size))
        self.dropout = nn.Dropout(p=0.5)

        self.activations = {}  # TODO:FIX Tensorboard

    def extract_features(self, videos):
        # Video encoding
        cnn_features = []
        if self.use_c2d:
            c2d_features = self.c2d_extractor.extract_features(videos)
            cnn_features.append(c2d_features)
        if self.use_c3d:
            c3d_features = self.c3d_extractor.extract_features(videos)
            cnn_features.append(c3d_features)
        h = torch.cat(cnn_features, 2)

        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(h)  # lstm_outputs:[8*48*512]

        return self.dropout(self.relu(self.fc(lstm_outputs)))