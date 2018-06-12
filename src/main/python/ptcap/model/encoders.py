import torch.nn as nn

from ptcap.model.feature_extractors import C3dExtractor, CausalC3dExtractor
from ptcap.tensorboardY import forward_hook_closure


class Encoder(nn.Module):
    def forward(self, video_batch):
        """BxCxTxWxH -> BxM"""
        raise NotImplementedError


class LSTMEncoder(Encoder):
    def __init__(self, encoder_output_size=52, out_ch=32, bidirectional=True,
                 rnn_output_size=51, num_lstm_layers=1, causal=True):
        """
        encoder_output_size: defines the output size of the encoder
        """

        super().__init__()

        self.encoder_output_size = encoder_output_size

        self.input_layer = nn.Linear(96*96*3, 256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(rnn_output_size, self.encoder_output_size)
        self.dropout = nn.Dropout(p=0.5)

        bidirectional = bidirectional and not causal
        lstm_hidden_size = int(
            rnn_output_size / 2) if bidirectional else rnn_output_size

        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, batch_first=True,
                            bidirectional=bidirectional)

        self.activations = {}

    def extract_features(self, videos):
        # videos: [batch_size*num_ch*len*w*h]
        lin_features = self.input_layer(videos.view(8, 48, 3*96*96))

        # self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(lin_features)
        return self.dropout(self.relu(
            self.fc(lstm_outputs)))  # [batch_size*num_step*num_features]


class C3dLSTMEncoder(Encoder):
    def __init__(self, encoder_output_size=52, out_ch=32, bidirectional=True,
                 rnn_output_size=51, num_lstm_layers=1, causal=False):
        """
        encoder_output_size: defines the output size of the encoder
        """

        super().__init__()

        self.encoder_output_size = encoder_output_size

        self.c3d_extractor = CausalC3dExtractor() if causal else C3dExtractor()

        self.relu = nn.ReLU()
        self.fc = (nn.Linear(rnn_output_size, self.encoder_output_size))
        self.dropout = nn.Dropout(p=0.5)
        if causal and bidirectional:
            print("Can not use bidirectional LSTM in causal mode \n "
                  "I'm changing it to unidirectional")
        bidirectional = bidirectional and not causal
        lstm_hidden_size = int(
            rnn_output_size / 2) if bidirectional else rnn_output_size

        self.lstm = nn.LSTM(input_size=8*out_ch, hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers, batch_first=True,
                            bidirectional=bidirectional)
        
        self.activations = self.register_forward_hooks()
        #self.activations = {}  # TODO:FIX Tensorboard

    def extract_features(self, videos):
        # videos: [batch_size*num_ch*len*w*h]
        c3d_features = self.c3d_extractor.extract_features(videos)

        #self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(c3d_features)
        return self.dropout(self.relu(self.fc(lstm_outputs)))  # [batch_size*num_step*num_features]

    def register_forward_hooks(self):
        master_dict = {}
        self.c3d_extractor.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_c3d_extractor"))
        self.lstm.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_lstm", 0, True))
        return master_dict


