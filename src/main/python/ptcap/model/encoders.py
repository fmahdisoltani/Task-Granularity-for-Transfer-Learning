import torch
import torch.nn as nn

from torch.autograd import Variable

from ptcap.model.layers import CNN3dLayer
from ptcap.tensorboardY import forward_hook_closure


class Encoder(nn.Module):
    def forward(self, video_batch):
        """BxCxTxWxH -> BxM"""
        raise NotImplementedError


class FullyConnectedEncoder(Encoder):
    def __init__(self, video_dims, num_features):
        super(FullyConnectedEncoder, self).__init__()
        C, T, W, H = video_dims
        self.linear = nn.Linear(C * T * W * H, num_features)

    def forward(self, video_batch):
        batch_size = video_batch.size()[0]
        return self.linear(video_batch.view(batch_size, -1))


class CNN3dEncoder(Encoder):
    def __init__(self, num_features=128, gpus=None):
        super(CNN3dEncoder, self).__init__()

        self.conv1 = CNN3dLayer(3, 16, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv2 = CNN3dLayer(16, 32, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv3 = CNN3dLayer(32, 64, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)

        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CNN3dLayer(64, 128, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))
        self.conv5 = CNN3dLayer(128, 128, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))
        self.conv6 = CNN3dLayer(128, num_features, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        self.activations = self.register_forward_hooks()

    def forward(self, videos):
        # Video encoding
        h = self.conv1(videos)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.pool2(h)

        h = self.conv3(h)
        h = self.pool3(h)

        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)

        h = self.pool4(h)  # batch_size * num_features * num_step * w * h

        h = h.mean(2)
        h = h.view(h.size()[0:2])

        return h

    def register_forward_hooks(self):
        master_dict = {}
        self.conv1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv1"))
        self.conv2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv2"))
        self.conv3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv3"))
        self.conv4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv4"))
        self.conv5.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv5"))
        self.conv6.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv6"))
        self.pool1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool1"))
        self.pool2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool2"))
        self.pool3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool3"))
        self.pool4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool4"))
        return master_dict


class C3dLSTMEncoder(Encoder):
    def __init__(self, encoder_output_size=52, gpus=None, out_ch=32,
                 bidirectional=True):
        """
        num_features: defines the output size of the encoder
        """

        super().__init__()

        self.num_layers = 1
        self.encoder_output_size = encoder_output_size
        self.use_cuda = True if gpus else False
        self.gpus = gpus
        
        # self.logsoftmax = nn.LogSoftmax()
        # self.classif_layer = nn.Linear(1024, self.num_classes)
        self.relu = nn.ReLU()
        self.fc = (nn.Linear(self.encoder_output_size, 2* encoder_output_size))

        #self.num_classes = 178
        #self.classif_layer = nn.Linear(1024, self.num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = CNN3dLayer(3, out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=1)
        self.conv2 = CNN3dLayer(out_ch, 2*out_ch, (3, 3, 3), nn.ReLU(), 
                                stride=1, padding=1)
        self.conv3 = CNN3dLayer(2*out_ch, 4*out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=1)

        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.conv4 = CNN3dLayer(4*out_ch, 8*out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=(1, 0, 0))
        self.conv5 = CNN3dLayer(8*out_ch, 8*out_ch, (3, 3, 3), nn.ReLU(),
                                stride=1, padding=(1, 0, 0))
        self.conv6 = CNN3dLayer(8*out_ch, 8*out_ch, (3, 3, 3),
                                nn.ReLU(), stride=1, padding=(1, 0, 0))

        self.pool4 = nn.MaxPool3d((1, 6, 6))

        lstm_hidden_size = int(
            self.encoder_output_size / 2) if bidirectional else self.encoder_output_size


        self.lstm = nn.LSTM(input_size=8*out_ch, hidden_size=lstm_hidden_size,
                            num_layers=self.num_layers, batch_first=True, 
                            bidirectional=True)
        
        
        
        self.activations = self.register_forward_hooks()

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.encoder_output_size))
        c0 = Variable(torch.zeros(1, batch_size, self.encoder_output_size))
        if self.use_cuda:
            h0 = h0.cuda(self.gpus[0])
            c0 = c0.cuda(self.gpus[0])
        return (h0, c0)

    def extract_features(self, videos):
        # videos: [batch_size*num_ch*len*w*h] (8*3*48*96*96)

        h = self.conv1(videos)  # [batch_size*num_ch*len*w*h] = 8*32*48*96*96
        h = self.pool1(h)  # 8*32*48*48 *48

        h = self.conv2(h)  # 8*64 *48*48*48
        h = self.pool2(h)  # 8*64 *48*24*24

        h = self.conv3(h)  # 8*128 *48*24*24
        h = self.pool3(h)  # 8*128 *48*12*12

        h = self.conv4(h)  # 8*256 *48*10*10
        h = self.conv5(h)  # 8*256 *48*8*8
        h = self.conv6(h)  # 8*256 *48*6*6
        h = self.pool4(h)  # 8*256 *48*1*1

        h = h.view(h.size()[0:3])  # [batch_size*num_feature*num_step](8*256*48)
        h = h.permute(0, 2, 1)  # [batch_size*num_step*num_features]

        lstm_hidden = self.init_hidden(batch_size=h.size()[0])

        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(h) #lstm_input:8*48*256 lstm_output:8*48*1024

        # h_mean = torch.mean(lstm_outputs, dim=1)

        return self.dropout(self.relu(self.fc(lstm_outputs)))  # 8* 48 * 1024

    # def predict_from_features(self, features):
    #     probs = self.logsoftmax(features)
    #     if probs.ndimension() == 3:
    #         probs = probs.mean(dim=1)
    #     return probs

    def register_forward_hooks(self):
        master_dict = {}
        self.conv1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv1"))
        self.conv2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv2"))
        self.conv3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv3"))
        self.conv4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv4"))
        self.conv5.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv5"))
        self.conv6.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_conv6"))
        self.pool1.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool1"))
        self.pool2.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool2"))
        self.pool3.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool3"))
        self.pool4.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_pool4"))
        self.lstm.register_forward_hook(
            forward_hook_closure(master_dict, "encoder_lstm", 0, True))
        return master_dict

    class CNN3dEncoder(Encoder):
        def __init__(self, num_features=128, gpus=None):
            super(CNN3dEncoder, self).__init__()

            self.conv1 = CNN3dLayer(3, 16, (3, 3, 3), nn.ReLU(),
                                    stride=1, padding=1)
            self.conv2 = CNN3dLayer(16, 32, (3, 3, 3), nn.ReLU(),
                                    stride=1, padding=1)
            self.conv3 = CNN3dLayer(32, 64, (3, 3, 3), nn.ReLU(),
                                    stride=1, padding=1)

            self.pool1 = nn.MaxPool3d((1, 2, 2))

            self.pool2 = nn.MaxPool3d((1, 2, 2))

            self.pool3 = nn.MaxPool3d((1, 2, 2))

            self.conv4 = CNN3dLayer(64, 128, (3, 3, 3), nn.ReLU(),
                                    stride=1, padding=(1, 0, 0))
            self.conv5 = CNN3dLayer(128, 128, (3, 3, 3), nn.ReLU(),
                                    stride=1, padding=(1, 0, 0))
            self.conv6 = CNN3dLayer(128, num_features, (3, 3, 3), nn.ReLU(),
                                    stride=1, padding=(1, 0, 0))

            self.pool4 = nn.MaxPool3d((1, 6, 6))

            self.activations = self.register_forward_hooks()

        def forward(self, videos):
            # Video encoding
            h = self.conv1(videos)
            h = self.pool1(h)

            h = self.conv2(h)
            h = self.pool2(h)

            h = self.conv3(h)
            h = self.pool3(h)

            h = self.conv4(h)
            h = self.conv5(h)
            h = self.conv6(h)

            h = self.pool4(h)  # batch_size * num_features * num_step * w * h

            h = h.mean(2)
            h = h.view(h.size()[0:2])

            return h

        def register_forward_hooks(self):
            master_dict = {}
            self.conv1.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv1"))
            self.conv2.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv2"))
            self.conv3.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv3"))
            self.conv4.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv4"))
            self.conv5.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv5"))
            self.conv6.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv6"))
            self.pool1.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool1"))
            self.pool2.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool2"))
            self.pool3.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool3"))
            self.pool4.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool4"))
            return master_dict

    class C2dEncoder(Encoder):
        def __init__(self, encoder_output_size=52, gpus=None, out_ch=32,
                     bidirectional=True):
            """
            num_features: defines the output size of the encoder
            """

            super().__init__()

            self.num_layers = 1
            self.encoder_output_size = encoder_output_size
            self.use_cuda = True if gpus else False
            self.gpus = gpus

            # self.logsoftmax = nn.LogSoftmax()
            # self.classif_layer = nn.Linear(1024, self.num_classes)
            self.relu = nn.ReLU()
            self.fc = (
            nn.Linear(self.encoder_output_size, 2 * encoder_output_size))

            # self.num_classes = 178
            # self.classif_layer = nn.Linear(1024, self.num_classes)

            self.dropout = nn.Dropout(p=0.5)

            self.conv1 = CNN3dLayer(3, out_ch, (1, 3, 3), nn.ReLU(),
                                    stride=1, padding=1)
            self.conv2 = CNN3dLayer(out_ch, 2 * out_ch, (1, 3, 3), nn.ReLU(),
                                    stride=1, padding=1)
            self.conv3 = CNN3dLayer(2 * out_ch, 4 * out_ch, (1, 3, 3),
                                    nn.ReLU(), stride=1, padding=1)

            self.pool1 = nn.MaxPool3d((1, 2, 2))

            self.pool2 = nn.MaxPool3d((1, 2, 2))

            self.pool3 = nn.MaxPool3d((1, 2, 2))

            self.conv4 = CNN3dLayer(4 * out_ch, 8 * out_ch, (1, 3, 3),
                                    nn.ReLU(), stride=1, padding=(1, 0, 0))
            self.conv5 = CNN3dLayer(8 * out_ch, 8 * out_ch, (1, 3, 3),
                                    nn.ReLU(),
                                    stride=1, padding=(1, 0, 0))
            self.conv6 = CNN3dLayer(8 * out_ch, 8 * out_ch, (1, 3, 3),
                                    nn.ReLU(), stride=1, padding=(1, 0, 0))

            self.pool4 = nn.MaxPool3d((1, 6, 6))

            lstm_hidden_size = int(
                self.encoder_output_size / 2) if bidirectional else self.encoder_output_size

            self.lstm = nn.LSTM(input_size=8 * out_ch,
                                hidden_size=lstm_hidden_size,
                                num_layers=self.num_layers, batch_first=True,
                                bidirectional=True)

            self.activations = self.register_forward_hooks()

        def init_hidden(self, batch_size):
            h0 = Variable(torch.zeros(1, batch_size, self.encoder_output_size))
            c0 = Variable(torch.zeros(1, batch_size, self.encoder_output_size))
            if self.use_cuda:
                h0 = h0.cuda(self.gpus[0])
                c0 = c0.cuda(self.gpus[0])
            return (h0, c0)

        def extract_features(self, videos):
            # videos: [batch_size*num_ch*len*w*h] (8*3*48*96*96)

            h = self.conv1(
                videos)  # [batch_size*num_ch*len*w*h] = 8*32*48*96*96
            h = self.pool1(h)  # 8*32*48*48 *48

            h = self.conv2(h)  # 8*64 *48*48*48
            h = self.pool2(h)  # 8*64 *48*24*24

            h = self.conv3(h)  # 8*128 *48*24*24
            h = self.pool3(h)  # 8*128 *48*12*12

            h = self.conv4(h)  # 8*256 *48*10*10
            h = self.conv5(h)  # 8*256 *48*8*8
            h = self.conv6(h)  # 8*256 *48*6*6
            h = self.pool4(h)  # 8*256 *48*1*1

            h = h.view(
                h.size()[0:3])  # [batch_size*num_feature*num_step](8*256*48)
            h = h.permute(0, 2, 1)  # [batch_size*num_step*num_features]

            lstm_hidden = self.init_hidden(batch_size=h.size()[0])

            self.lstm.flatten_parameters()
            lstm_outputs, _ = self.lstm(
                h)  # lstm_input:8*48*256 lstm_output:8*48*1024

            # h_mean = torch.mean(lstm_outputs, dim=1)

            return self.dropout(
                self.relu(self.fc(lstm_outputs)))  # 8* 48 * 1024

        # def predict_from_features(self, features):
        #     probs = self.logsoftmax(features)
        #     if probs.ndimension() == 3:
        #         probs = probs.mean(dim=1)
        #     return probs

        def register_forward_hooks(self):
            master_dict = {}
            self.conv1.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv1"))
            self.conv2.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv2"))
            self.conv3.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv3"))
            self.conv4.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv4"))
            self.conv5.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv5"))
            self.conv6.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_conv6"))
            self.pool1.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool1"))
            self.pool2.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool2"))
            self.pool3.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool3"))
            self.pool4.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_pool4"))
            self.lstm.register_forward_hook(
                forward_hook_closure(master_dict, "encoder_lstm", 0, True))
            return master_dict

