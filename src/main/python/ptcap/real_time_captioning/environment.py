import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple


class Environment(nn.Module):
    # statuses
    STATUS_CORRECT_WRITE = 'correct_write'
    STATUS_INCORRECT_WRITE = 'incorrect_write'
    STATUS_INVALID_READ = 'invalid_read'
    STATUS_READ = 'read'
    STATUS_DONE = 'done'

    def __init__(self, encoder, classif_layer):
        super().__init__()

        self.encoder = encoder

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.classif_layer = classif_layer

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        #self.reset()

    def reset(self, video=None, caption=None, classif=None):
        self.read_count = 0
        self.write_count = 0

        self.vid_encoding = torch.zeros([1,1024]).cuda()#self.encoder.extract_features(video)
        # input buffer contains the seen video frames, in the beginning only the
        # first frame of video

        self.input_buffer = self.vid_encoding#[:, 0, :]

    def get_state(self):
        return {
            "read_count": self.read_count,
            "write_count": self.write_count,
            "input_buffer": self.input_buffer
        }

    def update_state(self, action, action_seq=[], classif_targets=None):
        status = ""
        classif_probs = self.classify()
        value_prob = None

        if action.data.cpu().numpy()[0] == 0:  # READ
            if self.read_count == 47:
                status = Environment.STATUS_INVALID_READ
            else:
                status = Environment.STATUS_READ
                #self.input_buffer = self.vid_encoding#[:, 0, :]
                self.read_count += 1

        if action.data.cpu().numpy()[0] == 1:  # WRITE
            value_prob, prediction = torch.max(classif_probs, dim=1)

            if prediction.data.cpu().numpy()[0] == classif_targets.data.cpu().numpy()[0]:
                status = Environment.STATUS_CORRECT_WRITE
            else:
                status = Environment.STATUS_INCORRECT_WRITE
            self.write_count += 1

        reward = self.give_reward(status, value_prob)
    #    reward = torch.sum(classif_probs)
        return reward, classif_probs

    def check_finished(self):
        return self.write_count == 1 or self.read_count == 48

    def give_reward(self, status, value_prob=None):
        r =  {
            Environment.STATUS_CORRECT_WRITE: 100,
            Environment.STATUS_INCORRECT_WRITE: -1000,
            Environment.STATUS_READ: -1,
            Environment.STATUS_INVALID_READ: -10,
        }[status]
        #r = r if value_prob is None else (-1/(1+value_prob)) * r
        return r

    def classify(self):
        #features = self.input_buffer[-1]
        pre_activation = self.classif_layer(self.vid_encoding)#[:, self.read_count, :])
        probs = self.logsoftmax(pre_activation)
        if probs.ndimension() == 3:
            probs = probs.mean(dim=1)  # probs: [8*48*178]

        return probs
