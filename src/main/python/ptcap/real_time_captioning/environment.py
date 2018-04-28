import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple


class Environment:
    # statuses
    STATUS_CORRECT_WRITE = 'correct_write'
    STATUS_INCORRECT_WRITE = 'incorrect_write'
    STATUS_READ = 'read'
    STATUS_DONE = 'done'

    def __init__(self, encoder):

        self.encoder = encoder

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        #self.reset()

    def reset(self, video=None, caption=None, classif=None):
        self.read_count = 0
        self.write_count = 0

        self.vid_encoding = self.encoder.extract_features(video)
        # input buffer contains the seen video frames
        self.input_buffer = [self.vid_encoding[:, 0,:]]


    def get_state(self):
        return {
            "read_count": self.read_count,
            "write_count": self.write_count,
            "input_buffer": self.input_buffer
        }

    def update_state(self, action, action_seq=[], classif_targets=None):
        status = ""
        classif_probs = self.classify()
        #if action.data.numpy()[0] == 0:  # READ
        #    status = Environment.STATUS_READ
        #    self.input_buffer.append(self.vid_encoding[:, self.read_count, :])
        #    self.read_count += 1
        #    reward = self.give_reward(status)

        #if action.data.numpy()[0] == 1:  # WRITE

        #    _, prediction = torch.max(classif_probs, dim=1)

        #   if prediction.data.cpu().numpy()[0] == classif_targets.data.cpu().numpy()[0]:
        #       status = Environment.STATUS_CORRECT_WRITE
        #  else:
        #      status = Environment.STATUS_INCORRECT_WRITE
        #  self.write_count += 1

        reward = torch.sum(classif_probs)
        return reward

    def check_finished(self):
        return self.write_count == 1

    def give_reward(self, status):
        return {
            Environment.STATUS_CORRECT_WRITE: 100,
            Environment.STATUS_INCORRECT_WRITE: -1000,
            Environment.STATUS_READ: -1,
        }[status]

    def classify(self):
        features = self.input_buffer[-1]
        pre_activation = self.classif_layer(features)
        probs = self.logsoftmax(pre_activation)
        if probs.ndimension() == 3:
            probs = probs.mean(dim=1)  # probs: [8*48*178]

        return probs
