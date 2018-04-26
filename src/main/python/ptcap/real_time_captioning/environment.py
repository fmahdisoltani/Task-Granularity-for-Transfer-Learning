import numpy as np
import torch
from torch.autograd import Variable
from collections import namedtuple


class Environment:
    # statuses
    STATUS_VALID_WRITE = 'write'
    STATUS_VALID_ACTION = 'valid'
    STATUS_INVALID_ACTION = 'invalid'
    STATUS_DONE = 'done'

    def __init__(self, encoder):

        self.encoder = encoder
        #self.reset()

    def reset(self, video=None, caption=None):
        self.read_count = 0
        self.write_count = 0

        self.vid_encoding = self.encoder.module.extract_features(video)
        # input buffer contains the seen video frames
        self.input_buffer = [self.vid_encoding[:, 0,:]]



    def get_state(self):
        return {
            "read_count": self.read_count,
            "write_count": self.write_count,
            "input_buffer": self.input_buffer
        }

    def update_state(self, action, action_seq=[]):
        status = ""

        if action.data.numpy()[0] == 0:  # READ
            status = Environment.STATUS_VALID_ACTION
            self.input_buffer.append(self.vid_encoding[:, self.read_count, :])
            self.read_count += 1

        if action.data.numpy()[0] == 1:  # WRITE
            if len(action_seq) > 2 and action_seq[-2]+action_seq[-3] == 0:
                status = Environment.STATUS_VALID_WRITE

            else:

                status = Environment.STATUS_INVALID_ACTION
            self.write_count += 1

        reward = self.give_reward(status)
        return reward

    def check_finished(self):
        return self.write_count+self.read_count == 10

    def give_reward(self, status):
        return {
            Environment.STATUS_VALID_WRITE: 10,
            Environment.STATUS_VALID_ACTION: 1,
            Environment.STATUS_INVALID_ACTION: -1,
        }[status]