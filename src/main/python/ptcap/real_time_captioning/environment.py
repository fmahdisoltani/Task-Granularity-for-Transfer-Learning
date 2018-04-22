import numpy as np
import torch
from torch.autograd import Variable
from collections import namedtuple

class Environment:
    # statuses
    STATUS_VALID_ACTION = 'valid'
    STATUS_INVALID_ACTION = 'invalid'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        self.read_count = 0
        self.write_count = 0

    def get_state(self):
        return {
            "read_count": self.read_count,
            "write_count": self.write_count
        }

    def update_state(self, action):
        status = ""
        if action.data.numpy()[0] == 0:  # READ

            status = Environment.STATUS_VALID_ACTION
            self.read_count += 1

        if action.data.numpy()[0] == 1:  # WRITE
            status = Environment.STATUS_INVALID_ACTION
            self.write_count += 1

        reward = self.give_reward(status)
        return reward

    def check_finished(self):
        return self.write_count+self.read_count ==4

    def give_reward(self, status):
        return {
            Environment.STATUS_VALID_ACTION: 1,
            Environment.STATUS_INVALID_ACTION: -1,
        }[status]