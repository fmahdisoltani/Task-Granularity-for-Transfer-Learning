import numpy as np
import torch
from torch.autograd import Variable
from collections import namedtuple

class Environment:
    # statuses
    STATUS_VALID_ACTION = 'valid'
    STATUS_INVALID_ACTION = 'invalid'
    STATUS_DONE = 'done'
    STATUS_CORRECT_WORD = 'correct'
    STATUS_INCORRECT_WORD = 'incorrect'

    MAX_TOKEN_COUNT = 13
    MAX_FRAME_COUNT = 48

    def __init__(self, encoder, decoder):

        self.encoder = encoder
        self.decoder = decoder


    def reset(self, video, caption):
        self.caption = caption
        self.vid_encoding = self.encoder.extract_features(video)
        self.output_buffer = [torch.FloatTensor(1, 1).zero_()] #initialize with <GO>
        self.input_buffer = [self.vid_encoding[:, 0, :]]  # input buffer contains the seen video frames
        self.read_count = 0
        self.write_count =0

    def get_state(self):
        #TODO: fix the state
        # input_buffer: (Batch_size x num_frames x feature_size): 1 x 48 x 1024
        return {
            "output_buffer": self.output_buffer,
            "input_buffer": self.input_buffer,
            "read_count": self.read_count,
            "write_count": self.write_count

        }

    def update_state(self, action):
        status = ""
        if action.data.numpy()[0] == 0:  # READ
            if self.read_count == self.MAX_FRAME_COUNT:
                status = Environment.STATUS_INVALID_ACTION
            else:
                self.read_count += 1
                self.input_buffer.append(self.vid_encoding[:,self.read_count, :])
                status = Environment.STATUS_VALID_ACTION

        if action.data.numpy()[0] == 1:  # WRITE
            if self.read_count == 0:
                # force the agent to read before first write
                status = Environment.STATUS_INVALID_ACTION
            elif self.write_count == self.MAX_TOKEN_COUNT:
                status = Environment.STATUS_INVALID_ACTION
            else:# use input_buffer[-1].unsqueeze(dim=1) instead of vid_encoding
                word_probs = self.decoder(self.vid_encoding, Variable(self.output_buffer[-1].long()))

                #next_word = torch.max(word_probs)
                prob, next_word = torch.max(word_probs, dim=2)

                target_word = self.caption.data[:,self.write_count]
                self.write_count += 1
                if (next_word.data[0] == target_word[0]).numpy():
                    status = Environment.STATUS_CORRECT_WORD
                if (next_word.data[0] != target_word[0]).numpy():
                    status = Environment.STATUS_CORRECT_WORD
                self.output_buffer.append(next_word.data)

        reward = self.give_reward(status)
        return reward

    def check_finished(self):
        return self.write_count >= 12# self.MAX_TOKEN_COUNT

    def give_reward(self, status):
        return {
            Environment.STATUS_VALID_ACTION: 0,
            Environment.STATUS_INVALID_ACTION: -10000,
            Environment.STATUS_CORRECT_WORD: 100,
            Environment.STATUS_INCORRECT_WORD: -100
        }[status]