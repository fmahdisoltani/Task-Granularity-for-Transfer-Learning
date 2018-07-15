import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple


class ClassifEnv(nn.Module):
    # statuses
    STATUS_CORRECT_WRITE = 'correct_write'
    STATUS_INCORRECT_WRITE = 'incorrect_write'
    STATUS_INVALID_READ = 'invalid_read'
    STATUS_READ = 'read'
    STATUS_DONE = 'done'

    def __init__(self, encoder, classif_layer, correct_w_reward,
                 correct_r_reward, incorrect_w_reward, incorrect_r_reward,
                 tokenizer):
        super().__init__()

        self.encoder = encoder
        self.classif_layer = classif_layer

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.correct_w_reward = correct_w_reward
        self.correct_r_reward = correct_r_reward
        self.incorrect_w_reward = incorrect_w_reward
        self.incorrect_r_reward = incorrect_r_reward
        self.output_buffer = [0]
        self.tokenizer = tokenizer
        self.is_training = True
        #self.reset()

    def toggle_training_mode(self):
        self.is_training = not self.is_training

    def reset(self, video=None, classif=None):
        self.read_count = 0
        self.write_count = 0
        self.vid_encoding = self.encoder.extract_features(video)
        # input buffer contains the seen video frames, in the beginning only the
        # first frame of video

        #self.input_buffer = self.vid_encoding[:, 0, :]

    def set_rewards(self, correct_w_reward, correct_r_reward,
                    incorrect_w_reward, incorrect_r_reward):
        self.correct_w_reward = correct_w_reward
        self.correct_r_reward =  correct_r_reward,
        self.incorrect_w_reward = incorrect_w_reward
        self.incorrect_r_reward = incorrect_r_reward

    def get_state(self):
        return {
            "read_count": self.read_count,
            "write_count": self.write_count,
            "input_buffer": self.vid_encoding[:, self.read_count, :]
        }

    def update_state(self, action, action_seq=[], classif_targets=None):
        status = ""
        classif_probs = self.classify()
        value_prob = None
        ce_value = 1
        if action == 0:  # READ
            if self.read_count == 47:
                status = ClassifEnv.STATUS_INVALID_READ
            else:
                status = ClassifEnv.STATUS_READ
                #self.input_buffer = self.vid_encoding[:, self.read_count, :]
                self.read_count += 1

        if action == 1:  # WRITE

            classif_value_prob, classif_preds = torch.max(classif_probs, dim=1)
            if torch.equal(classif_preds, classif_targets):
                status = ClassifEnv.STATUS_CORRECT_WRITE
            else:
                status = ClassifEnv.STATUS_INCORRECT_WRITE

            self.write_count += 1
            ce_value = torch.max(classif_probs).data[0]

        reward = self.give_reward(status, value_prob, classif_probs, classif_targets)
    #    reward = torch.sum(classif_probs)
        return reward*ce_value, classif_probs

    def check_finished(self):
        return self.write_count == 1 or self.read_count == 48

    def give_reward(self, status, value_prob=None,
                    classif_probs=None,classif_targets=None):
        r = {
            ClassifEnv.STATUS_CORRECT_WRITE: self.correct_w_reward, #  classif_probs[0,classif_targets.data.cpu().numpy()[0]],
            ClassifEnv.STATUS_READ: self.correct_r_reward,
            ClassifEnv.STATUS_INCORRECT_WRITE: self.incorrect_w_reward,#*
            ClassifEnv.STATUS_INVALID_READ: self.incorrect_r_reward,
        }[status]
        #r = r if value_prob is None else (-1/(1+value_prob)) * r
        return r

    def classify(self):
        #features = self.input_buffer[-1]
        pre_activation = self.classif_layer(self.vid_encoding[:, self.read_count, :])
        probs = self.logsoftmax(pre_activation)
        if probs.ndimension() == 3:
            probs = probs.mean(dim=1)  # probs: [8*48*178]

        return probs



