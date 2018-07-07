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

    def __init__(self, encoder, decoder, classif_layer, correct_w_reward,
                 correct_r_reward, incorrect_w_reward, incorrect_r_reward,
                 tokenizer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.classif_layer = classif_layer

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.correct_w_reward = correct_w_reward
        self.correct_r_reward = correct_r_reward
        self.incorrect_w_reward = incorrect_w_reward
        self.incorrect_r_reward = incorrect_r_reward
        self.output_buffer = []
        self.tokenizer = tokenizer
        self.is_training = True
        #self.reset()

    def toggle_training_mode(self):
        self.is_training = not self.is_training

    def reset(self, video=None, caption=None, classif=None):
        self.read_count = 0
        self.write_count = 0
        self.vid_encoding = self.encoder.extract_features(video)
        self.output_buffer = []
        # input buffer contains the seen video frames, in the beginning only the
        # first frame of video

        #self.input_buffer = self.vid_encoding[:, 0, :]

        self.tf_input = self.get_input_captions(caption, True)  # shifted_targets
        self.non_tf_input = self.get_input_captions(caption, False) # <go> tokens only
        self.decoder.module.reset_hidden()

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
            "input_buffer": self.vid_encoding[:, min(self.read_count, 47), :],
            "output_buffer": self.output_buffer
        }

    def update_state(self, action, action_seq=[], classif_targets=None, caption_targets=None, non_tf_input=None):
        status = ""
        classif_probs = self.classify()
        caption_probs = None
        value_prob = None

        if action == 0:  # READ
            if self.read_count >= 47:
                status = Environment.STATUS_INVALID_READ
            else:
                status = Environment.STATUS_READ
                #self.input_buffer = self.vid_encoding[:, self.read_count, :]
            self.read_count += 1

        if action == 1:  # WRITE
            if self.is_training: #teacher_force
                decoder_input = self.tf_input[:, self.write_count:self.write_count + 1]
            else:
                decoder_input = self.non_tf_input

            caption_probs = self.step_decoder(decoder_input)

            # classif_value_prob, classif_preds = torch.max(classif_probs, dim=1)
            cap_value_probs, cap_preds = torch.max(caption_probs, dim=2)
            self.non_tf_input = cap_preds
            # if torch.equal(prediction, classif_targets):
            #from pycocoevalcap.bleu.bleu import Bleu
            #partial_bleu = Bleu()

            gg = cap_preds.cpu().numpy()[0][0]
            self.output_buffer.append(gg)

            if torch.equal(cap_preds, caption_targets[:, self.write_count:self.write_count+1]):
                status = Environment.STATUS_CORRECT_WRITE
            else:
                status = Environment.STATUS_INCORRECT_WRITE

            self.write_count += 1

        reward = self.give_reward(status, value_prob)
    #    reward = torch.sum(classif_probs)
        return reward, classif_probs, caption_probs

    def check_finished(self):
        return self.write_count == 13 or \
               (self.read_count + self.write_count)>=60

    def give_reward(self, status, value_prob=None):
        r = {
            Environment.STATUS_CORRECT_WRITE: self.correct_w_reward,
            Environment.STATUS_READ: self.correct_r_reward,
            Environment.STATUS_INCORRECT_WRITE: self.incorrect_w_reward,
            Environment.STATUS_INVALID_READ: self.incorrect_r_reward,
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


    def get_input_captions(self, captions, use_teacher_forcing):
        batch_size = captions.size(0)
        input_captions = torch.LongTensor(batch_size, 1).zero_().cuda()
        if use_teacher_forcing:
            input_captions = torch.cat([input_captions, captions[:, :-1]], 1)
        return input_captions

    def step_decoder(self, input_captions):
        #self.decoder(self.vid_encoding, self.output_buffer[self.write_count-1])
        caption_probs = self.decoder(self.vid_encoding[:, self.read_count:self.read_count+1, :],
                     input_captions)
        return caption_probs

    def run_decoder_multi_step(self, input_captions):
        caption_probs = self.decoder(self.vid_encoding, input_captions)

        return caption_probs



