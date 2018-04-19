import torch
import numpy as np

from collections import namedtuple
from collections import OrderedDict

from torch.autograd import Variable

from ptcap.scores import (ScoresOperator, caption_accuracy, classif_accuracy,
                          first_token_accuracy, loss_to_numpy, token_accuracy)
from ptcap.utils import DataParallelWrapper
from ptcap.real_time_captioning.environment import Environment
from ptcap.real_time_captioning.agent import Agent


class RLTrainer(object):
    def __init__(self, encoder,decoder, caption_loss_function, tokenizer, gpus=None):

        self.gpus = gpus
        self.use_cuda = True if gpus else False
        self.encoder = encoder if self.gpus is None else(
            DataParallelWrapper(encoder, device_ids=self.gpus).cuda(gpus[0])
        )
        self.decoder = decoder if self.gpus is None else(
            DataParallelWrapper(decoder, device_ids=self.gpus).cuda(gpus[0])
        )
        self.loss_function = caption_loss_function if self.gpus is None else(
            caption_loss_function.cuda(gpus[0])
        )

        self.tokenizer = tokenizer
        self.env = Environment(self.encoder, self.decoder)
        self.agent = Agent()

    def get_input_captions(self, captions, use_teacher_forcing):
        batch_size = captions.size(0)
        input_captions = torch.LongTensor(batch_size, 1).zero_()
        if use_teacher_forcing:
            input_captions = torch.cat([input_captions, captions[:, :-1]], 1)
        return input_captions

    def train(self, train_dataloader, teacher_force_train=True,
               verbose_train=False):
        for i_episode in range(1000):
            self.run_episode(train_dataloader, i_episode, is_training=True,
                             use_teacher_forcing=teacher_force_train,
                             verbose=verbose_train)

    def run_episode(self, dataloader, i_episode, is_training,
                  use_teacher_forcing=False, verbose=True):
        print("*****inside episode*****{}".format("%"*3))


        #if is_training: self.model.train()
        #else: self.model.eval()

        for sample_counter, (videos, _, captions, _) in enumerate(dataloader):

            input_captions = self.get_input_captions(captions,
                                                     use_teacher_forcing)

            videos, captions, input_captions = (Variable(videos),Variable(captions),
            Variable(input_captions))
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
                input_captions = input_captions.cuda(self.gpus[0])

            # Reset the RL environment
            self.env.reset(videos, input_captions)
            finished = False
            reward_seq = []
            action_seq = []
            logprob_seq = []
            while not finished:
                state = self.env.get_state()
                action, logprob = self.agent.select_action(state)
                action_seq.append(action)
                logprob_seq.append(logprob)
                reward = self.env.update_state(action)
                reward_seq.append(reward)
                finished = self.env.check_finished()
            print([a.data[0] for a in action_seq])
            print(reward_seq)

            self.agent.update_policy(reward_seq, logprob_seq)
            print("episode{}: sample{}".format(i_episode,sample_counter))
            if sample_counter==1:
                break




        return



