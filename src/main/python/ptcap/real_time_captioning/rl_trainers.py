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
    def __init__(self, model, caption_loss_function, w_caption_loss, scheduler,
                 tokenizer, logger,
                 writer, checkpointer, load_encoder_only, folder=None,
                 filename=None,
                 gpus=None, clip_grad=None, classif_loss_function=None,
                 w_classif_loss=0):


        self.gpus = gpus
        self.model = model if self.gpus is None else(
            DataParallelWrapper(model, device_ids=self.gpus).cuda(gpus[0])
        )
        self.loss_function = caption_loss_function if self.gpus is None else(
            caption_loss_function.cuda(gpus[0])
        )

        self.tokenizer = tokenizer
        self.env = Environment()
        self.agent = Agent()

    def get_input_captions(self, captions, use_teacher_forcing):
        batch_size = captions.size(0)
        input_captions = torch.LongTensor(batch_size, 1).zero_()
        if use_teacher_forcing:
            input_captions = torch.cat([input_captions, captions[:, :-1]], 1)
        return input_captions

    def train(self, train_dataloader, valid_dataloader, criteria,
              max_num_epochs=None, frequency_valid=1, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):
        for i_episode in range(1000):
            self.run_episode(train_dataloader,i_episode, is_training=True,
                             use_teacher_forcing=teacher_force_train,
                             verbose=verbose_train)



    def run_episode(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):


        if is_training:
            self.model.train()
        else:
            self.model.eval()

        for i_episode, (videos, _, captions, _) in enumerate(
                dataloader):

            input_captions = self.get_input_captions(captions,
                                                     use_teacher_forcing)

            videos, captions, input_captions = (
            Variable(videos),
            Variable(captions),
            Variable(input_captions))
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
                input_captions = input_captions.cuda(self.gpus[0])


            env.reset(videos, input_Captions)
            finished = False
            reward_seq = []
            action_seq = []
            logprob_seq = []
            while not finished:
                state = env.get_state()
                action, logprob = agent.select_action(state)
                action_seq.append(action)
                logprob_seq.append(logprob)
                reward = env.update_state(action)
                reward_seq.append(reward)
                finished = env.check_finished()
            agent.update_policy(reward_seq, logprob_seq)

            if is_training:

                self.model.zero_grad()
                loss.backward()



        return



