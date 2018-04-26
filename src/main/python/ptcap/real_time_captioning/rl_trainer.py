import torch.optim as optim
import torch

from itertools import count
from torch.autograd import Variable

from ptcap.real_time_captioning.environment import Environment
from ptcap.real_time_captioning.agent import Agent
from ptcap.utils import DataParallelWrapper


class RLTrainer(object):
    def __init__(self, encoder, checkpointer, gpus=None):

        self.env = Environment(encoder)
        self.agent = Agent()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(
                         self.optimizer, step_size=10000, gamma=0.9)
        self.gpus = gpus
        self.use_cuda = True if gpus else False
        self.encoder = encoder if self.gpus is None else(
            DataParallelWrapper(encoder, device_ids=self.gpus).cuda(gpus[0])
        )
        self.checkpointer = checkpointer


    def get_input_captions(self, captions, use_teacher_forcing):
        batch_size = captions.size(0)
        input_captions = torch.LongTensor(batch_size, 1).zero_()
        if use_teacher_forcing:
            input_captions = torch.cat([input_captions, captions[:, :-1]], 1)
        return input_captions

    def train(self, dataloader):

        running_reward = 0
        logging_interval = 1000
        #for i_episode in count(1):
        for i_episode, (videos, _, captions, _) in enumerate(dataloader):
            input_captions = self.get_input_captions(captions,
                                                     use_teacher_forcing=False)
            videos, captions, input_captions = (
            Variable(videos), Variable(captions),
                     Variable(input_captions))
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
                input_captions = input_captions.cuda(self.gpus[0])

            returns, action_seq = self.run_episode(i_episode, videos)
            R = returns[0]
            running_reward += R

            if i_episode % logging_interval == 0:
                print('Episode {}\tAverage return: {:.2f}'.format(
                    i_episode,
                    running_reward / logging_interval))
                print(action_seq)
                running_reward = 0

    def run_episode(self, i_episode, videos):

        #print("episode{}".format(i_episode))
        self.env.reset(videos)

        finished = False
        reward_seq = []
        action_seq = []
        logprob_seq = []
        while not finished:

            state = self.env.get_state()
            action, logprob = self.agent.select_action(state)
            action_seq.append(action.data.numpy()[0])
            logprob_seq.append(logprob)
            reward = self.env.update_state(action, action_seq)
            reward_seq.append(reward)
            finished = self.env.check_finished()

        returns = self.agent.update_policy(reward_seq, logprob_seq)

        if i_episode % 1 == 0:  # replace 1 with batch_size
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()


        return returns, action_seq



