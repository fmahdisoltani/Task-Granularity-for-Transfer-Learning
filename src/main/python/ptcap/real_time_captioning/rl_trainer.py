import torch.optim as optim

from itertools import count

from ptcap.real_time_captioning.environment import Environment
from ptcap.real_time_captioning.agent import Agent


class RLTrainer(object):
    def __init__(self):

        self.env = Environment()
        self.agent = Agent()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(
                         self.optimizer, step_size=10000, gamma=0.9)

    def train(self):
        running_reward = 0
        logging_interval = 1000
        for i_episode in count(1):
            returns, action_seq = self.run_episode(i_episode)
            R = returns[0]
            running_reward += R

            if i_episode % logging_interval == 0:
                print('Episode {}\tAverage return: {:.2f}'.format(
                    i_episode,
                    running_reward / logging_interval))
                print(action_seq)
                running_reward = 0

    def run_episode(self, i_episode):

        #print("episode{}".format(i_episode))
        self.env.reset()

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



