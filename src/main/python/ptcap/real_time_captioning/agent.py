import torch.nn as nn
import torch
import numpy as np

from torch.autograd import Variable


class Agent:
    def __init__(self, input_size=2, hidden_size=33, num_actions=2):

        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=2)
        )

    def get_action_probs(self, x):
        x = self.prepare_policy_input(x)
        return self.policy(x).squeeze(dim=1)

    def prepare_policy_input(self, state):
        rc = state['read_count']
        wc = state['write_count']
        policy_input = torch.cat([rc * torch.ones((1, 1)),
                                  wc * torch.ones((1,1))], dim=1)
        return Variable(torch.unsqueeze(policy_input,dim=1))

    def select_action(self, state):
        action_probs = self.get_action_probs(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action))
        return action, log_prob

    def compute_returns(self,rewards, gamma=1.0):
        """
        Compute returns for each time step, given the rewards
          @param rewards: list of floats, where rewards[t] is the reward
                          obtained at time step t
          @param gamma: the discount factor
          @returns list of floats representing the episode's returns
              G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...
        """
        G_last = 0
        G = []
        for r in reversed(rewards):
            G_t = G_last * gamma + r
            G.append(G_t)
            G_last = G_t
        G.reverse()
        return G

    def update_policy(self, reward_seq, logprobs_seq, gamma=1.0):

        policy_loss = []
        returns = self.compute_returns(reward_seq, gamma)
        returns = torch.Tensor(returns)
        # subtract mean and std for faster training
        #returns = (returns - returns.mean()) / (returns.std() +
        #                                        np.finfo(np.float32).eps)
        for log_prob, r in zip(logprobs_seq, returns):
            policy_loss.append(-log_prob * r)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward(retain_graph=True)
        # note: retain_graph=True allows for multiple calls to .backward()
        # in a single step
        return returns
