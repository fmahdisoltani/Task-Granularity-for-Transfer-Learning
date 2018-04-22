import torch.nn as nn
import torch
import numpy as np

from torch.autograd import Variable


class Agent:
    def __init__(self, input_size=2, hidden_size=33, num_actions=2):

        self.input_layer = nn.Linear(input_size, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax()
        )
        self.lstm_hidden = None

    def get_action_probs(self, x):
        #orig = self.prepare_policy_input(x)
        x = self.prepare_policy_input(x)
        x = self.input_layer(x)
        lstm_output, self.lstm_hidden = self.lstm(x, self.lstm_hidden)
        self.lstm.flatten_parameters()

        #lstm_out_projected = self.output_layer(lstm_output)
        #action_probs = self.softmax(lstm_out_projected)

        #return action_probs.squeeze(dim=1)
        return self.policy(x).squeeze(dim=1)
    def prepare_policy_input(self, state):
#        last_token = state['output_buffer'][-1]
#        last_vid_feature = state['input_buffer'][-1]
        rc = state['read_count']
        #rc2 = torch.from_numpy(np.resize(rc,(1,1)).float())

        #torch.cat([policy_input, torch.from_numpy(rc).float()], dim=1)
        wc = state['write_count']
 #       policy_input = torch.cat((last_vid_feature.data, last_token.float()), dim=1)
       # policy_input = torch.cat([policy_input, rc * torch.ones((1, 1)), wc * torch.ones((1,1))], dim=1)
        policy_input = torch.cat([rc * torch.ones((1, 1)), wc * torch.ones((1,1))], dim=1)
        return Variable(torch.unsqueeze(policy_input,dim=1))
    def select_action(self, state):
        action_probs = self.get_action_probs(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        log_prob = torch.sum(m.log_prob(action))

        #Hard coded policy
        # if state["read_count"] < 47:
        #     action = torch.zeros(1).long()
        #     log_prob = torch.from_numpy(np.array([1,0]))
        # else:
        #     action = torch.ones(1).long()
        #     log_prob = torch.from_numpy(np.array([ 0,1]))
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
        print("returns")
        print(returns)
        returns = torch.Tensor(returns)
        # subtract mean and std for faster training
        returns = (returns - returns.mean()) / (returns.std() +
                                                np.finfo(np.float32).eps)
        for log_prob, r in zip(logprobs_seq, returns):
            policy_loss.append(-log_prob * r)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward(retain_graph=True)
        # note: retain_graph=True allows for multiple calls to .backward()
        # in a single step