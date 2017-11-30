from torch import nn
import numpy as np
import torch

class SequenceCrossEntropy(nn.Module):

    def __init__(self, loss=nn.NLLLoss):
        super(SequenceCrossEntropy, self).__init__()
        self.loss_function = loss()

    def forward(self, preds, target):
        batch_size, num_step, _ = preds.size()
        loss = 0.
        for t in range(num_step):
            loss += self.loss_function(preds[:, t], target[:, t])
        return loss / (batch_size*num_step)
    
    
class WeightedSequenceCrossEntropy(nn.Module):

    def __init__(self, token_freqs, loss=nn.NLLLoss):
        super().__init__()
        self.token_freqs = token_freqs
        self.loss_function = loss(weight=torch.from_numpy(np.array([1/np.log(self.token_freqs[i]) for i in range(len(self.token_freqs))], dtype=np.float32)))


    def forward(self, preds, target):
        batch_size, num_step, _ = preds.size()
        loss = 0.

        for t in range(num_step):
             loss += self.loss_function(preds[:, t], target[:, t])

        return loss / (num_step)


class CrossEntropy(nn.Module):

    def __init__(self, loss=nn.NLLLoss):
        super().__init__()
        self.loss_function = loss()

    def forward(self, preds, targets):
        batch_size, num_features = preds.size()

        loss = self.loss_function(preds, targets)
        return loss
