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
        return loss / num_step
    
    
class WeightedSequenceCrossEntropy(nn.Module):

    def __init__(self, loss=nn.NLLLoss, kwargs=None):
        super().__init__()
        token_weights = None
        if kwargs and kwargs["token_freqs"]:

            token_freqs = kwargs["token_freqs"]
            token_weights = torch.from_numpy(np.array(
                [1/np.log(token_freqs[i]) for i in token_freqs],
                dtype=np.float32))
            print(token_weights)
        self.loss_function = loss(token_weights)

    def forward(self, preds, target):
        batch_size, num_step, _ = preds.size()
        loss = 0.

        for t in range(num_step):
             loss += self.loss_function(preds[:, t], target[:, t])

        return loss / num_step


class CrossEntropy(nn.Module):

    def __init__(self, loss=nn.NLLLoss):
        super().__init__()
        self.loss_function = loss()

    def forward(self, preds, targets):
        batch_size, num_features = preds.size()

        loss = self.loss_function(preds, targets)
        return loss
