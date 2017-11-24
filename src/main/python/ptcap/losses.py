from torch import nn


class SequenceCrossEntropy(nn.Module):

    def __init__(self, loss=nn.NLLLoss):
        super(SequenceCrossEntropy, self).__init__()
        self.loss_function = loss()

    def forward(self, preds, target):
        num_step = preds.size(1)
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

