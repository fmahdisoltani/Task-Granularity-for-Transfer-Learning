from torch import nn


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
