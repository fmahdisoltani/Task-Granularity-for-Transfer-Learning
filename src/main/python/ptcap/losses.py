from torch import nn


class SequenceCrossEntropy(nn.Module):
    def __init__(self):
        super(SequenceCrossEntropy, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, predictions, targets):
        num_step = predictions.size()[1]
        loss = 0.
        for t in range(num_step):
            loss += self.loss(predictions[:, t], targets[:, t])
        return loss / num_step
