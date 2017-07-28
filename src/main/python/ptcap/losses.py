from torch import nn


class SequenceCrossEntropy(nn.Module):
    loss_function = nn.NLLLoss()

    def __init__(self):
        super(SequenceCrossEntropy, self).__init__()


    @classmethod
    def forward(cls, preds, target):
        num_step = preds.size()[1]
        loss = 0.
        for t in range(num_step):
            loss += cls.loss_function(preds[:, t], target[:, t])
        return loss / num_step
