import torch


def token_level_accuracy(captions, outputs):
    batch_size, num_steps = captions.size()
    _, predictions = torch.max(outputs, dim=2)
    equal_values = captions.eq(predictions).sum().float()
    accuracy = equal_values * 100.0 / (batch_size * num_steps)
    return accuracy
