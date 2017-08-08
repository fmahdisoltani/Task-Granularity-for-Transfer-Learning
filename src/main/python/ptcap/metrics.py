
def token_level_accuracy(captions, predictions):
    batch_size, num_steps = captions.size()
    equal_values = captions.eq(predictions).sum().float()
    accuracy = equal_values * 100.0 / (batch_size * num_steps)
    return accuracy
