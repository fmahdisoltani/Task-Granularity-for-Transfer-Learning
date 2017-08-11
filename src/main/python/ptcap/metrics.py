
def token_level_accuracy(captions, predictions, num_tokens=None):
    batch_size, _ = captions.size()
    equal_values = captions[:, 0:num_tokens].eq(predictions[:, 0:num_tokens])
    accuracy = equal_values.sum().float() * 100.0 / batch_size
    return accuracy
