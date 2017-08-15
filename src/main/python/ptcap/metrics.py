
def token_level_accuracy(captions, predictions, num_tokens=None):
    batch_size, num_step = captions.size()
    if not num_tokens:
        num_tokens = num_step
    equal_values = captions[:, 0:num_tokens].eq(predictions[:, 0:num_tokens])
    accuracy = equal_values.sum().float() * 100.0 / (batch_size * num_tokens)
    return accuracy