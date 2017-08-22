
def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean() * 100.0
    return accuracy
