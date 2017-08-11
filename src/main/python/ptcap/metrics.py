
# def token_level_accuracy(captions, predictions):
#     batch_size, num_steps = captions.size()
#     equal_values = captions.eq(predictions).sum().float()
#     accuracy = equal_values * 100.0 / (batch_size * num_steps)
#     return accuracy
#
#
# def first_token_accuracy(captions, predictions):
#     batch_size, _ = captions.size()
#     equal_values = captions[:, 0].eq(predictions[:, 0]).sum().float()
#     accuracy = equal_values * 100.0 / batch_size
#     return accuracy


def token_level_accuracy(captions, predictions, num_tokens=-1):
    batch_size, _ = captions.size()
    equal_values = captions[:, 0:num_tokens].eq(predictions[:, 0:num_tokens])
    accuracy = equal_values.sum().float() * 100.0 / batch_size
    return accuracy
