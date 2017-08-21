from collections import OrderedDict

from ptcap.metrics import token_level_accuracy


def print_captions_and_predictions(tokenizer, captions, predictions):
    for cap, pred in zip(captions, predictions):
        decoded_cap = tokenizer.decode_caption(cap.data.numpy())
        decoded_pred = tokenizer.decode_caption(pred.data.numpy())

        print("__TARGET__: {}".format(decoded_cap))
        print("PREDICTION: {}\n".format(decoded_pred))

    print("*" * 30)


def print_dict(scores_dict):
    for key, value in scores_dict.items():
        print("{} is: {}".format(key, value))


def print_stuff(loss, tokenizer, is_training, captions, predictions,
                epoch_counter, sample_counter, total_samples, verbose=True):
    #loss = loss.cpu()
    captions = captions.cpu()
    predictions = predictions.cpu()

    scores_dict = OrderedDict()

    scores_dict["loss"] = loss
    scores_dict["accuracy"] = token_level_accuracy(captions, predictions)
    scores_dict["first_token_accuracy"] = token_level_accuracy(captions,
                                                         predictions, 1)
    status = "Training..." if is_training else "Validating..."

    print("Epoch {}".format(epoch_counter + 1))
    print(status + " batch #{} out of {} batches".
          format(sample_counter+1, total_samples))

    print_dict(scores_dict)
    if verbose:
        print_captions_and_predictions(tokenizer, captions, predictions)
