def print_captions_and_predictions(tokenizer, captions, predictions):
    for cap, pred in zip(captions, predictions):
        decoded_cap = tokenizer.decode_caption(cap.data.numpy())
        decoded_pred = tokenizer.decode_caption(pred.data.numpy())

        print("__TARGET__: {}".format(decoded_cap))
        print("PREDICTION: {}\n".format(decoded_pred))

    print("*" * 30)


def print_dict(scores_dict):
    for key, value in scores_dict.items():
        print("{} is: {:.4f} -".format(key, value), end=" ")


def print_stuff(scores_dict, tokenizer, is_training, captions, predictions,
                epoch_counter, sample_counter, total_samples, verbose=True):

    phase = "Training" if is_training else "Validating"

    print("\rEpoch {} - {} - batch {}/{} -".
          format(epoch_counter + 1, phase, sample_counter+1, total_samples),
          end=" ")

    print_dict(scores_dict)
    if verbose:
        print_captions_and_predictions(tokenizer, captions, predictions)
