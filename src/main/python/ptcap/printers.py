from ptcap.metrics import token_level_accuracy


def print_captions_and_predictions(tokenizer, captions, predictions):
    for cap, pred in zip(captions, predictions):
        decoded_cap = tokenizer.decode_caption(cap.data.numpy())
        decoded_pred = tokenizer.decode_caption(pred.data.numpy())

        print("__TARGET__: {}".format(decoded_cap))
        print("PREDICTION: {}\n".format(decoded_pred))

    print("*" * 30)


def print_metrics(accuracy):
    print("Batch Accuracy is: {}".format(accuracy.data.numpy()[0]))


def print_stuff(tokenizer, is_training, captions, predictions, epoch_counter,
                sample_counter, total_samples, verbose=True):
    predictions = predictions.cpu()
    captions = captions.cpu()
    # compute accuracy
    accuracy = token_level_accuracy(captions, predictions)
    status = "Training..." if is_training else "Validating..."

    print("Epoch {}".format(epoch_counter + 1))
    print(status + " sample #{} out of {} samples".
          format(sample_counter, total_samples))
    print_metrics(accuracy)
    if verbose:
        print_captions_and_predictions(tokenizer, captions, predictions)

