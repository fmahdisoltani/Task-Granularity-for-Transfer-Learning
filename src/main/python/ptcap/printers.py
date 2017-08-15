from ptcap.metrics import token_level_accuracy


def print_captions_and_predictions(tokenizer, captions, predictions):
    for cap, pred in zip(captions, predictions):
        decoded_cap = tokenizer.decode_caption(cap.data.numpy())
        decoded_pred = tokenizer.decode_caption(pred.data.numpy())

        print("__TARGET__: {}".format(decoded_cap))
        print("PREDICTION: {}\n".format(decoded_pred))

    print("*" * 30)


def print_metrics(accuracy, first_token_accuracy):
    print("Batch Accuracy is: {}".format(accuracy.data.numpy()[0]))
    print("First Token Accuracy is: {}".
          format(first_token_accuracy.data.numpy()[0]))


def print_loss(loss):
    print ("Loss: {}".format(loss.data.numpy()))

def print_stuff(loss, tokenizer, is_training, captions, predictions, epoch_counter,
                sample_counter, total_samples, verbose=True):
    predictions = predictions.cpu()
    captions = captions.cpu()
    # compute accuracy
    accuracy = token_level_accuracy(captions, predictions)
    first_accuracy = token_level_accuracy(captions, predictions, 1)
    status = "Training..." if is_training else "Validating..."

    print("Epoch {}".format(epoch_counter + 1))
    print(status + " batch #{} out of {} batches".
          format(sample_counter+1, total_samples))
    print_metrics(accuracy, first_accuracy)
    print_loss(loss.cpu())
    if verbose:
        print_captions_and_predictions(tokenizer, captions, predictions)