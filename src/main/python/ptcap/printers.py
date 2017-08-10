from ptcap.metrics import token_level_accuracy


def print_captions_and_predictions(self, captions, predictions):
    for cap, pred in zip(captions, predictions):
        decoded_cap = self.tokenizer.decode_caption(cap.data.numpy())
        decoded_pred = self.tokenizer.decode_caption(pred.data.numpy())

        print("__TARGET__: {}".format(decoded_cap))
        print("PREDICTION: {}\n".format(decoded_pred))

    print("*" * 30)


def print_metrics(self, accuracy):
    print("Batch Accuracy is: {}".format(accuracy.data.numpy()[0]))


def print_stuff(self, is_training, captions, predictions, epoch_counter,
                sample_counter, total_samples, verbose=True):
    # compute accuracy
    accuracy = token_level_accuracy(captions, predictions)
    print("Training..." if is_training else "Validating...")

    print("Epoch {}".format(epoch_counter + 1))
    print("Sample #{} out of {} samples".
          format(sample_counter, total_samples))
    self.print_metrics(accuracy)
    if verbose:
        self.print_captions_and_predictions(captions, predictions)
