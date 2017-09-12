import logging
import os


class CustomLogger(object):
    def __init__(self, folder, verbose=False):
        self.logging_path = os.path.join(folder, "log.txt")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.logging_path)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        if verbose:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            self.logger.addHandler(sh)

    def log_captions_and_predictions(self, tokenizer, captions, predictions):
        for cap, pred in zip(captions, predictions):
            decoded_cap = tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = tokenizer.decode_caption(pred.data.numpy())

            self.logger.info("\n__TARGET__: {}".format(decoded_cap))
            self.logger.info("PREDICTION: {}\n".format(decoded_pred))

        self.logger.info("*" * 30)

    def log_stuff(self, scores_dict, tokenizer, is_training, captions,
                  predictions, epoch_counter, total_samples, verbose, sample_count):

        phase = "Training" if is_training else "Validating"
        self.logger.info("Epoch {} - {} - batch_size {} -".
                    format(epoch_counter, phase, total_samples))
        self.log_dict(scores_dict)
        if verbose:
            self.log_captions_and_predictions(tokenizer, captions, predictions)

    def log_dict(self, scores_dict):
        for key, value in scores_dict.items():
            if "average" in key:
                self.logger.info("{}: {:.4} -".format(key, value))
