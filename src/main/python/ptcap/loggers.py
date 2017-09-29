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

        sh = logging.StreamHandler()
        sh.terminator = ""
        sh.setLevel(logging.WARNING)

        if verbose:
            sh.setLevel(logging.INFO)

        self.logger.addHandler(sh)

        self.epoch = 0

    def log_captions_and_predictions(self, tokenizer, captions, predictions):
        for cap, pred in zip(captions, predictions):
            decoded_cap = tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = tokenizer.decode_caption(pred.data.numpy())

            self.logger.info("\n__TARGET__: {}".format(decoded_cap))
            self.logger.info("\nPREDICTION: {}\n".format(decoded_pred))

        self.logger.info("*" * 30)

    def log_batch_begin(self):
        raise NotImplementedError

    def log_batch_end(self, scores_dict, tokenizer, captions, predictions,
                      is_training, sample_counter, total_samples, verbose):
        phase = "Training" if is_training else "Validating"
        self.logger.info("\rEpoch {} - {} - batch {}/{} - ".
                         format(self.epoch, phase, sample_counter,
                                total_samples))
        self.log_dict(scores_dict)
        if verbose:
            self.log_captions_and_predictions(tokenizer, captions, predictions)

    def log_dict(self, scores_dict):
        for key, value in scores_dict.items():
            self.logger.info("{}: {:.4} ".format(key, value))

    def log_epoch_begin(self, is_training, epoch_counter):
        phase = "Training" if is_training else "Validating"
        self.logger.info("Epoch {} - {}\n".format(epoch_counter, phase))
        self.epoch = epoch_counter

    def log_epoch_end(self, scores_dict):
        self.logger.info("\n")
        self.log_dict(scores_dict)
        self.logger.info("\n")
