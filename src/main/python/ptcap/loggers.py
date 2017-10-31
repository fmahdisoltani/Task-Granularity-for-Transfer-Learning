import logging
import os
import time


class CustomLogger(object):
    def __init__(self, folder, tokenizer):
        self.logging_path = os.path.join(folder, "log.txt")
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.logging_path)
        fh.setLevel(logging.CRITICAL)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.terminator = ""
        sh.setLevel(logging.INFO)

        self.logger.addHandler(sh)

        self.epoch_counter = 0
        self.sample_counter = 0
        self.tokenizer = tokenizer

        self.outputs_path = os.path.join(folder, "out.txt")
        self.outputs_logger = logging.getLogger("outputs_logger")
        self.outputs_logger.setLevel(logging.CRITICAL)

        fh = logging.FileHandler(self.outputs_path)
        fh.setLevel(logging.CRITICAL)
        self.outputs_logger.addHandler(fh)

        self.start_time = None

    def on_epoch_begin(self, epoch_counter):
        self.epoch_counter = epoch_counter

    def on_epoch_end(self, scores_dict, is_training, total_samples):

        batch_msg = self.get_batch_msg(scores_dict, is_training, total_samples)
        self.logger.critical(batch_msg)
        if not is_training:
            self.outputs_logger.critical(batch_msg)
        self.sample_counter = 0

    def on_batch_begin(self):
        self.sample_counter += 1

    def on_batch_end(self, scores_dict, captions, predictions, is_training,
                     total_samples, verbose):

        batch_msg = self.get_batch_msg(scores_dict, is_training, total_samples)

        self.logger.info(batch_msg)

        if verbose:
            self.log_captions_and_predictions(captions, predictions)

    def on_train_init(self, folder, filename):
        if folder is None or filename is None:
            self.logger.critical("Running the model from scratch")
        else:
            self.logger.critical("Loaded checkpoint {}/{}".
                                 format(folder, filename))

    def on_train_begin(self):
        self.start_time = time.time()

    def on_train_end(self, best_score):
        end_time = time.time()
        self.logger.info("\nTraining complete!!!")
        self.logger.info("\nBest model has a score of {:.4}".format(best_score))
        self.logger.info("program took {}".format(end_time - self.start_time))

    def log_captions_and_predictions(self, captions, predictions):

        for cap, pred in zip(captions, predictions):
            decoded_cap = self.tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = self.tokenizer.decode_caption(pred.data.numpy())

            decoded_cap_str = "\n__TARGET__: {}".format(decoded_cap)
            decoded_pred_str = "\nPREDICTION: {}\n".format(decoded_pred)

            self.logger.info(decoded_cap_str)
            self.logger.info(decoded_pred_str)

            self.outputs_logger.critical(decoded_cap_str)
            self.outputs_logger.critical(decoded_pred_str)

        self.outputs_logger.critical("*" * 30)

    def log_message(self, message, args):
        self.logger.critical(message.format(*args))

    def get_batch_msg(self, scores_dict, is_training, total_samples):
        phase = "Train" if is_training else "Valid"
        msg = ("\rEpoch {} - {} - batch {}/{} - ".
                         format(self.epoch_counter, phase, self.sample_counter,
                                total_samples))
        for key, value in scores_dict.items():
            msg += "{}: {:.4} ".format(key, value)

        return msg
