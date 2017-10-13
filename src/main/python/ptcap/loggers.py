import logging
import os


class CustomLogger(object):
    def __init__(self, folder, tokenizer):
        self.logging_path = os.path.join(folder, "log.txt")
        self.pred_path = os.path.join(folder, "out.txt")
        self.logger = logging.getLogger()
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

    def on_epoch_begin(self, is_training):
        self.epoch_counter += int(is_training)

    def on_epoch_end(self, scores_dict, is_training, total_samples):

        batch_msg = self.get_batch_msg(scores_dict, is_training, total_samples)
        self.logger.critical(batch_msg)
        if not is_training:
            ofile = open(self.pred_path, "a")
            ofile.write(batch_msg)

        self.sample_counter = 0

    def on_batch_begin(self):
        self.sample_counter += 1

    def on_batch_end(self, scores_dict, captions, predictions, is_training,
                     total_samples, verbose):

        batch_msg = self.get_batch_msg(scores_dict, is_training, total_samples)

        self.logger.info(batch_msg)

        if verbose:
            self.log_captions_and_predictions(captions, predictions)

    def on_train_end(self, best_score):
        self.logger.info("\nTrain complete!!!")
        self.logger.info("\nBest model has a score of {:.4}".format(best_score))

    def log_captions_and_predictions(self, captions, predictions):

        ofile = open(self.pred_path, "a")
        for cap, pred in zip(captions, predictions):
            decoded_cap = self.tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = self.tokenizer.decode_caption(pred.data.numpy())

            decoded_cap_str = "\n__TARGET__: {}".format(decoded_cap)
            decoded_pred_str = "\nPREDICTION: {}\n".format(decoded_pred)

            self.logger.info(decoded_cap_str)
            self.logger.info(decoded_pred_str)

            ofile.write(decoded_cap_str)
            ofile.write(decoded_pred_str)

        self.logger.info("*" * 30)

        ofile.write("*" * 30)
        ofile.close()

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
