import logging
import os


class CustomLogger(object):
    def __init__(self, folder, verbose=True):
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

    def log_captions_and_predictions(self, tokenizer, captions, predictions):

        ofile = open(self.pred_path, "a")
        for cap, pred in zip(captions, predictions):
            decoded_cap = tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = tokenizer.decode_caption(pred.data.numpy())

            decoded_cap_str = "\n__TARGET__: {}".format(decoded_cap)
            decoded_pred_str = "\nPREDICTION: {}\n".format(decoded_pred)

            self.logger.info(decoded_cap_str)
            self.logger.info(decoded_pred_str)

            ofile.write(decoded_cap_str)
            ofile.write(decoded_pred_str)

        self.logger.info("*" * 30)

        ofile.write("*" * 30)
        ofile.close()

    def log_batch_end(self, scores_dict, tokenizer, captions, predictions,
                      is_training, sample_counter, total_samples, epoch_counter,
                      verbose):
        phase = "_Train_" if is_training else "_Valid_"
        to_be_logged = ("\rEpoch {} - {} - batch {}/{} - ".
                         format(epoch_counter, phase, sample_counter,
                                total_samples))
        for key, value in scores_dict.items():
            to_be_logged += "{}: {:.4} ".format(key, value)

        if sample_counter == total_samples:
            # if it is the last batch, also write info to log file
            self.logger.critical(to_be_logged)
            if not is_training:
                ofile = open(self.pred_path, "a")
                ofile.write(to_be_logged)
        else:
            self.logger.info(to_be_logged)

        if verbose:
            self.log_captions_and_predictions(tokenizer, captions, predictions)

    def log_train_end(self, best_score):
        self.logger.info("\nTrain complete!!!")
        self.logger.info("\nBest model has a score of {:.4}".format(best_score))

    def log_message(self, message, args):
        self.logger.critical(message.format(*args))