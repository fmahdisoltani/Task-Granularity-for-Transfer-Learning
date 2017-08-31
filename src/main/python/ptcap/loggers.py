import logging
import os


def log_captions_and_predictions(tokenizer, captions, predictions, logger):
    for cap, pred in zip(captions, predictions):
        decoded_cap = tokenizer.decode_caption(cap.data.numpy())
        decoded_pred = tokenizer.decode_caption(pred.data.numpy())

        logger.info("\n__TARGET__: {}".format(decoded_cap))
        logger.info("PREDICTION: {}\n".format(decoded_pred))

    logger.info("*" * 30)


def log_stuff(scores_dict, tokenizer, is_training, captions, predictions,
              epoch_counter, total_samples, verbose, logger,sample_count):

    phase = "Training" if is_training else "Validating"
    logger.info("Epoch {} - {} - batch_size {} -".
                format(epoch_counter, phase, total_samples))
    log_dict(scores_dict, logger)
    if verbose:
        log_captions_and_predictions(tokenizer, captions, predictions, logger)


def log_dict(scores_dict, logger):
    for key, value in scores_dict.items():
        if "average" in key:
            logger.info("{}: {:.4f} -".format(key, value))


def info_logger(folder, verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging_path = os.path.join(folder, "log_filename.txt")
    fh = logging.FileHandler(logging_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    return logger