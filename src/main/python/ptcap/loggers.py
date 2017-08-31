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
    logger.info("***********************************************")


def log_dict(scores_dict, logger):
    for key, value in scores_dict.items():
        if "average" in key:
            logger.info("{}: {:.4f} -".format(key, value))


def info_logger(verbose=False):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # formatter = logging.Formatter(
    #      '%(asctime)s - %(process)d - %(message)s')

    fh = logging.FileHandler('log_filename.txt')
    fh.setLevel(logging.INFO)
    # fh.setFormatter(formatter)
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger