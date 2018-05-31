import copy
import os

import torch.optim

import ptcap.data.preprocessing as prep
import ptcap.losses
import ptcap.model.captioners
import ptcap.model.decoders as dec
import ptcap.model as all_models

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ptcap.checkpointers import Checkpointer
from ptcap.data.annotation_parser import JsonParser, V2Parser
from ptcap.data.dataset import (JpegVideoDataset, GulpVideoDataset,
                                NumpyVideoDataset)
from ptcap.data.tokenizer import Tokenizer
from ptcap.loggers import CustomLogger
from ptcap.tensorboardY import Seq2seqAdapter
from ptcap.trainers import Trainer
from rtorchn.data.preprocessing import CenterCropper
from ptcap.losses import *


def train_model(config_obj, relative_path=""):
    # Find paths to training, validation and test sets
    training_path = os.path.join(relative_path,
                                 config_obj.get("paths", "train_annot"))
    validation_path = os.path.join(relative_path,
                                   config_obj.get("paths", "validation_annot"))

    # Load attributes of config file
    caption_type = config_obj.get("targets", "caption_type")
    checkpoint_folder = os.path.join(
        relative_path, config_obj.get("paths", "checkpoint_folder"))
    higher_is_better = config_obj.get("criteria", "higher_is_better")
    clip_grad = config_obj.get("training", "clip_grad")
    frequency_valid = config_obj.get("validation", "frequency")
    gpus = config_obj.get("device", "gpus")
    num_epoch = config_obj.get("training", "num_epochs")
    pretrained_folder = config_obj.get("pretrained", "pretrained_folder")
    pretrained_file = config_obj.get("pretrained", "pretrained_file")
    pretrained_folder = os.path.join(relative_path, pretrained_folder
                                   ) if pretrained_folder else None
    teacher_force_train = config_obj.get("training", "teacher_force")
    teacher_force_valid = config_obj.get("validation", "teacher_force")
    verbose_train = config_obj.get("training", "verbose")
    verbose_valid = config_obj.get("validation", "verbose")
    # annot_parser = config_obj.get("annot_type")
    annot_type = "v2"

    # Get model, loss, optimizer, scheduler, and criteria from config_file
    model_type = config_obj.get("model", "type")
    caption_loss_type = config_obj.get("loss", "caption_loss")
    balanced_loss = config_obj.get("loss", "balanced")
    w_caption_loss = config_obj.get("loss", "w_caption_loss")
    classif_loss_type = config_obj.get("loss", "classif_loss")
    w_classif_loss = config_obj.get("loss", "w_classif_loss")
    optimizer_type = config_obj.get("optimizer", "type")
    scheduler_type = config_obj.get("scheduler", "type")
    criteria = config_obj.get("criteria", "score")
    videos_folder = config_obj.get("paths", "videos_folder")


    # Load Json annotation files
    if annot_type=="json":
        training_parser = JsonParser(training_path, os.path.join(relative_path,
                                 videos_folder), caption_type=caption_type)
        validation_parser = JsonParser(validation_path, os.path.join(relative_path,
                                   videos_folder), caption_type=caption_type)
    elif annot_type=="v2":
        training_parser = V2Parser(training_path, os.path.join(relative_path,
                                                                 videos_folder),
                                     caption_type=caption_type)
        validation_parser = V2Parser(validation_path,
                                       os.path.join(relative_path,
                                                    videos_folder),
                                       caption_type=caption_type)

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(**config_obj.get("tokenizer", "kwargs"))
    if pretrained_folder:
        tokenizer.load_dictionaries(pretrained_folder)
        print("Inside pretrained", tokenizer.get_vocab_size())
    else:
        tokenizer.build_dictionaries(training_parser.get_captions_from_tmp_and_lbl())


    prep_list = []
    for i in config_obj.get("preprocess", "train"):
        args = i["args"] or ()
        prep_list.append(getattr(ptcap.data.preprocessing, i["type"])(*args))

    train_preprocessor = Compose(prep_list)

    # train_prep_type = config_obj.get("preprocess", "train_prep_type")
    # train_preprocessor = getattr(ptcap.data.preprocessing, train_prep_type)(
    #      **config_obj.get("preprocess", "train_prep_kwargs"))



    train_dataset_type = config_obj.get("dataset", "train_dataset_type")
    train_dataset_kwargs = config_obj.get("dataset", "train_dataset_kwargs")
    train_dataset_kwargs = train_dataset_kwargs or {}
    train_dataset = getattr(ptcap.data.dataset, train_dataset_type)(
                            annotation_parser=training_parser,
                            tokenizer=tokenizer,
                            preprocess=train_preprocessor,
                            **train_dataset_kwargs)

    val_prep_list = []
    for i in config_obj.get("preprocess", "valid"):
        args = i["args"] or ()
        val_prep_list.append(getattr(ptcap.data.preprocessing, i["type"])(*args))

    val_preprocessor = Compose(val_prep_list)

    # val_prep_type = config_obj.get("preprocess", "val_prep_type")
    # val_preprocessor = getattr(ptcap.data.preprocessing, val_prep_type)(
    #     **config_obj.get("preprocess", "val_prep_kwargs"))
    #
    # validation_set = NumpyVideoDataset(annotation_parser=validation_parser,
    #                                    tokenizer=tokenizer,
    #                                    preprocess=val_preprocessor)

    validation_set = train_dataset

    dataloader = DataLoader(train_dataset, shuffle=True, drop_last=False,
                            **config_obj.get("dataloaders", "kwargs"))

    val_dataloader = DataLoader(validation_set, shuffle=True, drop_last=False,
                                **config_obj.get("dataloaders", "kwargs"))

    encoder_type = config_obj.get("model", "encoder")
    decoder_type = config_obj.get("model", "decoder")
    encoder_args = config_obj.get("model", "encoder_args")
    encoder_kwargs = config_obj.get("model", "encoder_kwargs")
    decoder_args = config_obj.get("model", "decoder_args")
    decoder_kwargs = config_obj.get("model", "decoder_kwargs")
    decoder_kwargs["vocab_size"] = tokenizer.get_vocab_size()

    # TODO: Remove GPUs?
    gpus = config_obj.get("device", "gpus")

    # decoder_kwargs["gpus"] = gpus

    # Create model, loss, and optimizer objects
    model = getattr(ptcap.model.captioners, model_type)(
        encoder=getattr(all_models, encoder_type),
        decoder=getattr(all_models, decoder_type),
        encoder_args=encoder_args,
        encoder_kwargs=encoder_kwargs,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        gpus=gpus)

    # loss_function = getattr(ptcap.losses, loss_type)()
    caption_loss_kwargs = {}
    if balanced_loss:
        caption_loss_kwargs["token_freqs"]= tokenizer.get_token_freqs(training_parser.get_captions_from_tmp_and_lbl())
    #loss_function = WeightedSequenceCrossEntropy(kwargs=loss_kwargs)
    caption_loss_function = getattr(ptcap.losses, caption_loss_type)(kwargs=caption_loss_kwargs)
    classif_loss_function = getattr(ptcap.losses, classif_loss_type)()

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, optimizer_type)(
        params=params, **config_obj.get("optimizer", "kwargs"))

    scheduler_kwargs = copy.deepcopy(config_obj.get("scheduler", "kwargs"))
    scheduler_kwargs["optimizer"] = optimizer
    scheduler_kwargs["mode"] = "max" if higher_is_better else "min"

    scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
        **scheduler_kwargs)

    writer = Seq2seqAdapter(os.path.join(checkpoint_folder, "runs"),
                            config_obj.get("logging", "tensorboard_frequency"))
    # Prepare checkpoint directory and save config
    Checkpointer.save_meta(checkpoint_folder, config_obj, tokenizer)

    checkpointer = Checkpointer(checkpoint_folder, higher_is_better)

    # Setup the logger
    logger = CustomLogger(folder=checkpoint_folder, tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(model, caption_loss_function, w_caption_loss, scheduler, tokenizer, logger,
                      writer, checkpointer, folder=pretrained_folder,
                      filename=pretrained_file, gpus=gpus, clip_grad=clip_grad,
                      classif_loss_function=classif_loss_function, 
                      w_classif_loss=w_classif_loss)

    # Train the Model
    valid_captions, valid_preds = trainer.train(
        dataloader, val_dataloader, criteria, num_epoch, frequency_valid,
        teacher_force_train, teacher_force_valid, verbose_train, verbose_valid)

    return valid_captions, valid_preds, tokenizer
