import copy
import os

import torch.optim

import ptcap.data.preprocessing as prep
import ptcap.losses
import ptcap.model.captioners
import ptcap.model.decoders as dec
import ptcap.model as all_models

from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ptcap.checkpointers import Checkpointer
from ptcap.data.annotation_parser import JsonParser
from ptcap.data.dataset import (JpegVideoDataset, GulpVideoDataset)
from ptcap.data.tokenizer import Tokenizer
from ptcap.loggers import CustomLogger
from ptcap.tensorboardY import Seq2seqAdapter
from ptcap.trainers import Trainer
from rtorchn.data.preprocessing import CenterCropper


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

    # Get model, loss, optimizer, scheduler, and criteria from config_file
    model_type = config_obj.get("model", "type")
    loss_type = config_obj.get("loss", "type")
    optimizer_type = config_obj.get("optimizer", "type")
    scheduler_type = config_obj.get("scheduler", "type")
    criteria = config_obj.get("criteria", "score")
    videos_folder = config_obj.get("paths", "videos_folder")

    # Load Json annotation files
    training_parser = JsonParser(training_path, os.path.join(relative_path,
                                 videos_folder), caption_type=caption_type)
    validation_parser = JsonParser(validation_path, os.path.join(relative_path,
                                   videos_folder), caption_type=caption_type)

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(**config_obj.get("tokenizer", "kwargs"))
    if pretrained_folder:
        tokenizer.load_dictionaries(pretrained_folder)
        print("Inside pretrained" , tokenizer.get_vocab_size())
    else:
        tokenizer.build_dictionaries(training_parser.get_captions_from_tmp_and_lbl())


        #tokenizer.build_dictionaries(training_parser.get_captions())
    preprocessor = Compose([prep.RandomCrop([48, 96, 96]),
                            prep.PadVideo([48, 96, 96]),
                            prep.Float32Converter(64.),
                            prep.PytorchTransposer()])

    val_preprocessor = Compose([CenterCropper([48, 96, 96]),
                                prep.PadVideo([48, 96, 96]),
                                prep.Float32Converter(64.),
                                prep.PytorchTransposer()])

    training_set = GulpVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer,
                                    preprocess=preprocessor,
                                    gulp_dir=videos_folder,
                                    size=[128, 128])

    validation_set = GulpVideoDataset(annotation_parser=validation_parser,
                                      tokenizer=tokenizer,
                                      preprocess=val_preprocessor,
                                      gulp_dir=videos_folder,
                                      size=[128, 128])

    dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
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
    decoder_kwargs["num_step"] = tokenizer.maxlen

    # Create model, loss, and optimizer objects
    model = getattr(ptcap.model.captioners, model_type)(
        encoder=getattr(all_models, encoder_type),
        decoder=getattr(all_models, decoder_type),
        encoder_args=encoder_args,
        encoder_kwargs=encoder_kwargs,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs)

    loss_function = getattr(ptcap.losses, loss_type)()

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
    trainer = Trainer(model, loss_function, scheduler, tokenizer, logger,
                      writer, checkpointer, folder=pretrained_folder,
                      filename=pretrained_file, gpus=gpus, clip_grad=clip_grad)

    # Train the Model
    trainer.train(dataloader, val_dataloader, criteria, num_epoch,
                  frequency_valid, teacher_force_train, teacher_force_valid,
                  verbose_train, verbose_valid)
