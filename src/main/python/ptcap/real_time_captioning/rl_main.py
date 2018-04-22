"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""
import torch
import os
import copy

from docopt import docopt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


import ptcap.model.captioners
import ptcap.data.preprocessing as prep
import ptcap.model as all_models
from ptcap.data.dataset import  GulpVideoDataset
from ptcap.data.annotation_parser import JsonParser, V2Parser
from ptcap.data.config_parser import YamlConfig
from ptcap.main import train_model
from ptcap.data.tokenizer import Tokenizer
from rtorchn.data.preprocessing import CenterCropper
from ptcap.real_time_captioning.rl_trainers import RLTrainer
from ptcap.checkpointers import Checkpointer


def train_model(config_obj, relative_path=""):

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

if __name__ == "__main__":
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args["<config_path>"])

    relative_path = ""
    training_path = os.path.join(relative_path,
                                 config_obj.get("paths", "train_annot"))
    validation_path = os.path.join(relative_path,
                                   config_obj.get("paths", "validation_annot"))

    test_path = os.path.join(relative_path,
                             config_obj.get("paths", "test_annot"))

    # Load attributes of config file
    caption_type = config_obj.get("targets", "caption_type")
    pretrained_encoder = config_obj.get("pretrained", "pretrained_encoder")
    pretrained_decoder = config_obj.get("pretrained", "pretrained_decoder")
    pretrained_file = config_obj.get("pretrained", "pretrained_file")
    pretrained_folder = os.path.join(relative_path, pretrained_encoder
                                     ) if pretrained_encoder else None
    encoder_type = "C3dLSTMEncoder"
    decoder_type = "CoupledLSTMDecoder"
    caption_loss_type = config_obj.get("loss", "caption_loss")
    balanced_loss = config_obj.get("loss", "balanced")
    w_caption_loss = config_obj.get("loss", "w_caption_loss")
    classif_loss_type = config_obj.get("loss", "classif_loss")
    w_classif_loss = config_obj.get("loss", "w_classif_loss")
    optimizer_type = config_obj.get("optimizer", "type")
    scheduler_type = config_obj.get("scheduler", "type")
    criteria = config_obj.get("criteria", "score")
    videos_folder = config_obj.get("paths", "videos_folder")
    checkpoint_folder = os.path.join(
        relative_path, config_obj.get("paths", "checkpoint_folder"))
    higher_is_better = config_obj.get("criteria", "higher_is_better")

    # Preprocess
    crop_size = config_obj.get("preprocess", "crop_size")
    scale = config_obj.get("preprocess", "scale")
    input_resize = config_obj.get("preprocess", "input_resize")

    training_parser = V2Parser(training_path, os.path.join(relative_path,
                                                           videos_folder),
                               caption_type=caption_type)
    validation_parser = V2Parser(validation_path,
                                 os.path.join(relative_path,
                                              videos_folder),
                                 caption_type=caption_type)

    test_parser = V2Parser(test_path,
                           os.path.join(relative_path,
                                        videos_folder),
                           caption_type=caption_type)

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(**config_obj.get("tokenizer", "kwargs"))
    if pretrained_decoder:
        tokenizer.load_dictionaries(pretrained_encoder)
        print("Inside pretrained", tokenizer.get_vocab_size())
        print("pretty fucked up")
    else:
        tokenizer.build_dictionaries(
            training_parser.get_captions_from_tmp_and_lbl())

        # tokenizer.build_dictionaries(training_parser.get_captions())
    preprocessor = Compose([prep.RandomCrop(crop_size),
                            prep.PadVideo(crop_size),
                            prep.Float32Converter(scale),
                            prep.PytorchTransposer()])

    val_preprocessor = Compose([CenterCropper(crop_size),
                                prep.PadVideo(crop_size),
                                prep.Float32Converter(scale),
                                prep.PytorchTransposer()])

    training_set = GulpVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer,
                                    preprocess=preprocessor,
                                    gulp_dir=videos_folder,
                                    size=input_resize)

    validation_set = GulpVideoDataset(annotation_parser=validation_parser,
                                      tokenizer=tokenizer,
                                      preprocess=val_preprocessor,
                                      gulp_dir=videos_folder,
                                      size=input_resize)

    test_set = GulpVideoDataset(annotation_parser=test_parser,
                                tokenizer=tokenizer,
                                preprocess=val_preprocessor,
                                gulp_dir=videos_folder,
                                size=input_resize)  # TODO: This is shit, fix the shit

    train_dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
                            **config_obj.get("dataloaders", "kwargs"))

    val_dataloader = DataLoader(validation_set, shuffle=True, drop_last=False,
                                **config_obj.get("dataloaders", "kwargs"))

    test_dataloader = DataLoader(test_set, shuffle=True, drop_last=False,
                                 **config_obj.get("dataloaders", "kwargs"))

    encoder_type = config_obj.get("model", "encoder")
    decoder_type = config_obj.get("model", "decoder")
    encoder_args = config_obj.get("model", "encoder_args")
    encoder_kwargs = config_obj.get("model", "encoder_kwargs")
    decoder_args = config_obj.get("model", "decoder_args")
    decoder_kwargs = config_obj.get("model", "decoder_kwargs")
    decoder_kwargs["vocab_size"] = tokenizer.get_vocab_size()
    # decoder_kwargs["go_token"] = tokenizer.encode_token(tokenizer.GO)

    # TODO: Remove GPUs?
    gpus = config_obj.get("device", "gpus")

    checkpointer = Checkpointer(checkpoint_folder, higher_is_better)

    # Create encoder, decoder loss, and optimizer objects
    encoder_args = encoder_args or ()
    encoder_kwargs = encoder_kwargs or {}
    decoder_args = decoder_args or ()
    decoder_kwargs = decoder_kwargs or {}

    encoder = getattr(ptcap.model.encoders, encoder_type)(
        *encoder_args,
        **encoder_kwargs)

    decoder = getattr(ptcap.model.decoders, decoder_type)(
        *decoder_args,
        **decoder_kwargs)

    # loss_function = getattr(ptcap.losses, loss_type)()
    caption_loss_kwargs = {}

    # loss_function = WeightedSequenceCrossEntropy(kwargs=loss_kwargs)
    caption_loss_function = getattr(ptcap.losses, caption_loss_type)(
        kwargs=caption_loss_kwargs)
    classif_loss_function = getattr(ptcap.losses, classif_loss_type)()

    #TODO: add parameters of Encoder if needed
    params = filter(lambda p: p.requires_grad, decoder.parameters())
    optimizer = getattr(torch.optim, optimizer_type)(
        params=params, **config_obj.get("optimizer", "kwargs"))

    scheduler_kwargs = copy.deepcopy(config_obj.get("scheduler", "kwargs"))
    scheduler_kwargs["optimizer"] = optimizer
    scheduler_kwargs["mode"] = "max" if higher_is_better else "min"

    scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
        **scheduler_kwargs)

    rl_trainer = RLTrainer(encoder, decoder, caption_loss_function, tokenizer,
                           checkpointer, scheduler, folder=pretrained_folder,
                      filename=pretrained_file, gpus=None)
    rl_trainer.train(train_dataloader, teacher_force_train=True)

