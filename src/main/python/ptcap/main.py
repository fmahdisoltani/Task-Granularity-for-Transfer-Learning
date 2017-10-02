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
from ptcap.data.annotation_parser import JsonParser
from ptcap.data.dataset import (JpegVideoDataset, NumpyVideoDataset)
from ptcap.data.tokenizer import Tokenizer
from ptcap.loggers import CustomLogger
from ptcap.tensorboardY import Seq2seqAdapter
from ptcap.trainers import Trainer
from rtorchn.data.preprocessing import CenterCropper


def train_model(config_obj, relative_path=""):
    # Find paths to training, validation and test sets
    training_path = os.path.join(relative_path,
                                 config_obj.get('paths', 'train_annot'))
    validation_path = os.path.join(relative_path,
                                   config_obj.get('paths', 'validation_annot'))

    # Load attributes of config file
    checkpoint_folder = os.path.join(
        relative_path, config_obj.get('paths', 'checkpoint_folder'))
    frequency_valid = config_obj.get('validation', 'frequency')
    gpus = config_obj.get("device", "gpus")
    num_epoch = config_obj.get('training', 'num_epochs')
    pretrained_folder = config_obj.get('pretrained', 'pretrained_folder')
    pretrained_file = config_obj.get('pretrained', 'pretrained_file')
    pretrained_folder = os.path.join(relative_path, pretrained_folder
                                   ) if pretrained_folder else None
    teacher_force_train = config_obj.get('training', 'teacher_force')
    teacher_force_valid = config_obj.get('validation', 'teacher_force')
    verbose_train = config_obj.get('training', 'verbose')
    verbose_valid = config_obj.get('validation', 'verbose')

    # Get model, loss, and optimizer types from config_file
    model_type = config_obj.get("model", "type")
    loss_type = config_obj.get("loss", "type")
    optimizer_type = config_obj.get("optimizer", "type")

    # Load Json annotation files
    training_parser = JsonParser(training_path, os.path.join(relative_path,
                                 config_obj.get('paths', 'videos_folder')))
    validation_parser = JsonParser(validation_path, os.path.join(relative_path,
                                   config_obj.get('paths', 'videos_folder')))

    # Build a tokenizer that contains all captions from annotation files

    # tokenizer = Tokenizer(user_maxlen=config_obj.get("targets", "user_maxlen"))

    tokenizer = Tokenizer(**config_obj.get("tokenizer", "kwargs"))
    if pretrained_folder:
        tokenizer.load_dictionaries(pretrained_folder)
    else:
        tokenizer.build_dictionaries(training_parser.get_captions())

    preprocessor = Compose([prep.RandomCrop([24, 96, 96]),
                            prep.PadVideo([24, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    val_preprocessor = Compose([CenterCropper([24, 96, 96]),
                                prep.PadVideo([24, 96, 96]),
                                prep.Float32Converter(),
                                prep.PytorchTransposer()])

    training_set = NumpyVideoDataset(annotation_parser=training_parser,
                                     tokenizer=tokenizer,
                                     preprocess=preprocessor)

    validation_set = NumpyVideoDataset(annotation_parser=validation_parser,
                                       tokenizer=tokenizer,
                                       preprocess=val_preprocessor)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
                            **config_obj.get("dataloaders", "kwargs"))

    val_dataloader = DataLoader(validation_set, shuffle=True, drop_last=False,
                                **config_obj.get("dataloaders", "kwargs"))

    encoder_type = config_obj.get("model", "encoder")
    print("*"*100)
    print(encoder_type)

    decoder_type = config_obj.get("model", "decoder")
    encoder_kwargs = config_obj.get("model", "encoder_kwargs")
    decoder_kwargs = config_obj.get("model", "decoder_kwargs")
    decoder_kwargs["vocab_size"] = tokenizer.get_vocab_size()
    decoder_kwargs["go_token"] = tokenizer.encode_token(tokenizer.GO)
    decoder_kwargs["hidden_size"] = encoder_kwargs["encoder_output_size"]


    print(encoder_kwargs)

    print(decoder_kwargs)

    # TODO: Remove GPUs?
    gpus = config_obj.get("device", "gpus")

    decoder_kwargs["gpus"] = gpus

    # Create model, loss, and optimizer objects
    model = getattr(ptcap.model.captioners, model_type)(
        encoder=getattr(all_models, encoder_type),
        decoder=getattr(all_models, decoder_type),
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        gpus=gpus)

    loss_function = getattr(ptcap.losses, loss_type)()

    optimizer = getattr(torch.optim, optimizer_type)(
        params=list(model.parameters()), **config_obj.get("optimizer", "kwargs"))

    writer = Seq2seqAdapter(os.path.join(checkpoint_folder, "runs"))
    # Prepare checkpoint directory and save config
    Checkpointer.save_meta(checkpoint_folder, config_obj, tokenizer)

    # Setup the logger
    logger = CustomLogger(folder=checkpoint_folder,
                          verbose=config_obj.get("logging", "verbose"))

    # Trainer
    trainer = Trainer(model, loss_function, optimizer, tokenizer, logger,
                      writer, checkpoint_folder, folder=pretrained_folder,
                      filename=pretrained_file, gpus=gpus)

    # Train the Model
    trainer.train(dataloader, val_dataloader, num_epoch, frequency_valid,
                  teacher_force_train, teacher_force_valid, verbose_train,
                  verbose_valid)
