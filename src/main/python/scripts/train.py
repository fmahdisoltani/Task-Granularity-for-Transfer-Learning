"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

import torch.optim

from docopt import docopt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import ptcap.data.preprocessing as prep
import ptcap.losses
import ptcap.model.captioners

from ptcap.checkpointers import Checkpointer
from ptcap.data.config_parser import YamlConfig
from ptcap.data.dataset import (JpegVideoDataset, NumpyVideoDataset)
from ptcap.data.tokenizer import Tokenizer

from ptcap.data.annotation_parser import JsonParser

from ptcap.trainers import Trainer
from rtorchn.data.preprocessing import CenterCropper

if __name__ == '__main__':
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args['<config_path>'])

    # Find paths to training, validation and test sets
    training_path = config_obj.get('paths', 'train_annot')
    validation_path = config_obj.get('paths', 'validation_annot')

    # Load attributes of config file
    pretrained_path = config_obj.get('paths', 'pretrained_path')

    # Load Json annotation files
    training_parser = JsonParser(training_path,
                                 config_obj.get('paths', 'videos_folder'))
    validation_parser = JsonParser(validation_path,
                                   config_obj.get('paths', 'videos_folder'))

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer()
    if pretrained_path:
        tokenizer.load_dictionaries(pretrained_path)
    else:
        tokenizer.build_dictionaries(training_parser.get_captions())

    preprocesser = Compose([prep.RandomCrop([24, 96, 96]),
                            prep.PadVideo([24, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    val_preprocesser = Compose([CenterCropper([24, 96, 96]),
                                prep.PadVideo([24, 96, 96]),
                                prep.Float32Converter(),
                                prep.PytorchTransposer()])

    training_set = NumpyVideoDataset(annotation_parser=training_parser,
                                     tokenizer=tokenizer,
                                     preprocess=preprocesser)

    validation_set = NumpyVideoDataset(annotation_parser=validation_parser,
                                       tokenizer=tokenizer,
                                       preprocess=val_preprocesser)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
                            **config_obj.get("dataloaders", "kwargs"))

    val_dataloader = DataLoader(validation_set, shuffle=True, drop_last=False,
                                **config_obj.get("dataloaders", "kwargs"))

    # Get model, loss, and optimizer types from config_file
    gpus = config_obj.get("device", "gpus")
    model_type = config_obj.get("model", "type")
    # Create model, loss, and optimizer objects
    model = getattr(ptcap.model.captioners, model_type)(
        vocab_size=tokenizer.get_vocab_size(),
        go_token=tokenizer.encode_token(tokenizer.GO), gpus=gpus)

    loss_type = config_obj.get("loss", "type")
    loss_function = getattr(ptcap.losses, loss_type)()

    optimizer_type = config_obj.get("optimizer", "type")
    optimizer = getattr(torch.optim, optimizer_type)(params=list(model.parameters()),
                     lr=config_obj.get("training", "learning_rate"))

    checkpoint_folder = config_obj.get('paths', 'checkpoint_folder')
    # Prepare checkpoint directory and save config
    Checkpointer.save_meta(checkpoint_folder, config_obj, tokenizer)

    # Trainer
    trainer = Trainer(model, loss_function, optimizer, tokenizer,
                      checkpoint_folder, folder=pretrained_path,
                      filename="model.best", gpus=gpus)

    num_epoch = config_obj.get('training', 'num_epochs')
    frequency_valid = config_obj.get('validation', 'frequency')
    verbose_train = config_obj.get('training', 'verbose')
    verbose_valid = config_obj.get('validation', 'verbose')
    teacher_force_train = config_obj.get('training', 'teacher_force')
    teacher_force_valid = config_obj.get('validation', 'teacher_force')
    # Train the Model
    trainer.train(dataloader, val_dataloader, num_epoch, frequency_valid,
                  teacher_force_train, teacher_force_valid, verbose_train,
                  verbose_valid)
