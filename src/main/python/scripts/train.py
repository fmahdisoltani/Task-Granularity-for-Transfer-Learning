"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""


from docopt import docopt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import ptcap.data.preprocessing as prep

from ptcap.checkpointers import Checkpointer
from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import (JpegVideoDataset, NumpyVideoDataset)
from ptcap.data.config_parser import YamlConfig
from ptcap.data.annotation_parser import JsonParser
from ptcap.losses import SequenceCrossEntropy
from ptcap.model.captioners import *
from ptcap.trainers import Trainer
from rtorchn.preprocessing import CenterCropper

if __name__ == '__main__':
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args['<config_path>'])

    # Find paths to training, validation and test sets
    training_path = config_obj.get('paths', 'train_annot')
    validation_path = config_obj.get('paths', 'validation_annot')

    # Load attributes of config file
    num_epoch = config_obj.get('training', 'num_epochs')
    frequency_valid = config_obj.get('validation', 'frequency')
    verbose_train = config_obj.get('training', 'verbose')
    verbose_valid = config_obj.get('validation', 'verbose')
    teacher_force_train = config_obj.get('training', 'teacher_force')
    teacher_force_valid = config_obj.get('validation', 'teacher_force')
    # use_cuda = config_obj.get('device', 'use_cuda')
    gpus = config_obj.get("device", "gpus")
    checkpoint_folder = config_obj.get('paths', 'checkpoint_folder')
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

    preprocesser = Compose([prep.RandomCrop([48, 96, 96]),
                            prep.PadVideo([48, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    val_preprocesser = Compose([CenterCropper([48, 96, 96]),
                                prep.PadVideo([48, 96, 96]),
                                prep.Float32Converter(),
                                prep.PytorchTransposer()])

    training_set = NumpyVideoDataset(annotation_parser=training_parser,
                                     tokenizer=tokenizer,
                                     preprocess=preprocesser)

    validation_set = NumpyVideoDataset(annotation_parser=validation_parser,
                                       tokenizer=tokenizer,
                                       preprocess=val_preprocesser)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
                            **config_obj.get('dataloaders', 'kwargs'))

    val_dataloader = DataLoader(validation_set, shuffle=True, drop_last=False,
                                **config_obj.get('dataloaders', 'kwargs'))


    # captioner = CNN3dLSTM(vocab_size=tokenizer.get_vocab_size(),
    #                       go_token=tokenizer.encode_token(tokenizer.GO),
    #                       gpus=gpus)
    # captioner = RtorchnCaptioner(tokenizer.get_vocab_size())

    captioner =  RtorchnCaptionerP(vocab_size=tokenizer.get_vocab_size(),
                                   go_token=tokenizer.encode_token(tokenizer.GO),
                                   gpus=gpus)


    # Loss and Optimizer
    loss_function = SequenceCrossEntropy()

    params = filter(lambda p: p.requires_grad, captioner.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=config_obj.get('training', 'learning_rate'))

    # Prepare checkpoint directory and save config
    Checkpointer.save_meta(checkpoint_folder, config_obj, tokenizer)

    # Trainer
    pretrained_folder = config_obj.get("paths", "pretrained_path")
    trainer = Trainer(captioner, loss_function, optimizer, tokenizer,
                      checkpoint_folder, folder=pretrained_folder,
                      filename="model.best", gpus=gpus)

    # Train the Model
    trainer.train(dataloader, val_dataloader, num_epoch, frequency_valid,
                  teacher_force_train, teacher_force_valid, verbose_train,
                  verbose_valid)
