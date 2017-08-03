"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

import torch
from torch.utils.data import DataLoader
from docopt import docopt

from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import JpegVideoDataset
from ptcap.data.config_parser import YamlConfig
from ptcap.data.annotation_parser import JsonParser
from ptcap.model.captioners import RtorchnCaptioner
from ptcap.losses import SequenceCrossEntropy
from ptcap.trainers import Trainer


if __name__ == '__main__':

    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args['<config_path>'])

    # Find paths to training, validation and test sets
    training_path = config_obj.get('paths', 'train_annot')

    # Load Json annotation files
    training_parser = JsonParser(training_path,
                                 config_obj.get('paths', 'videos_folder'))

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(training_parser.get_captions())

    training_set = JpegVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=True,
                            **config_obj.get('dataloaders', 'kwargs'))

    # vocab_size, batchnorm=True, stateful=False, **kwargs
    captioner = RtorchnCaptioner(tokenizer.get_vocab_size(), is_training=True,
                                 use_cuda=config_obj.get('device', 'use_cuda'))

    # Loss and Optimizer
    loss_function = SequenceCrossEntropy()
    params = list(captioner.parameters())

    optimizer = torch.optim.Adam(params,
                                 lr=config_obj.get('training', 'learning_rate'))

    # Train the Model
    num_epoch = config_obj.get('training','num_epochs')
    valid_frequency = config_obj.get('training', 'valid_frequency')
    trainer = Trainer(captioner,
                      loss_function, optimizer, num_epoch, valid_frequency)

    # trainer.train(dataloader, dataloader)


    ################################<<UGLY>>####################################
    from ptcap.model.captioners import EncoderDecoder
    encoder = EncoderDecoder()
    trainer2 = Trainer(encoder, loss_function, optimizer, num_epoch,
                       valid_frequency)
    trainer2.train(dataloader, dataloader)
