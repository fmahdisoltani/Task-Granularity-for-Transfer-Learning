"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from docopt import docopt
from torch.autograd import Variable
from torchvision.transforms import Compose

from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import JpegVideoDataset
from ptcap.data.config_parser import ConfigParser
from ptcap.data.annotation_parser import JsonParser
from ptcap.model.captioners import RtorchnCaptioner
from ptcap.losses import SequenceCrossEntropy
import ptcap.data.preprocessing as prep


if __name__ == '__main__':

    # Get argument
    args = docopt(__doc__)

    #Build a dictionary that contains fields of config file
    config_obj = ConfigParser(args['<config_path>'])

    #Find paths to training, validation and test sets
    training_path = config_obj.config_dict['paths']['train_annot']

    # Load Json annotation files
    training_parser = JsonParser(training_path,
                                 config_obj.config_dict['paths']['videos_folder'])

    #Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(training_parser.get_captions())

    preprocesser = Compose([prep.RandomCrop([24, 96, 96]),
                            prep.PadVideo([24, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    training_set = JpegVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer,
                                    preprocess=preprocesser)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=True,
                            **config_obj.config_dict['dataloaders']['kwargs'])

    # vocab_size, batchnorm=True, stateful=False, **kwargs
    captioner = RtorchnCaptioner(tokenizer.get_vocab_size(), is_training=True,
                                 use_cuda=config_obj.config_dict['device']
                                 ['use_cuda'])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(captioner.parameters())

    optimizer = torch.optim.Adam(params,
                                 lr=config_obj.config_dict['training']
                                 ['learning_rate'])

    # Train the Models
    total_step = len(dataloader)

    for epoch in range(config_obj.config_dict['training']['num_epochs']):
        print("Epoch {}:".format(epoch+1))
        pass
