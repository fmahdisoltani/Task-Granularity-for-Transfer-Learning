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

from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import JpegVideoDataset
from ptcap.data.config_parser import ConfigParser
from ptcap.data.annotation_parser import JsonParser


if __name__ == '__main__':

    # Get argument
    args = docopt(__doc__)

    #Build a dictionary that contains fields of config file
    config_obj = ConfigParser(args['<config_path>'])

    #Find paths to training, validation and test sets
    training_path = config_obj.config_dict['paths']['train_annot']

    # Load Json annotation files
    training_annot = JsonParser(training_path,
                            config_obj.config_dict['paths']['videos_folder'])

    #Build a tokenizer that contains all captions from annotation files
    training_captions = training_annot.get_captions()
    tokenizer_obj = Tokenizer(training_captions)

    training_set = JpegVideoDataset(annotation_obj=training_annot,
                                    tokenizer_obj=tokenizer_obj)

    dataloader = DataLoader(training_set, shuffle=True,
                            **config_obj.config_dict['dataloaders']['kwargs'])

    for it, sample_batch in enumerate(dataloader):
        video, string_caption, tokenized_caption = sample_batch
