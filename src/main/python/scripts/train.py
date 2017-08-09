"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

from torch.utils.data import DataLoader
from docopt import docopt
from torchvision.transforms import Compose


from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import JpegVideoDataset
from ptcap.data.config_parser import YamlConfig
from ptcap.data.annotation_parser import JsonParser
from ptcap.model.captioners import *
from ptcap.losses import SequenceCrossEntropy
from ptcap.trainers import Trainer
import ptcap.data.preprocessing as prep


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

    preprocesser = Compose([prep.RandomCrop([24, 96, 96]),
                            prep.PadVideo([24, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    training_set = JpegVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer,
                                    preprocess=preprocesser)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=True,
                            **config_obj.get('dataloaders', 'kwargs'))

    # vocab_size, batchnorm=True, stateful=False, **kwargs
    rcaptioner = RtorchnCaptioner(tokenizer.get_vocab_size(), is_training=True,
                                use_cuda=config_obj.get('device', 'use_cuda'))

    captioner = CNN3dLSTM(vocab_size=tokenizer.get_vocab_size())
    # Loss and Optimizer
    loss_function = SequenceCrossEntropy()
    params = list(captioner.parameters())

    optimizer = torch.optim.Adam(params,
                                 lr=config_obj.get('training', 'learning_rate'))

    # Train the Model
    num_epoch = config_obj.get('training', 'num_epochs')
    frequency_valid = config_obj.get('validation', 'frequency')
    verbose_train = config_obj.get('training', 'verbose')
    verbose_valid = config_obj.get('validation', 'verbose')
    teacher_forcing_train = config_obj.get('training', 'teacher_force')
    teacher_forcing_valid = config_obj.get('validation', 'teacher_force')

    trainer = Trainer(captioner, loss_function, optimizer, num_epoch,
                      frequency_valid, tokenizer, verbose_train=verbose_train,
                      verbose_valid=verbose_valid)

    trainer.train(dataloader, dataloader, teacher_forcing_train,
                  teacher_forcing_valid)
