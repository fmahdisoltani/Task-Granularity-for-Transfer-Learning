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

from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import JpegVideoDataset
from ptcap.data.config_parser import ConfigParser
from ptcap.data.annotation_parser import JsonParser
from ptcap.model.captioners import RtorchnCaptioner


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

    training_set = JpegVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer)

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
        for it, sample_batch in enumerate(dataloader):
            videos, string_captions, tokenized_captions = sample_batch
            captioner.zero_grad()

            outputs = captioner((Variable(videos), Variable(tokenized_captions)))

            # prepare targets
            batch_size, timesteps, vocab_size = outputs.size()
            end_tensor = torch.LongTensor(
                [tokenizer.caption_dict[tokenizer.END]]*batch_size)
            end_tensor = end_tensor.view(-1,1)
            first_part = tokenized_captions[:,1:]
            targets = Variable(torch.cat([first_part, end_tensor], 1))

            # compute loss
            loss = 0
            for t in range(timesteps):
                loss += criterion(outputs[:,t], targets[:,t])
            loss /= (timesteps * batch_size)
            print("loss at iteration {}: {}".format(it+1, loss.data))

            # compute accuracy
            max_probs, output_tokens = torch.max(outputs,2)
            #output_tokens = output_tokens.view(-1)
            #targets = targets.view(-1)
            equal_values = targets.eq(output_tokens).sum().float()
            accuracy = equal_values*100.0/(batch_size*timesteps)
            print("accuracy is: {}".format(accuracy))

            for caption_index in range(batch_size):
                print(string_captions[caption_index])
                model_generations = output_tokens[caption_index].view(-1)\
                model_generations_as_numpy = model_generations.data.numpy()
                print(tokenizer.get_string(model_generations_as_numpy))

            # optimize
            loss.backward()
            optimizer.step()
