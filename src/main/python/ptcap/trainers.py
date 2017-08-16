import torch
from torch.autograd import Variable

import ptcap.printers as prt


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, tokenizer, use_cuda=False):

        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.model = model.cuda() if use_cuda else model
        self.loss_function = loss_function.cuda() if use_cuda else loss_function
        self.use_cuda = use_cuda

    def train(self, train_dataloader, valid_dataloader, num_epoch,
              frequency_valid, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        for epoch in range(num_epoch):
            self.run_epoch(train_dataloader, epoch, is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=verbose_train)

            if (epoch + 1) % frequency_valid == 0:
                self.run_epoch(valid_dataloader, epoch, is_training=False,
                               use_teacher_forcing=teacher_force_valid,
                               verbose=verbose_valid)

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):

        for sample_counter, (videos, _, captions) in enumerate(dataloader):

            videos, captions = Variable(videos), Variable(captions)
            if self.use_cuda:
                videos = videos.cuda()
                captions = captions.cuda()
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)
            predictions = torch.squeeze(predictions)

            prt.print_stuff(loss, self.tokenizer, is_training, captions, predictions,
                            epoch, sample_counter, len(dataloader), verbose)

