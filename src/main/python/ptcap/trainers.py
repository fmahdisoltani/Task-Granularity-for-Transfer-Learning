import torch

import ptcap.printers as prt

from collections import OrderedDict

from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer
from ptcap.metrics import token_level_accuracy


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, tokenizer, config_obj,
                 use_cuda=False):

        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.model = model.cuda() if use_cuda else model
        self.initial_model = self.model
        self.loss_function = loss_function.cuda() if use_cuda else loss_function
        self.config_obj = config_obj
        self.use_cuda = use_cuda

    def train(self, train_dataloader, valid_dataloader, num_epoch,
              frequency_valid, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        pretrained_model = self.config_obj.get("paths", "pretrained_model")
        checkpointer = Checkpointer()
        init_epoch, model, optimizer = \
            checkpointer.init_model(pretrained_model, self.model,
                                    self.optimizer)

        checkpointer.save_meta(self.config_obj, self.tokenizer)
        for epoch in range(init_epoch, num_epoch):
            self.run_epoch(train_dataloader, epoch, is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=verbose_train)

            if (epoch + 1) % frequency_valid == 0:
                average_loss = self.run_epoch(valid_dataloader, epoch, is_training=False,
                               use_teacher_forcing=teacher_force_valid,
                               verbose=verbose_valid)

                state_dict = {
                    'epoch': epoch + 1,
                    'model': self.model.state_dict(),
                    'best_loss': checkpointer.best_loss,
                    'optimizer': self.optimizer.state_dict(),
                    }
                # remember best loss and save checkpoint
                checkpointer.save_model(state_dict, average_loss,
                                        is_higher_better=False)

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):

        average_loss = 0.
        count = 0
        scores_dict = OrderedDict()

        for sample_counter, (videos, _, captions) in enumerate(dataloader):

            videos, captions = Variable(videos), Variable(captions)
            if self.use_cuda:
                videos = videos.cuda()
                captions = captions.cuda()
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            count += 1

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)
            predictions = torch.squeeze(predictions)

            captions = captions.cpu()
            predictions = predictions.cpu()

            scores_dict["loss"] = loss.data.cpu().numpy()
            scores_dict["accuracy"] = token_level_accuracy(captions,
                                                           predictions)
            scores_dict["first_accuracy"] = token_level_accuracy(captions,
                                                                 predictions, 1)
            # Calculate a moving average of the metrics
            self.moving_average(scores_dict, count)

            prt.print_stuff(average_loss, self.tokenizer, is_training, captions,
                            predictions, epoch, sample_counter, len(dataloader),
                            verbose)

        return average_loss

    def moving_average(self, scores_dict, count):
        for metric in scores_dict:
            average_metric = "average_" + metric
            if average_metric in scores_dict:
                scores_dict[average_metric] += \
                    (scores_dict[metric] - scores_dict[average_metric])/count
            else:
                scores_dict[average_metric] = scores_dict[metric]
