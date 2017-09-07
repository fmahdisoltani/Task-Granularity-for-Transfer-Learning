import torch

import ptcap.loggers as lg
import ptcap.printers as prt

from collections import namedtuple
from collections import OrderedDict

from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer
from ptcap.scores import (first_token_accuracy, loss_to_numpy, ScoresOperator,
                          token_accuracy)
from ptcap.loggers import CustomLogger


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, tokenizer,
                 checkpoint_path, folder=None, filename=None, gpus=None):

        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.checkpointer = Checkpointer(checkpoint_path)

        init_state = self.checkpointer.load_model(model, optimizer,
                                                  folder, filename)

        self.num_epochs, self.model, self.optimizer = init_state
        self.model = self.model.cuda(gpus[0]) if self.use_cuda else self.model
        self.loss_function = (loss_function.cuda(gpus[0])
                              if self.use_cuda else loss_function)

        self.logger = CustomLogger(folder=checkpoint_path)
        self.tokenizer = tokenizer


    def train(self, train_dataloader, valid_dataloader, num_epoch,
              frequency_valid, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        valid_avg_loss = None

        for epoch in range(num_epoch):
            self.num_epochs += 1
            train_average_scores = self.run_epoch(train_dataloader,
                                                  epoch, is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=verbose_train)

            train_avg_loss = train_average_scores["average_loss"]

            state_dict = self.get_trainer_state(valid_avg_loss)

            self.checkpointer.save_latest(state_dict, train_avg_loss)
            self.checkpointer.save_value_csv((epoch, train_avg_loss),
                                             filename="train_loss")

            # Validation
            if (epoch + 1) % frequency_valid == 0:
                valid_average_scores = self.run_epoch(
                    valid_dataloader, epoch, is_training=False,
                    use_teacher_forcing=teacher_force_valid,
                    verbose=verbose_valid
                )

                # remember best loss and save checkpoint
                valid_avg_loss = valid_average_scores["average_loss"]

                state_dict = self.get_trainer_state(valid_avg_loss)

                self.checkpointer.save_best(state_dict, valid_avg_loss)
                self.checkpointer.save_value_csv([epoch, valid_avg_loss],
                                                 filename="valid_loss")

    def get_trainer_state(self, score):
        return {
            'epoch': self.num_epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'score': score,
        }

    def get_function_dict(self):
        function_dict = OrderedDict()
        function_dict["loss"] = loss_to_numpy

        function_dict["accuracy"] = token_accuracy

        function_dict["first_accuracy"] = first_token_accuracy
        return function_dict

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):
      
        ScoreAttr = namedtuple("ScoresAttr", "loss captions predictions")
        scores = ScoresOperator(self.get_function_dict())

        for sample_counter, (videos, _, captions) in enumerate(dataloader):

            videos, captions = Variable(videos), Variable(captions)
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)

            captions = captions.cpu()
            predictions = predictions.cpu()

            batch_outputs = ScoreAttr(loss, captions, predictions)

            scores_dict = scores.compute_scores(batch_outputs,
                                                sample_counter + 1)

            # Print after each batch
            prt.print_stuff(scores_dict, self.tokenizer,
                            is_training, captions, predictions, epoch + 1,
                            sample_counter + 1, len(dataloader), verbose)

        # Log at the end of epoch
        self.logger.log_stuff(scores_dict, self.tokenizer, is_training, captions,
                     predictions, epoch + 1, len(dataloader),
                     verbose, sample_counter)

        # Take only the average of the scores in scores_dict
        average_scores_dict = scores.get_average_scores()
        return average_scores_dict
