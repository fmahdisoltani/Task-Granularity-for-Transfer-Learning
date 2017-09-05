import torch

import ptcap.printers as prt

from collections import namedtuple
from collections import OrderedDict

from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer
from ptcap.scores import (first_token_accuracy, loss_to_numpy, ScoresOperator,
                          token_accuracy)


class Trainer_classif(object):
    def __init__(self, model, loss_function, optimizer, tokenizer,
                 checkpoint_path, folder=None, filename=None, gpus=None):

        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.checkpointer = Checkpointer(checkpoint_path)
        init_state = self.checkpointer.load_model( model,
                                                  optimizer, tokenizer, folder, filename)

        self.num_epochs, self.model, self.optimizer, self.tokenizer = init_state
        self.model = self.model.cuda(gpus[0]) if self.use_cuda else self.model
        self.loss_function = (loss_function.cuda(gpus[0])
                              if self.use_cuda else loss_function)

    def train(self, train_dataloader, valid_dataloader, num_epoch,
              frequency_valid, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        for epoch in range(num_epoch):
            self.num_epochs += 1
            train_average_scores = self.run_epoch(train_dataloader, epoch,
                                                  is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=verbose_train)

            state_dict = self.get_trainer_state()
            train_avg_loss = train_average_scores["average_loss"]
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

                state_dict = self.get_trainer_state()

                # remember best loss and save checkpoint
                valid_average_loss = valid_average_scores["average_loss"]
                self.checkpointer.save_best(state_dict, valid_average_loss)
                self.checkpointer.save_value_csv([epoch, valid_average_loss],
                                                 filename="valid_loss")

    def get_trainer_state(self):
        return {
            'epoch': self.num_epochs,
            'model': self.model.state_dict(),
            'best_score': self.checkpointer.best_score,
            'optimizer': self.optimizer.state_dict(),
        }

    def get_function_dict(self):
        function_dict = OrderedDict()
        function_dict["loss"] = loss_to_numpy

        #function_dict["accuracy"] = token_accuracy

        #function_dict["first_accuracy"] = first_token_accuracy
        return function_dict

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):
      
        ScoreAttr = namedtuple("ScoresAttr", "loss captions predictions")
        scores = ScoresOperator(self.get_function_dict())

        for sample_counter, (videos, labels, _) in enumerate(dataloader):

            videos, labels = Variable(videos), Variable(labels)
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                labels = labels.cuda(self.gpus[0])
            probs = self.model((videos, labels), use_teacher_forcing)

            loss = self.loss_function(probs, labels)

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=1)

            labels = labels.cpu()
            predictions = predictions.cpu()

            batch_outputs = ScoreAttr(loss, labels, predictions)

            scores_dict = scores.compute_scores(batch_outputs,
                                                sample_counter + 1)

            prt.print_stuff(scores_dict, self.tokenizer,
                           is_training, labels, predictions, epoch + 1,
                           sample_counter + 1, len(dataloader), verbose)

        # Take only the average of the scores in scores_dict
        average_scores_dict = scores.get_average_scores()
        print (average_scores_dict)
        return average_scores_dict