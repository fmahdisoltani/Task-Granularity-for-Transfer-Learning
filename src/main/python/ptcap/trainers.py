import numpy as np

import torch

from collections import namedtuple
from collections import OrderedDict

from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer
from ptcap.scores import (ScoresOperator, caption_accuracy, 
                          first_token_accuracy, loss_to_numpy, token_accuracy)


class Trainer(object):
    def __init__(self, model, loss_function, scheduler, tokenizer, logger,
                 writer, checkpoint_path, folder=None, filename=None,
                 gpus=None):

        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.checkpointer = Checkpointer(checkpoint_path)

        init_state = self.checkpointer.load_model(model, scheduler,
                                                  folder, filename)

        self.num_epochs, self.model, self.scheduler = init_state
        self.model = self.model.cuda(gpus[0]) if self.use_cuda else self.model
        self.loss_function = (loss_function.cuda(gpus[0])
                              if self.use_cuda else loss_function)

        self.logger = logger
        self.tokenizer = tokenizer
        self.score = 100
        self.writer = writer

    def train(self, train_dataloader, valid_dataloader, criteria,
              num_epoch=None, frequency_valid=1, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        epoch = 0
        train_model = True

        while train_model:
            self.num_epochs += 1
            train_average_scores = self.run_epoch(train_dataloader,
                                                  epoch, is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=verbose_train)

            train_avg_loss = train_average_scores["average_loss"]

            state_dict = self.get_trainer_state()

            self.checkpointer.save_latest(state_dict)
            self.checkpointer.save_value_csv([epoch, train_avg_loss],
                                             filename="train_loss")

            # Validation
            if (epoch + 1) % frequency_valid == 0:
                valid_average_scores = self.run_epoch(
                    valid_dataloader, epoch, is_training=False,
                    use_teacher_forcing=teacher_force_valid,
                    verbose=verbose_valid
                )

                # remember best loss and save checkpoint
                self.score = valid_average_scores["average_" + criteria]

                self.scheduler.step(self.score)

                state_dict = self.get_trainer_state()

                self.checkpointer.save_best(state_dict)
                self.checkpointer.save_value_csv([epoch, self.score],
                                                 filename="valid_loss")

            epoch += 1
            train_model = self.update_train_model(epoch, num_epoch)

        self.logger.log_train_end(self.scheduler.best)

    def get_trainer_state(self):
        return {
            'epoch': self.num_epochs,
            'model': self.model.state_dict(),
            'optimizer': self.scheduler.optimizer.state_dict(),
            'score': self.score,
        }

    def update_train_model(self, epoch, num_epoch):
        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        # Assuming all parameters have the same minimum learning rate
        min_lr = self.scheduler.min_lrs[0]

        # Check if the maximum number of epochs has been reached
        if num_epoch is not None and epoch >= num_epoch:
            self.logger.log_message("Maximum number of epochs reached {}/{}",
                                    (epoch, num_epoch))
            return False

        elif current_lr <= min_lr:
            self.logger.log_message("Learning rate is equal to the minimum "
                                    "learning rate ({:.4})", (min_lr,))
            return False

        return True

    def get_function_dict(self):
        function_dict = OrderedDict()
        function_dict["loss"] = loss_to_numpy

        function_dict["accuracy"] = token_accuracy

        function_dict["first_accuracy"] = first_token_accuracy

        function_dict["caption_accuracy"] = caption_accuracy
        return function_dict

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):

        # Log at the beginning of epoch
        self.logger.log_epoch_begin(is_training, epoch + 1)
      
        ScoreAttr = namedtuple("ScoresAttr", "loss captions predictions")
        scores = ScoresOperator(self.get_function_dict())

        for sample_counter, (videos, _, captions) in enumerate(dataloader):

            videos, captions = (Variable(videos),
                                Variable(captions))
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)

            captions = captions.cpu()
            predictions = predictions.cpu()

            batch_outputs = ScoreAttr(loss, captions, predictions)

            scores_dict = scores.compute_scores(batch_outputs,
                                                sample_counter + 1)

            global_step = len(dataloader) * epoch + sample_counter

            if is_training:
                self.writer.add_activations(self.model, global_step)
                self.writer.add_state_dict(self.model, global_step)

                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                self.scheduler.optimizer.step()

                self.writer.add_gradients(self.model, global_step)

            self.writer.add_scalars(scores.get_average_scores(), global_step,
                                    is_training)

            # Log at the end of batch
            self.logger.log_batch_end(
                scores_dict, self.tokenizer, captions, predictions, is_training,
                sample_counter + 1, len(dataloader), verbose)

        # Take only the average of the scores in scores_dict
        average_scores_dict = scores.get_average_scores()

        # Log at the end of epoch
        self.logger.log_epoch_end(average_scores_dict)

        return average_scores_dict
