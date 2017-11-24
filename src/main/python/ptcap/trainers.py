from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.metrics import MultiScorer
from pycocoevalcap.rouge.rouge import Rouge
from torch.autograd import Variable

from ptcap.scores import (LCS, MultiScoreAdapter, ScoresOperator,
                          caption_accuracy, classif_accuracy,
                          first_token_accuracy, fscore, gmeasure, loss_to_numpy,
                          token_accuracy)
from ptcap.utils import DataParallelWrapper


class Trainer(object):
    def __init__(self, model, loss_function, scheduler, tokenizer, logger,
                 writer, checkpointer, folder=None, filename=None,
                 gpus=None, clip_grad=None, classif_loss_function=None, do_classif=False):

        self.gpus = gpus
        self.checkpointer = checkpointer

        self.model = model if self.gpus is None else (
            DataParallelWrapper(model, device_ids=self.gpus).cuda(
                self.gpus[0]))
        self.loss_function = loss_function if self.gpus is None else (
            loss_function.cuda(self.gpus[0]))

        self.classif_loss_function = classif_loss_function if self.gpus is None else (
            classif_loss_function.cuda(self.gpus[0]))

        self.do_classif = do_classif


        init_state = self.checkpointer.load_model(self.model,
                                                  scheduler.optimizer,
                                                  folder, filename)

        self.num_epochs, self.model, scheduler.optimizer = init_state

        self.clip_grad = clip_grad
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.score = self.scheduler.best
        self.writer = writer

        self.logger = logger

        self.multiscore_adapter = MultiScoreAdapter(
            MultiScorer(BLEU=Bleu(4), ROUGE_L=Rouge(), METEOR=Meteor()),
            self.tokenizer)

        self.logger.on_train_init(folder, filename)

    def train(self, train_dataloader, valid_dataloader, criteria,
              max_num_epochs=None, frequency_valid=1, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False, ):

        self.logger.on_train_begin()

        epoch = 0
        stop_training = False

        while not stop_training:

            # we need a first round of evaluation without any training
            if epoch % frequency_valid == 0:
                valid_average_scores = self.run_epoch(
                    valid_dataloader, epoch, is_training=False,
                    use_teacher_forcing=teacher_force_valid,
                    verbose=verbose_valid
                )

                # remember best loss and save checkpoint
                self.score = valid_average_scores["avg_" + criteria]

                self.scheduler.step(self.score)

                state_dict = self.get_trainer_state()
                self.checkpointer.save_best(state_dict)
                self.checkpointer.save_value_csv([epoch, self.score],
                                                 filename="valid_loss")

            self.num_epochs += 1
            train_average_scores = self.run_epoch(train_dataloader,
                                                  epoch, is_training=True,
                                                  use_teacher_forcing=teacher_force_train,
                                                  verbose=verbose_train)

            train_avg_loss = train_average_scores["avg_loss"]

            state_dict = self.get_trainer_state()

            self.checkpointer.save_latest(state_dict)
            self.checkpointer.save_value_csv([epoch, train_avg_loss],
                                             filename="train_loss")

            epoch += 1
            stop_training = self.update_stopping_criteria(epoch, max_num_epochs)

        self.logger.on_train_end(self.scheduler.best)

    def get_trainer_state(self):
        return {
            "epoch": self.num_epochs,
            "model": self.model.state_dict(),
            "optimizer": self.scheduler.optimizer.state_dict(),
            "score": self.score,
        }

    def update_stopping_criteria(self, epoch, num_epoch):
        current_lr = max([param_group['lr'] for param_group in
                          self.scheduler.optimizer.param_groups])
        self.writer.add_scalars({"learning rate": np.log10(current_lr)},
                                global_step=epoch, is_training=True)
        # Assuming all parameters have the same minimum learning rate
        min_lr = max([lr for lr in self.scheduler.min_lrs])
        eps = 0.01 * min_lr

        # Check if the maximum number of epochs has been reached
        if num_epoch is not None and epoch >= num_epoch:
            self.logger.log_message("Maximum number of epochs reached {}/{}",
                                    (epoch, num_epoch))
            return True

        elif current_lr <= (min_lr + eps):
            self.logger.log_message("Learning rate is equal to the minimum "
                                    "learning rate ({:.4})", (min_lr,))
            return True

        return False

    def get_scoring_functions(self):

        scoring_functions = []

        scoring_functions.append(loss_to_numpy)
        scoring_functions.append(token_accuracy)
        scoring_functions.append(first_token_accuracy)
        scoring_functions.append(caption_accuracy)
        if self.do_classif:
            scoring_functions.append(classif_accuracy)
        scoring_functions.append(self.multiscore_adapter)
        scoring_functions.append(LCS([fscore, gmeasure], self.tokenizer))

        return scoring_functions

    def get_input_captions(self, captions, use_teacher_forcing):
        batch_size = captions.size(0)
        input_captions = torch.LongTensor(batch_size, 1).zero_()
        if use_teacher_forcing:
            input_captions = torch.cat([input_captions, captions[:,:-1]], 1)
        return input_captions

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):
  
        # Log at the beginning of epoch
        self.logger.on_epoch_begin(epoch + 1)

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        ScoreAttr = namedtuple("ScoresAttr", "loss string_captions captions "
                                             "predictions")
        if self.do_classif:
            ScoreAttr = namedtuple("ScoresAttr", "loss string_captions captions "
                               "predictions classif_targets classif_probs")

        scores = ScoresOperator(self.get_scoring_functions())

        for sample_counter, item in enumerate(dataloader):

            if self.do_classif:
                (videos, string_captions,
                 captions, classif_targets) = item
                classif_targets = Variable(classif_targets)

            else:
                (videos, string_captions,
                 captions) = item

            self.logger.on_batch_begin()
            input_captions = self.get_input_captions(captions,
                                                     use_teacher_forcing)

            videos, captions, input_captions = (Variable(videos),
                                                Variable(captions),
                                                Variable(input_captions)
                                               )
            if self.gpus:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0], async=True)
                input_captions = input_captions.cuda(self.gpus[0])
                if self.do_classif:
                    classif_targets = classif_targets.cuda(self.gpus[0])

            model_output = self.model((videos, input_captions), use_teacher_forcing)

            if self.do_classif:
                probs, classif_probs = model_output

            else:
                probs = model_output


            captioning_loss = self.loss_function(probs, captions)

            loss = captioning_loss

            if self.do_classif:
                classif_loss = self.classif_loss_function(classif_probs,
                                                          classif_targets)
                loss = captioning_loss + classif_loss



            global_step = len(dataloader) * epoch + sample_counter

            if is_training:

                self.model.zero_grad()
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                                  self.clip_grad)

                self.writer.add_activations(self.model, global_step)
                self.writer.add_state_dict(self.model, global_step)
                self.writer.add_gradients(self.model, global_step)

                self.scheduler.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)

            captions = captions.cpu()
            predictions = predictions.cpu()

            batch_outputs = ScoreAttr(loss, string_captions, captions,
                                      predictions)


            if self.do_classif:
                classif_targets = classif_targets.cpu()
                classif_probs = classif_probs.cpu()

                batch_outputs = ScoreAttr(loss, string_captions, captions,
                                      predictions, classif_targets, classif_probs)

            scores_dict = scores.compute_scores(batch_outputs,
                                                sample_counter + 1)

            # Take only the average of the scores in scores_dict
            average_scores_dict = scores.get_average_scores()

            self.logger.on_batch_end(
                average_scores_dict, captions,
                predictions, is_training, len(dataloader),
                verbose=verbose)

        self.logger.on_epoch_end(average_scores_dict, is_training,
                                 total_samples=len(dataloader))

        # Display average scores on tensorboard
        self.writer.add_scalars(average_scores_dict, epoch, is_training)

        return average_scores_dict


class DataParallelWrapper(nn.DataParallel):
    def __init__(self, model, device_ids):
        super().__init__(model, device_ids=device_ids)

    @property
    def activations(self):
        return self.module.activations
