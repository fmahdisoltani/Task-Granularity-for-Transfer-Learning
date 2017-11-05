import time

import torch

from collections import namedtuple
from collections import OrderedDict

from torch.autograd import Variable

from ptcap.scores import (ScoresOperator, caption_accuracy,
                          first_token_accuracy, loss_to_numpy, token_accuracy)


class Trainer(object):
    def __init__(self, model, loss_function, scheduler, tokenizer, logger,
                 writer, checkpointer, folder=None, filename=None,
                 gpus=None, clip_grad=None):

        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.checkpointer = checkpointer

        init_state = self.checkpointer.load_model(model, scheduler.optimizer,
                                                  folder, filename)

        #self.num_epochs, self.model, scheduler.optimizer = init_state
        #self.model = self.model.cuda(gpus[0]) if self.use_cuda else self.model
        #self.loss_function = (loss_function.cuda(gpus[0])
        #                       if self.use_cuda else loss_function)


        self.num_epochs, self.model, scheduler.optimizer = init_state
        self.model = self.model if self.gpus is None else (
            torch.nn.parallel.DataParallel(model, device_ids=self.gpus).cuda(
                self.gpus[0]))
        self.loss_function = loss_function if self.gpus is None else (
            loss_function.cuda(self.gpus[0]))



        self.clip_grad = clip_grad
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.score = self.scheduler.best
        self.writer = writer

        self.tensorboard_frequency = 1
        self.logger = logger
        self.logger.on_train_init(folder, filename)


    def train(self, train_dataloader, valid_dataloader, criteria,
              max_num_epochs=None, frequency_valid=1, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        # start_time = time.time()

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
            stop_training = self.update_stop_training(epoch, max_num_epochs)

        self.logger.on_train_end(self.scheduler.best)

        # end_time = time.time()
        #
        # print("Total time {}".format(end_time - start_time))

    def get_trainer_state(self):
        return {
            "epoch": self.num_epochs,
            "model": self.model.state_dict(),
            "optimizer": self.scheduler.optimizer.state_dict(),
            "score": self.score,
        }

    def update_stop_training(self, epoch, num_epoch):
        current_lr = max([param_group['lr'] for param_group in
                          self.scheduler.optimizer.param_groups])
        # Assuming all parameters have the same minimum learning rate
        min_lr = max([lr for lr in self.scheduler.min_lrs])

        # Check if the maximum number of epochs has been reached
        if num_epoch is not None and epoch >= num_epoch:
            self.logger.log_message("Maximum number of epochs reached {}/{}",
                                    (epoch, num_epoch))
            return True

        elif current_lr <= min_lr:
            self.logger.log_message("Learning rate is equal to the minimum "
                                    "learning rate ({:.4})", (min_lr,))
            return True

        return False

    def get_function_dict(self):

        function_dict = OrderedDict()
        function_dict["loss"] = loss_to_numpy
        function_dict["accuracy"] = token_accuracy
        function_dict["first_accuracy"] = first_token_accuracy
        function_dict["caption_accuracy"] = caption_accuracy

        return function_dict

    
    def get_input_captions(self, captions, is_training):
       batch_size = captions.size(0)
       input_captions = torch.LongTensor(batch_size, 1).zero_()
       if is_training:
           input_captions = torch.cat([input_captions, captions[:,:-1]], 1)
       return input_captions

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):
        self.logger.on_epoch_begin(epoch)

        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        ScoreAttr = namedtuple("ScoresAttr", "loss captions predictions")
        scores = ScoresOperator(self.get_function_dict())

        for sample_counter, (videos, _, captions) in enumerate(dataloader):
            self.logger.on_batch_begin()

            
            input_captions = self.get_input_captions(captions, is_training)

            videos, captions, input_captions = (Variable(videos),
                                                Variable(captions),
                                                Variable(input_captions))
            #if self.gpus:
            #    captions = captions.cuda(self.gpus[0], async=True)
            #videos, captions = (Variable(videos),
            #                    Variable(captions))

            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                input_captions = input_captions.cuda(self.gpus[0])
            probs = self.model((videos, input_captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            global_step = len(dataloader) * epoch + sample_counter

            if is_training:

                self.model.zero_grad()
                loss.backward()
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                                  self.clip_grad)

                # self.writer.add_activations(self.model, global_step)
                # self.writer.add_state_dict(self.model, global_step)
                # self.writer.add_gradients(self.model, global_step)

                self.scheduler.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)

            captions = captions.cpu()
            predictions = predictions.cpu()

            batch_outputs = ScoreAttr(loss, captions, predictions)

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
