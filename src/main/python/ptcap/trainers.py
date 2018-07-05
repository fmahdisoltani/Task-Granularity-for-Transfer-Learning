import torch
import numpy as np

from collections import namedtuple
from collections import OrderedDict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.fudge.fudge import Fudge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.metrics import MultiScorer
from pycocoevalcap.rouge.rouge import Rouge
from torch.autograd import Variable

from ptcap.scores import (MultiScoreAdapter, ScoresOperator, caption_accuracy,
                          classif_accuracy, first_token_accuracy, loss_to_numpy,
                          token_accuracy)
from ptcap.utils import DataParallelWrapper


class Trainer(object):
    def __init__(self, model, caption_loss_function, w_caption_loss, scheduler, tokenizer, logger,
                 writer, checkpointer, load_encoder_only, folder=None, filename=None,
                 gpus=None, clip_grad=None, classif_loss_function=None,
                 w_classif_loss=0, compute_metrics=False):

        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.checkpointer = checkpointer
        self.load_encoder_only = load_encoder_only

        self.model = model if self.gpus is None else(
            DataParallelWrapper(model, device_ids=self.gpus).cuda(gpus[0])
        )
        self.loss_function = caption_loss_function if self.gpus is None else(
            caption_loss_function.cuda(gpus[0])
        )
        self.w_caption_loss = w_caption_loss
        self.classif_loss_function = classif_loss_function if self.gpus is None \
            else(classif_loss_function.cuda(gpus[0]))
        self.w_classif_loss = w_classif_loss

        init_state = self.checkpointer.load_model(self.model, scheduler.optimizer,
                                                  folder, filename,
                                                  load_encoder_only=load_encoder_only)

        self.num_epochs, self.model, scheduler.optimizer = init_state

        self.clip_grad = clip_grad
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.score = self.scheduler.best
        self.writer = writer

        self.tensorboard_frequency = 1
        self.logger = logger

        self.logger.on_train_init(folder, filename)
        if compute_metrics:
            self.multiscore_adapter = MultiScoreAdapter(
                MultiScorer(aggregator=Fudge(), BLEU=Bleu(4),
                            ROUGE_L=Rouge(), METEOR=Meteor()), self.tokenizer)

    def train(self, train_dataloader, valid_dataloader, criteria,
              max_num_epochs=None, frequency_valid=1, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        epoch = 0
        stop_training = False

        while not stop_training:

            # we need a first round of evaluation without any training
            if epoch % frequency_valid == 10:
                valid_average_scores, valid_captions, valid_preds = (
                    self.run_epoch(valid_dataloader, epoch, is_training=False,
                                   use_teacher_forcing=teacher_force_valid,
                                   verbose=verbose_valid))

                # remember best loss and save checkpoint
                self.score = valid_average_scores["avg_" + criteria]

                self.scheduler.step(self.score)

                state_dict = self.get_trainer_state()
                self.checkpointer.save_best(state_dict)
                self.checkpointer.save_value_csv([epoch, self.score],
                                                 filename="valid_loss")

            self.num_epochs += 1
            if max_num_epochs > 0:
                train_average_scores, train_captions, train_preds = self.run_epoch(
                    train_dataloader,
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
        return valid_captions, valid_preds

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

        self.writer.add_scalars({"learning rate": np.log10(current_lr)},
                                global_step=epoch, is_training=True)

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

    def get_function_dict(self, is_training):

        scoring_functions = []

        scoring_functions.append(loss_to_numpy)
        scoring_functions.append(token_accuracy)
        scoring_functions.append(first_token_accuracy)
        scoring_functions.append(caption_accuracy)
        scoring_functions.append(classif_accuracy)
        #if not is_training:
        #    scoring_functions.append(self.multiscore_adapter)
            #scoring_functions.append(LCS([fscore, gmeasure], self.tokenizer))

        return scoring_functions

    def get_input_captions(self, captions, use_teacher_forcing):
        batch_size = captions.size(0)
        input_captions = torch.LongTensor(batch_size, 1).zero_()
        if use_teacher_forcing:
            input_captions = torch.cat([input_captions, captions[:, :-1]], 1)
        return input_captions

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):
        self.logger.on_epoch_begin(epoch)
        predictions_list, captions_list = [], []
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        ScoreAttr = namedtuple("ScoresAttr",
                               "loss string_captions captions predictions classif_targets classif_probs")
        scores = ScoresOperator(self.get_function_dict(is_training))

        for sample_counter, (videos, string_captions, captions, classif_targets) in enumerate(
                dataloader):
            self.logger.on_batch_begin()

            input_captions = self.get_input_captions(captions,
                                                     use_teacher_forcing)

            videos, captions, input_captions, classif_targets = list(
                map(Variable,
                    [videos, captions, input_captions, classif_targets]))

            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
                input_captions = input_captions.cuda(self.gpus[0])
                classif_targets = classif_targets.cuda(self.gpus[0])

            probs, classif_probs = \
                self.model((videos, input_captions), use_teacher_forcing)

            classif_loss = self.classif_loss_function(classif_probs,
                                                      classif_targets)
            captioning_loss = self.loss_function(probs, captions)

            loss = self.w_classif_loss * classif_loss + self.w_caption_loss * captioning_loss

            # print(">>>>>>>>>>>>>>>>>>> classif_loss: {}".format( classif_loss))
            # print(">>>>>>>>>>>>>>>>>>> captioning_loss: {}".format(captioning_loss))
            # loss = captioning_loss

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
                self.model.module.encoder.c3d_extractor.conv1.slant()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)

            captions = captions.cpu()
            predictions = predictions.cpu()

            captions_list.append(captions)
            predictions_list.append(predictions)

            classif_targets = classif_targets.cpu()
            classif_probs = classif_probs.cpu()

            batch_outputs = ScoreAttr(loss, string_captions, captions, predictions,
                                      classif_targets, classif_probs)

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
        
        return average_scores_dict, captions_list, predictions_list

    def test(self, test_dataloader, verbose_valid=False):
        test_average_scores = self.run_epoch(
        test_dataloader, 0, is_training=False, use_teacher_forcing = False,
                                             verbose = True)


