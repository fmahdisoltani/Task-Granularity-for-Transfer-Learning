import torch

import ptcap.printers as prt

from collections import namedtuple
from collections import OrderedDict

from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer
from ptcap.metrics import Metrics


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, tokenizer, checkpoint_path,
                 pretrained_path=None, use_cuda=False):

        self.checkpointer = Checkpointer(checkpoint_path,
                                         pretrained_path=pretrained_path)
        init_state = self.checkpointer.load_model(model, optimizer, tokenizer)
        self.num_epochs, self.model, self.optimizer, self.tokenizer = init_state

        self.model = self.model.cuda() if use_cuda else self.model
        self.loss_function = loss_function.cuda() if use_cuda else loss_function
        self.use_cuda = use_cuda

    def train(self, train_dataloader, valid_dataloader, num_epoch,
              frequency_valid, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        for epoch in range(num_epoch):
            self.num_epochs += 1

            self.run_epoch(train_dataloader, epoch + 1, is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=verbose_train)

            if (epoch + 1) % frequency_valid == 0:
                average_loss = self.run_epoch(
                    valid_dataloader, epoch + 1, is_training=False,
                    use_teacher_forcing=teacher_force_valid,
                    verbose=verbose_valid
                )

                state_dict = self.get_state_dict()

                # remember best loss and save checkpoint
                self.checkpointer.save_model(state_dict, average_loss["average_loss"])

    def get_state_dict(self):
        return {
                    'epoch': self.num_epochs,
                    'model': self.model.state_dict(),
                    'best_score': self.checkpointer.best_score,
                    'optimizer': self.optimizer.state_dict(),
                }

    def get_function_dict(self):
        function_dict = OrderedDict()
        function_dict["loss"] = Metrics.compute_loss

        function_dict["accuracy"] = Metrics.token_level_accuracy

        function_dict["first_accuracy"] = Metrics.first_token_accuracy
        return function_dict

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):

        outputs = namedtuple("outputs", "loss captions predictions")
        metrics = Metrics(self.get_function_dict())

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

            captions = captions.cpu()
            predictions = predictions.cpu()

            epoch_outputs = outputs(loss, captions, predictions)

            metrics.compute_metrics(epoch_outputs, sample_counter + 1)

            prt.print_stuff(metrics.metrics_dict, self.tokenizer, is_training,
                            captions, predictions, epoch, sample_counter,
                            len(dataloader), verbose)

        # Take only the average of the metrics in metrics_dict
        average_metrics_dict = metrics.get_average_metrics()
        return average_metrics_dict


