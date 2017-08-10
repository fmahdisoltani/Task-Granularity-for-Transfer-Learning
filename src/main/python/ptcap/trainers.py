import torch
from torch.autograd import Variable

from ptcap.metrics import token_level_accuracy


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, tokenizer):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.tokenizer = tokenizer

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

        sample_counter = 0
        total_samples = len(dataloader)
        for i, (videos, _, captions) in enumerate(dataloader):
            sample_counter += 1

            videos, captions = Variable(videos), Variable(captions)
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)
            predictions = torch.squeeze(predictions)

            # compute accuracy
            accuracy = token_level_accuracy(captions, predictions)

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # print stuff
            if is_training:
                print("Training...")
            else:
                print("Validating...")
            print("Epoch {}".format(epoch + 1))
            print("Sample #{} out of {} samples".
                  format(sample_counter, total_samples))
            self.print_metrics(accuracy)
            if verbose:
                self.print_captions_and_predictions(captions, predictions)

    def print_captions_and_predictions(self, captions, predictions):

        for cap, pred in zip(captions, predictions):

            decoded_cap = self.tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = self.tokenizer.decode_caption(pred.data.numpy())

            print("__TARGET__: {}".format(decoded_cap))
            print("PREDICTION: {}\n".format(decoded_pred))

        print("*"*30)

    def print_metrics(self, accuracy):

        print("Batch Accuracy is: {}".format(accuracy.data.numpy()[0]))
