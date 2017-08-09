import torch
import ptcap.metrics as metrics

from torch.autograd import Variable


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, num_epoch, valid_frequency,
                 tokenizer, verbose_train=False, verbose_valid=False):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.valid_frequency = valid_frequency
        self.tokenizer = tokenizer
        self.verbose_train = verbose_train
        self.verbose_valid = verbose_valid

    def train(self, train_dataloader, valid_dataloader, teacher_force_train,
              teacher_force_valid):
        for epoch in range(self.num_epoch):
            print("Epoch {}".format(epoch + 1))
            self.run_epoch(train_dataloader, is_training=True,
                           use_teacher_forcing=teacher_force_train,
                           verbose=self.verbose_train)

            if (epoch + 1) % self.valid_frequency == 0:
                self.run_epoch(valid_dataloader, is_training=False,
                               use_teacher_forcing=teacher_force_valid,
                               verbose=self.verbose_valid)

    def run_epoch(self, dataloader, is_training, use_teacher_forcing, verbose):

        for i, (videos, _, captions) in enumerate(dataloader):
            videos, captions = Variable(videos), Variable(captions)
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)
            predictions = torch.squeeze(predictions)

            # compute accuracy
            accuracy = metrics.token_level_accuracy(captions, predictions)

            if verbose:
                self.print_metrics(accuracy)
                self.print_captions_and_predictions(captions, predictions)

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

    def print_captions_and_predictions(self, captions, predictions):

        for cap, pred in zip(captions, predictions):

            decoded_cap = self.tokenizer.decode_caption(cap.data.numpy())
            decoded_pred = self.tokenizer.decode_caption(pred.data.numpy())

            print("__TARGET__: {}".format(decoded_cap))
            print("PREDICTION: {}\n".format(decoded_pred))

        print("*"*30)

    def print_metrics(self, accuracy):

        print("Batch Accuracy is: {}".format(accuracy.data.numpy()[0]))
