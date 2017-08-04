import torch
import ptcap.metrics as metrics

from torch.autograd import Variable


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, num_epoch, valid_frequency,
                 tokenizer):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.valid_frequency = valid_frequency
        self.tokenizer = tokenizer

    def train(self, train_dataloader, valid_dataloader):
        for epoch in range(self.num_epoch):
            print("Epoch {}:".format(epoch + 1))
            self.run_epoch(train_dataloader, is_training=True)

            if (epoch + 1) % self.valid_frequency == 0:
                self.run_epoch(valid_dataloader, is_training=False)

    def run_epoch(self, dataloader, is_training):

        for i, (videos, _, captions) in enumerate(dataloader):
            videos, captions = Variable(videos), Variable(captions)
            probs = self.model((videos, captions))
            loss = self.loss_function(probs, captions)

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)
            _, predictions = torch.max(probs, dim=2)
            predictions = torch.squeeze(predictions)
            # compute accuracy
            accuracy = metrics.token_level_accuracy(captions, predictions)

            counter = 0
            for cap, pred in zip(captions, predictions):

                decoded_cap = self.tokenizer.decode_caption(cap.data.numpy())
                decoded_pred = self.tokenizer.decode_caption(pred.data.numpy())
                counter += 1

                print("__TARGET__: {}".format(decoded_cap))
                print("PREDICTION: {}\n".format(decoded_pred))

            print("*"*15)
            print("accuracy is: {}".format(accuracy))
            print ("\n \n")

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
