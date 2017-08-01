import torch

from torch.autograd import Variable


class Trainer(object):
    def __init__(self, learning_rate, model,
                 loss_function, optimizer, num_epoch, num_valid):
        self.learning_rate = learning_rate
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.num_valid = num_valid

    def train(self, train_dataloader, valid_dataloader):
        for ep in range(self.num_epoch):
            self.run_epoch(train_dataloader, is_training=True)

            if (ep + 1) % self.num_valid == 0:
                self.run_epoch(valid_dataloader, is_training=False)

    def run_epoch(self, dataloader, is_training):

        for i, (videos, _, captions) in enumerate(dataloader):
            videos, captions = Variable(videos), Variable(captions)
            batch_size, num_steps = captions.size()
            outputs = self.model((videos, captions))
            loss = self.loss_function(outputs, captions)

            # compute accuracy
            _, predictions = torch.max(outputs, dim=2)
            equal_values = captions.eq(predictions).sum().float()
            accuracy = equal_values * 100.0 / (batch_size * num_steps)
            print("accuracy is: {}".format(accuracy))

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()