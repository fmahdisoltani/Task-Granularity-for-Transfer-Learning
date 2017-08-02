from torch.autograd import Variable

import ptcap.metrics as metrics


class Trainer(object):
    def __init__(self, model,
                 loss_function, optimizer, num_epoch, num_valid):
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
            outputs = self.model((videos, captions))
            loss = self.loss_function(outputs, captions)

            # compute accuracy
            accuracy = metrics.token_level_accuracy(captions, outputs)
            print("accuracy is: {}".format(accuracy))

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

