import torch

import ptcap.printers as prt

import tables


from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer


class Trainer(object):
    def __init__(self, model, loss_function, optimizer, tokenizer,
                 checkpoint_path, folder=None, filename=None, gpus=None):

        self.use_cuda = True if gpus else False
        self.gpus = gpus
        self.checkpointer = Checkpointer(checkpoint_path)
        init_state = self.checkpointer.load_model( model,
                                                  optimizer, tokenizer, folder, filename)

        self.num_epochs, self.model, self.optimizer, self.tokenizer = init_state
        self.model = self.model.cuda(gpus[0]) if self.use_cuda else self.model
        self.loss_function = (loss_function.cuda(gpus[0])
                              if self.use_cuda else loss_function)

    def train(self, train_dataloader, valid_dataloader, num_epoch,
              frequency_valid, teacher_force_train=True,
              teacher_force_valid=False, verbose_train=False,
              verbose_valid=False):

        for epoch in range(num_epoch):

            average_loss_train = self.run_epoch(train_dataloader, epoch,
                            is_training=True,
                            use_teacher_forcing=teacher_force_train,
                            verbose=verbose_train)

            if (epoch) % frequency_valid == 0:
                average_loss_valid = self.run_epoch(

                    valid_dataloader, epoch, is_training=False,
                    use_teacher_forcing=teacher_force_valid,
                    verbose=verbose_valid
                )

                state_dict = self.get_state_dict()
                # remember best loss and save checkpoint
                self.checkpointer.save_model(state_dict, average_loss_valid)
                self.checkpointer.save_value_csv("valid_loss.csv",
                                                 [epoch, average_loss_valid])

            self.checkpointer.save_value_csv("train_loss.csv",
                                             [epoch, average_loss_train])




    def get_state_dict(self):
        return {
            'epoch': self.num_epochs,
            'model': self.model.state_dict(),
            'best_score': self.checkpointer.best_score,
            'optimizer': self.optimizer.state_dict(),
        }

    def run_epoch(self, dataloader, epoch, is_training,
                  use_teacher_forcing=False, verbose=True):

        average_loss = 0.

        for sample_counter, (videos, _, captions) in enumerate(dataloader):

            videos, captions = Variable(videos), Variable(captions)
            if self.use_cuda:
                videos = videos.cuda(self.gpus[0])
                captions = captions.cuda(self.gpus[0])
            probs = self.model((videos, captions), use_teacher_forcing)
            loss = self.loss_function(probs, captions)

            # Calculate a moving average of the loss
            average_loss += ((loss.data.cpu().numpy() - average_loss) /
                             (sample_counter + 1))
            # average_loss = loss

            if is_training:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            # convert probabilities to predictions
            _, predictions = torch.max(probs, dim=2)

            prt.print_stuff(average_loss, self.tokenizer, is_training, captions,
                            predictions, epoch + 1, sample_counter + 1,
                            len(dataloader), verbose)

        return average_loss
