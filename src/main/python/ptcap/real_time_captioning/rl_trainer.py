import torch.optim as optim
import torch

from collections import OrderedDict
from collections import namedtuple

from itertools import count
from torch.autograd import Variable

from ptcap.utils import DataParallelWrapper


from torch.autograd import Variable
from ptcap.scores import (ScoresOperator, caption_accuracy, classif_accuracy,
                          first_token_accuracy, policy_loss_to_numpy, classif_loss_to_numpy,
                          token_accuracy, get_wait_time, get_reward)


class RLTrainer(object):
    def __init__(self, env, agent, checkpointer, logger, tokenizer, writer,
                 w_classif, w_captioning, gpus=None):

        self.gpus = gpus
        self.use_cuda = True if gpus else False
        self.logger = logger
        self.env = env #Environment(encoder, classif_layer)

        self.agent = agent # Agent().cuda()

        params = list(self.agent.parameters())\
                 + list(self.env.decoder.parameters()) \
                 # + list(self.env.classif_layer.parameters())
        self.optimizer = optim.Adam(params, lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(
                         self.optimizer, step_size=10000, gamma=0.9)

        self.checkpointer = checkpointer
        self.num_epochs = 0
        self.tokenizer = tokenizer
        self.writer = writer
        self.w_classif = w_classif

    def get_function_dict(self, is_training=True):
        scoring_functions = []
        scoring_functions.append(get_wait_time)
        scoring_functions.append(policy_loss_to_numpy)
        scoring_functions.append(classif_loss_to_numpy)
        scoring_functions.append(classif_accuracy)
        scoring_functions.append(get_reward)

        return scoring_functions

    def train(self, dataloader, val_dataloader, criteria, valid_frequency=1):
        print("*"*10)
        running_reward = 0
        logging_interval = 1000
        stop_training = False
        epoch = 0

        while not stop_training:

            if epoch % valid_frequency == 0:
                self.env.toggle_training_mode()

                valid_average_scores = self.run_epoch(val_dataloader, epoch, is_training=self.env.is_training)

                # remember best loss and save checkpoint
                #self.score = valid_average_scores["avg_" + criteria]
                self.score = valid_average_scores["avg_" + criteria]
                state_dict = self.get_trainer_state()
                self.checkpointer.save_best(state_dict)
                self.checkpointer.save_value_csv([epoch, self.score],
                                                 filename="valid_loss")
                self.env.toggle_training_mode()

            self.num_epochs += 1
            epoch += 1
            train_average_scores = self.run_epoch(dataloader, epoch, is_training=self.env.is_training)
            self.score = train_average_scores["avg_"+criteria]
            state_dict = self.get_trainer_state()
            self.checkpointer.save_latest(state_dict)
            self.checkpointer.save_value_csv([epoch, self.score ],
                                             filename="train_loss")



            #stop_training = self.update_stop_training(epoch, max_num_epochs)


#        self.logger.on_train_end(self.scheduler.best)

    def get_trainer_state(self):
        return {
            "env": self.env.state_dict(),
            "agent": self.agent.state_dict(),
            "score": self.score,
            "epoch": self.num_epochs
        }

    def run_episode(self, i_episode, videos, classif_targets, is_training):


        #print("episode{}".format(i_episode))
        self.env.reset(videos)

        finished = False
        reward_seq = []
        action_seq = []
        logprob_seq = []
        while not finished:

            state = self.env.get_state()
            action, logprob = self.agent.select_action(state)
            action_seq.append(action)
            logprob_seq.append(logprob)
            reward, classif_probs = \
                self.env.update_state(action, action_seq, classif_targets)
            reward_seq.append(reward)
            finished = self.env.check_finished()


        return action_seq, reward_seq, logprob_seq, classif_probs

    def run_epoch(self, dataloader, epoch, is_training, logging_interval=1000):
        self.logger.on_epoch_begin(epoch)
        if is_training:
            self.env.train()
            self.agent.train()
        else:
            self.env.eval()
            self.agent.eval()
        running_reward = 0
        ScoreAttr = namedtuple("ScoresAttr",
                           "wait_time reward policy_loss classif_loss "
                           "classif_targets classif_probs "
                           " predictions")

        scores = ScoresOperator(self.get_function_dict())
        loss = 0
        for i_episode, (videos_b, _, _, classif_targets_b) in enumerate(dataloader):
            videos_b = Variable(videos_b)

            if self.use_cuda:
                videos_b = videos_b.cuda(self.gpus[0])
                classif_targets_b = classif_targets_b.cuda(self.gpus[0])

            self.logger.on_batch_begin()
            num_samples = videos_b.size()[0]
            for i_sample in range(num_samples):
                videos = videos_b[i_sample:i_sample+1]
                classif_targets = classif_targets_b[i_sample:i_sample+1]

                action_seq, reward_seq, logprob_seq, classif_probs= \
                    self.run_episode(i_episode, videos, classif_targets,

                                     is_training)

                returns, policy_loss, classif_loss = \
                    self.agent.compute_losses(reward_seq,
                                              logprob_seq,
                                              classif_probs,
                                              classif_targets
                                              )

                loss = loss + policy_loss.cuda() * 0.01 + \
                       self.w_classif* classif_loss.cuda()
                wait_time = len(action_seq)
                running_reward += returns[0]



                # self.checkpointer.save_value_csv([epoch, train_avg_loss],
                #                                  filename="train_loss")

                # convert probabilities to predictions
                _, predictions = torch.max(classif_probs, dim=1)
                predictions = predictions.cpu()




                episode_outputs = ScoreAttr(
                                        wait_time,
                                        running_reward,
                                        policy_loss,
                                        classif_loss.cpu(),

                                        classif_targets.cpu(),
                                        classif_probs.cpu(),
                                        predictions

                                       )

                scores_dict = scores.compute_scores(episode_outputs,
                                                    i_episode + 1)

                # Take only the average of the scores in scores_dict
                average_scores_dict = scores.get_average_scores()
            if i_episode % logging_interval == 0:
                print('\nEpisode {}\tAverage return: {:.2f}'.format(
                    i_episode,
                    running_reward / logging_interval))
                print(action_seq)
                running_reward = 0
            self.logger.on_batch_end(average_scores_dict, None, predictions,
                 is_training, total_samples=len(dataloader), verbose=False)

            #if i_episode % 3 ==0:
            if is_training:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            loss = 0



        self.logger.on_epoch_end(average_scores_dict, is_training,
                                     total_samples=len(dataloader))

        self.writer.add_scalars(average_scores_dict, epoch, is_training)
        return average_scores_dict



