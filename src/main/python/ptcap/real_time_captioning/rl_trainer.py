import torch.optim as optim
import torch

from collections import OrderedDict
from collections import namedtuple

from itertools import count
from torch.autograd import Variable

from ptcap.real_time_captioning.environment import Environment
from ptcap.real_time_captioning.agent import Agent
from ptcap.utils import DataParallelWrapper

from torch.autograd import Variable
from ptcap.scores import (ScoresOperator, caption_accuracy, classif_accuracy,
                          first_token_accuracy, policy_loss_to_numpy,
                          classif_loss_to_numpy,
                          token_accuracy)


class RLTrainer(object):
    def __init__(self, encoder, classif_layer, checkpointer, logger, gpus=None):

        self.gpus = gpus
        self.use_cuda = True if gpus else False
        self.logger = logger
        self.env = Environment(encoder, classif_layer)

        self.agent = Agent()

        params = list(self.env.parameters()) + \
                 list(self.agent.parameters()) \
            # + list(self.env.module.classif_layer.parameters())
        self.optimizer = optim.Adam(params, lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.9)

        self.checkpointer = checkpointer
        self.num_epochs = 0

    def get_function_dict(self):

        function_dict = OrderedDict()
        function_dict["policy_loss"] = policy_loss_to_numpy
        function_dict["classif_loss"] = classif_loss_to_numpy
        function_dict["classif_accuracy"] = classif_accuracy

        return function_dict

    def train(self, dataloader, val_dataloader, criteria):
        print("*" * 10)
        running_reward = 0
        logging_interval = 1000
        stop_training = False
        valid_frequency = 1
        epoch = 0

        while not stop_training:
            if epoch % valid_frequency == 0:
                valid_average_scores = self.run_epoch(val_dataloader, epoch,
                                                      is_training=True)  # TODO: HACK

                # remember best loss and save checkpoint
                # self.score = valid_average_scores["avg_" + criteria]
#                self.score = valid_average_scores["avg_" + criteria]
                state_dict = self.get_trainer_state()
                self.checkpointer.save_best(state_dict)
                self.checkpointer.save_value_csv([epoch, 0],#self.score],
                                                 filename="valid_loss")

            self.num_epochs += 1
            epoch += 1
            train_average_scores = self.run_epoch(dataloader, epoch,
                                                  is_training=True)
#            train_avg_loss = train_average_scores["avg_" + criteria]
            state_dict = self.get_trainer_state()
           # self.checkpointer.save_latest(state_dict)
           # self.checkpointer.save_value_csv([epoch, train_avg_loss],
                                         #    filename="train_loss")

            # stop_training = self.update_stop_training(epoch, max_num_epochs)

        self.logger.on_train_end(self.scheduler.best)

    def get_trainer_state(self):
        return {
            "env": self.env.state_dict(),
            "agent": self.agent.state_dict(),
            "score": 0,#self.score,
            "epoch": self.num_epochs
        }

    def run_episode(self, i_episode, videos, classif_targets, is_training):


        # print("episode{}".format(i_episode))
        self.env.reset(videos)
        self.agent.lstm_hidden = None

        finished = False
        reward_seq = []
        action_seq = []
        logprob_seq = []
        while not finished:
            print("jjjjj")
            state = self.env.get_state()
            action, logprob = self.agent.select_action(state)
            action_seq.append(action.data.numpy()[0])
            logprob_seq.append(logprob)
            reward, classif_probs = self.env.update_state(action, action_seq,
                                                          classif_targets)
            reward_seq.append(reward)
            finished = self.env.check_finished()

        # returns, policy_loss, classif_loss = \
        #     self.agent.compute_losses(reward_seq, logprob_seq, classif_probs, classif_targets)
        # loss = policy_loss.cuda() * 0.01 + classif_loss.cuda()
        #
        # if is_training:
        #     loss.backward()
        #
        # if i_episode % 1 == 0:  # replace 1 with batch_size
        #     # policy_loss.backward()
        #     self.optimizer.step()
        #     self.scheduler.step()
        #     self.optimizer.zero_grad()

        return action_seq, reward_seq, logprob_seq, classif_probs

    def run_epoch(self, dataloader, epoch, is_training, logging_interval=1000,
                  batch_size=4):
        self.logger.on_epoch_begin(epoch)
        if is_training:
            self.env.train()
            self.agent.train()
        else:
            self.env.eval()
            self.agent.eval()
        running_reward = 0
        ScoreAttr = namedtuple("ScoresAttr",
                               "policy_loss classif_loss preds "
                               "classif_targets classif_probs")
        scores = ScoresOperator(self.get_function_dict())

        # loop over videos
        batch_reward_seq = []
        batch_action_seq = []
        batch_logprob_seq = []

        for i_episode, (videos_batch, _, _, classif_targets_batch) \
                              in enumerate(dataloader):
            videos_batch = Variable(videos_batch)
            if self.use_cuda:

                videos_batch = videos_batch.cuda(self.gpus[0])
                classif_targets_batch = classif_targets_batch.cuda(self.gpus[0])

            self.logger.on_batch_begin()
            for jj in range(10): #batch_size instead of 10

                video = videos_batch[jj:jj+1]
                classif_target = classif_targets_batch[jj:jj+1]


                # action_seq, reward_seq, logprob_seq, classif_probs = \
                #    self.run_episode(i_episode, videos, classif_targets,
                #                     is_training)

                ####################
                self.env.reset(video)
                self.agent.lstm_hidden = None

                finished = False
                reward_seq = []
                action_seq = []
                logprob_seq = []
                # while not finished:
                state = self.env.get_state()

                x = self.agent.prepare_policy_input(state)
                self.agent.cuda()
                x = self.agent.input_layer(x.cuda())
                lstm_output, self.agent.lstm_hidden = self.agent.lstm(x,
                                                                      self.agent.lstm_hidden)
                self.agent.lstm.flatten_parameters()
                lstm_out_projected = self.agent.output_layer(lstm_output)
                action_probs = self.agent.softmax(lstm_out_projected)
                action_probs = action_probs.squeeze(dim=1)

                # $$$$$$$$$$$$$$
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                logprob = torch.sum(dist.log_prob(action))
                #action_seq.cuda()
                action_seq.append(action)
                logprob_seq.append(logprob)
                reward, classif_probs = self.env.update_state(action,
                                                              action_seq,
                                                              classif_target)
                reward_seq.append(reward)

                batch_action_seq = batch_action_seq +  action_seq
                batch_logprob_seq = batch_logprob_seq + logprob_seq
                batch_reward_seq = batch_reward_seq + reward_seq

                ########################################
                returns, policy_loss, classif_loss = \
                    self.agent.compute_losses(batch_reward_seq,
                                              batch_logprob_seq,
                                              classif_probs,
                                              classif_target)

                loss = policy_loss[0] * 0.01 + classif_loss
                running_reward += returns[0]

                if jj == 9 :


                    # action, logprob = self.agent.select_action(state)



                    if is_training:
                        print("("*100)
                        loss.backward()
                    # policy_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    batch_reward_seq = []
                    batch_action_seq = []
                    batch_logprob_seq = []

                # self.checkpointer.save_value_csv([epoch, train_avg_loss],
                #                                  filename="train_loss")

                # convert probabilities to predictions
                _, predictions = torch.max(classif_probs, dim=1)
                predictions = predictions.cpu()

                episode_outputs = ScoreAttr(policy_loss,
                                            classif_loss,
                                            predictions,
                                            classif_target,
                                            classif_probs)

                #scores_dict = scores.compute_scores(episode_outputs,
                #                                    i_episode + 1)

                # Take only the average of the scores in scores_dict
                #average_scores_dict = scores.get_average_scores()
                #if i_episode % logging_interval == 0:
                ##    print('\nEpisode {}\tAverage return: {:.2f}'.format(
                #        i_episode,
                #        running_reward / logging_interval))
                #    print(action_seq)
                #    running_reward = 0
                #self.logger.on_batch_end(average_scores_dict, None,
                #                             predictions,
                #                             is_training,
                #                             total_samples=len(dataloader),
                #                             verbose=False)


        #self.logger.on_epoch_end(average_scores_dict, True,
        #                         total_samples=len(dataloader))
        #return average_scores_dict
                return



