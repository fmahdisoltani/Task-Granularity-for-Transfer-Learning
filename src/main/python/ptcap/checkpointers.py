import csv
import os
import numpy as np

from ptcap.model.captioners import *


class Checkpointer(object):

    def __init__(self, checkpoint_folder, higher_is_better=False):
        self.checkpoint_folder = checkpoint_folder
        self.higher_is_better = higher_is_better
        self.best_score = np.Inf
        self.best_epoch = -1
        if self.higher_is_better:
            self.best_score *= -1

    def set_best_score(self, score=None, epoch=-1):
        if score is not None:
            better_higher_score = (score > self.best_score and
                                   self.higher_is_better)
            better_lower_score = (score < self.best_score and not
                                   self.higher_is_better)
            if better_higher_score or better_lower_score:
                self.best_score = score
                self.best_epoch = epoch
                return True

        return False

    def load_model(self, model, optimizer, folder=None, filename=None):
        pretrained_path = None if not folder or not filename else (
            os.path.join(folder, filename))
        init_epoch = 0
        if pretrained_path is None:
            print("Running the model from scratch")
        elif os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path)
            init_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            self.set_best_score(checkpoint["score"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded checkpoint {} @ epoch {}"
                  .format(pretrained_path, checkpoint["epoch"]))
        else:
            print("No checkpoint found at {}".format(pretrained_path))
        return init_epoch, model, optimizer

    def save_best(self, state, folder=None, filename="model.best"):
        if not folder:
            folder = self.checkpoint_folder
        new_best = self.set_best_score(state["score"], state["epoch"])
        if new_best:
            print("Saving best model, score: {:.4} @ epoch {}".
                  format(self.best_score, self.best_epoch))
            torch.save(state, os.path.join(folder, filename))
        else:
            print("Best model has a score of {:.4} @ epoch {}".
                  format(self.best_score, self.best_epoch))

    def save_latest(self, state, folder=None, filename="model.latest"):
        if not folder:
            folder = self.checkpoint_folder
        if state["score"] is not None:
            print("Saving latest model, score: {:.4} @ epoch {}".
                  format(state["score"], state["epoch"]))
        else:
            print("Saving latest model, score: None @ epoch {}".
                  format(state["epoch"]))
        torch.save(state, os.path.join(folder, filename))

    @classmethod
    def save_meta(cls, folder, config_obj, tokenizer):
        if not os.path.exists(folder):
            os.makedirs(folder)

        config_obj.save(folder)
        tokenizer.save_dictionaries(folder)

    def save_value_csv(self, value, folder=None, filename="loss"):
        if not folder:
            folder = self.checkpoint_folder
        ofile = open(os.path.join(folder, filename), "a")

        writer = csv.writer(ofile, delimiter=",")

        writer.writerow(value)

        ofile.close()
