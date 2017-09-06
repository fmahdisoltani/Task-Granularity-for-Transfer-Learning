import csv
import os
import numpy as np

from ptcap.data.tokenizer import Tokenizer
from ptcap.model.captioners import *


class Checkpointer(object):

    def __init__(self, checkpoint_folder, higher_is_better=False):
        self.checkpoint_folder = checkpoint_folder
        self.best_score = np.Inf
        self.higher_is_better = higher_is_better
        if self.higher_is_better:
            self.best_score *= -1

    def load_model(self, model, optimizer, tokenizer,
                   folder=None, filename=None):
        pretrained_path = None if not folder or not filename else (
            os.path.join(folder, filename))
        init_epoch = 0
        if pretrained_path is None:
            print("Running the model from scratch")
        elif os.path.isfile(pretrained_path):
            tokenizer = Tokenizer()
            tokenizer.load_dictionaries(folder)
            checkpoint = torch.load(pretrained_path)
            init_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            self.best_score = checkpoint["score"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded checkpoint {} @ epoch {}"
                  .format(pretrained_path, checkpoint["epoch"]))
        else:
            print("No checkpoint found at {}".format(pretrained_path))
        return init_epoch, model, optimizer, tokenizer

    def save_best(self, state, folder=None, filename="model.best"):
        if not folder:
            folder = self.checkpoint_folder
        score = state["score"]
        if not ((score > self.best_score) ^ self.higher_is_better):
            self.best_score = score
            print("Saving best model, score: {} @ epoch {}".
                  format(score, state["epoch"]))
            torch.save(state, os.path.join(folder, "model.best"))

    def save_latest(self, state, folder=None, filename="model.latest"):
        if not folder:
            folder = self.checkpoint_folder
        print("Saving latest model, score: {} @ epoch {}".
              format(state["score"], state["epoch"]))
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
