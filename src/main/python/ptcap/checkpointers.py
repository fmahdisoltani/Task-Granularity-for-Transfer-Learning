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


    def init_model(self, pretrained_path, model, optimizer, tokenizer):
        # optionally resume from a checkpoint
        if pretrained_path:
            init_epoch, model, optimizer, tokenizer = \
                self.load_model(pretrained_path, model, optimizer, tokenizer)
        else:
            init_epoch = 0

        return init_epoch, model, optimizer, tokenizer

    def load_model(self, folder, filename, model, optimizer, tokenizer):
        pretrained_path = os.path.join(folder, filename) if folder else None
        init_epoch = 0
        if pretrained_path is None:
            print("Running the model from scratch")
        elif os.path.isfile(pretrained_path):
            tokenizer = Tokenizer()
            tokenizer.load_dictionaries(folder)
            checkpoint = torch.load(pretrained_path)
            init_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            self.best_score = checkpoint["best_score"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded checkpoint {} @ epoch {}"
                  .format(pretrained_path, checkpoint["epoch"]))
        else:
            print("No checkpoint found at {}".format(pretrained_path))
        return init_epoch, model, optimizer, tokenizer

    @classmethod
    def save_meta(cls, folder, config_obj, tokenizer):
        if not os.path.exists(folder):
            os.makedirs(folder)

        config_obj.save(folder)
        tokenizer.save_dictionaries(folder)

    def save_model(self, state, score, folder=None):
        if not folder:
            folder = self.checkpoint_folder
        torch.save(state, os.path.join(folder, "model.latest"))
        if not ((score > self.best_score) ^ self.higher_is_better):
            self.best_score = score
            print("Saving best model, score: {} @ epoch {}".
                  format(score, state["epoch"]))
            torch.save(state, os.path.join(folder, "model.best"))
