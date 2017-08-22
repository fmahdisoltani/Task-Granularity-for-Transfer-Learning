import os
import numpy as np

from ptcap.data.tokenizer import Tokenizer
from ptcap.model.captioners import *


class Checkpointer(object):

    def __init__(self, checkpoint_folder, pretrained_path=None,
                 higher_is_better=False):
        self.checkpoint_folder = checkpoint_folder
        self.pretrained_path = pretrained_path
        self.best_score = np.Inf
        self.higher_is_better = higher_is_better
        if self.higher_is_better:
            self.best_score *= -1

    def load_model(self, model, optimizer, tokenizer):
        init_epoch = 0
        if self.pretrained_path is None:
            print("Running the model from scratch")
        elif os.path.isfile(self.pretrained_path):
            folder = self.pretrained_path[:self.pretrained_path.rfind("/")]
            tokenizer = Tokenizer()
            tokenizer.load_dictionaries(folder)

            checkpoint = torch.load(self.pretrained_path)
            init_epoch = checkpoint["epoch"]
            model = model.load_state_dict(checkpoint["model"])

            self.best_score = checkpoint["best_score"]

            optimizer = optimizer.load_state_dict(checkpoint["optimizer"])

            print("Loaded checkpoint {} @ epoch {}"
                  .format(self.pretrained_path, checkpoint["epoch"]))
        else:
            print("No checkpoint found at {}".format(self.pretrained_path))
        return init_epoch, model, optimizer, tokenizer

    def save_model(self, state, score, filename="model"):
        torch.save(state, os.path.join(self.checkpoint_folder,
                                       filename + ".latest"))
        if not ((score > self.best_score) ^ self.higher_is_better):
            self.best_score = score
            print("Saving best model, score: {} @ epoch {}".
                  format(score, state["epoch"]))
            torch.save(state, os.path.join(self.checkpoint_folder,
                                           filename + ".best"))

    @classmethod
    def save_meta(cls, config_obj, tokenizer):
        path = config_obj.get('paths', 'checkpoint_folder')

        if not os.path.exists(path):
            os.makedirs(path)

        config_obj.save(os.path.join(path, "config.yaml"))
        tokenizer.save_dictionaries(path)

