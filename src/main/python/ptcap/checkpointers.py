import os
import numpy as np

from ptcap.data.tokenizer import Tokenizer
from ptcap.model.captioners import *


class Checkpointer(object):

    def __init__(self, higher_is_better=False):
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

    def load_model(self, pretrained_path, model, optimizer, tokenizer):
        print("Loading checkpoint {}".format(pretrained_path))
        if os.path.isfile(pretrained_path):
            model_folder = pretrained_path[:pretrained_path.rfind("/")]
            tokenizer = Tokenizer()
            tokenizer.load_dictionaries(model_folder)
            checkpoint = torch.load(pretrained_path)
            init_epoch = checkpoint["epoch"]
            model = model.load_state_dict(checkpoint["model"])
            self.best_score = checkpoint["best_score"]
            optimizer = optimizer.load_state_dict(checkpoint["optimizer"])

            print("Loaded checkpoint {} @ epoch {}"
                  .format(pretrained_path, checkpoint["epoch"]))
        else:
            init_epoch = 0
            print("No checkpoint found at {}".format(pretrained_path))
        return init_epoch, model, optimizer, tokenizer

    def save_meta(self, config_obj, tokenizer):
        self.folder = config_obj.get("paths", "checkpoint_folder")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        config_obj.save(self.folder + "config.yaml")
        tokenizer.save_dictionaries(self.folder)

    def save_model(self, state, score, filename="model"):
        torch.save(state, os.path.join(self.folder, filename + ".latest"))
        if not ((score > self.best_score) ^ self.higher_is_better):
            self.best_score = score
            print("Saving best model, score: {} @ epoch {}".
                  format(score, state["epoch"]))
            torch.save(state, os.path.join(self.folder, filename + ".best"))
