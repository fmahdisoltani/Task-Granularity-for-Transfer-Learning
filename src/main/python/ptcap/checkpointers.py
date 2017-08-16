import os
import re

from ptcap.data.tokenizer import Tokenizer
from ptcap.data.config_parser import YamlConfig
from ptcap.model.captioners import *


class Checkpointer(object):

    def __init__(self):
        self.best_criterion = 1000


    def load(self, path):
        # read config file
        loaded_config = YamlConfig(path)
        # construct model based on config file
        #model = construct_model(loaded_config.get("model"))
        model = RtorchnCaptioner(tokenizer.get_vocab_size(), is_training=True,
                                  use_cuda=True)
        # load the model
        return model.load_state_dict(torch.load(path + "saved_model"))

    def initial_save(self, path, model, config_obj, tokenizer_obj, epoch=0):
        # Make output dir if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            files_list = os.listdir(path)
            for file in files_list:
                if file.contains("latest_model"):
                    epoch = re.match(".*?([0-9]+)$", file).group(1)
        # Save config file
        config_obj.save(path + "config" + str(epoch) + ".yaml")

        if epoch == 0:
            # Save the tokenizer
            tokenizer_obj.save_dictionaries(path)
            # Save the model as best_model and latest_model
            self.save_model(path, model, epoch)
        return epoch

    def save_model(self, path, model, epoch):
        torch.save(model.state_dict(), path + "latest_model" + str(epoch))
        torch.save(model.state_dict(), path + "best_model" + str(epoch))
