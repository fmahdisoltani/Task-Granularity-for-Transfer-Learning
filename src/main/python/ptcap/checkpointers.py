import os

import torch

from ptcap.data.config_parser import YamlConfig


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


    def save(self, model, checkpoint_path, config_obj, tokenizer_obj):
        """
        Create the checkpoint directory, save the config file,
         save the network definition file (necessary to reload
        the model) and `models.utils.py`
        """

        # Make output dir
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(model.state_dict(), checkpoint_path + "saved_model")

        # Save config file
        config_obj.save(checkpoint_path + "saved_config.yaml")
        # # Save model definition file
        # copy2(os.path.join(MODELS_DIR, config.configdict['model']['type'].replace('.', '/') + '.py'),
        #       os.path.join(checkpoint_path, 'model.py'))
        #
        # copy2(os.path.join(MODELS_DIR, 'utils.py'),
        #       os.path.join(checkpoint_path, 'utils.py'))