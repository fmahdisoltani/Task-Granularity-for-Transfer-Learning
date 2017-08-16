import os

import torch


class Checkpointer(object):

    #def __init__(self):


    def load(self, path):
        pass

    def save(self, model, checkpoint_path, config_obj):
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
        config_obj.save(checkpoint_path + "config.yaml")
        # # Save model definition file
        # copy2(os.path.join(MODELS_DIR, config.configdict['model']['type'].replace('.', '/') + '.py'),
        #       os.path.join(checkpoint_path, 'model.py'))
        #
        # copy2(os.path.join(MODELS_DIR, 'utils.py'),
        #       os.path.join(checkpoint_path, 'utils.py'))