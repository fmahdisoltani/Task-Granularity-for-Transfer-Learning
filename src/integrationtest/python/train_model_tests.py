# Code largely inspired by 20bn-rtorchn's repo

import os

import fake_data as fkdata

from ptcap.data.config_parser import YamlConfig
from ptcap.ptcap import caption

CONFIG_PATH = [os.path.join(
    os.getcwd(), "src/main/configs/integration_test.yaml"), os.path.join(
    os.getcwd(), "src/main/configs/load_model_test.yaml")]

CONFIG_PATH = ["/home/waseem/20bn-gitrepo/pytorch-captioning/src/main/configs/integration_test.yaml",
               "/home/waseem/20bn-gitrepo/pytorch-captioning/src/main/configs/load_model_test.yaml"]

CHECKPOINT_PATH = "model_checkpoints"


def check_saved_files(checkpoint_path, files_list):
    for file_name in files_list:
        if not os.path.exists(os.path.join(checkpoint_path, file_name)):
            raise FileNotFoundError


if __name__ == '__main__':
    # Make sure you have a clean start
    fkdata.remove_dir(fkdata.TMP_DIR)

    # Create fake data first
    fkdata.create_fake_video_data()

    # Training the model and check that it is saved
    config_obj = YamlConfig(CONFIG_PATH[0])

    checkpoint_folder = os.path.join(
        os.getcwd(), config_obj.get('paths', 'checkpoint_folder'))

    # Clean up checkpoint folder before training starts
    fkdata.remove_dir(checkpoint_folder)

    # Run captioning model
    caption(config_obj, os.getcwd())

    # Check checkpoint folder
    check_saved_files(checkpoint_folder, ["config.yaml", "model.best",
                                          "model.latest", "tokenizer_dicts"])

    caption(YamlConfig(CONFIG_PATH[1]), os.getcwd())

    # Clean up checkpoint folder
    fkdata.remove_dir(checkpoint_folder)

    # Remove everything
    fkdata.remove_dir(fkdata.TMP_DIR)
