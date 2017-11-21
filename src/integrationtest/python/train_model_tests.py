# Code largely inspired by 20bn-rtorchn repo

import os

import torch

import fake_data as fkdata

from ptcap.data.config_parser import YamlConfig
from ptcap.train import train_model

CONFIG_PATH = [
    os.path.join(os.getcwd(), "src/main/configs/integration_test.yaml"),
    os.path.join(os.getcwd(), "src/main/configs/load_model_test.yaml")]


def check_saved_files(checkpoint_path, files_list):
    for file_name in files_list:
        if not os.path.exists(os.path.join(checkpoint_path, file_name)):
            raise FileNotFoundError


def test_on_device(cuda=False):
    checkpoints = []

    # Run models and get their checkpoint folders
    for config_path in CONFIG_PATH:
        checkpoint_folder = simulate_training(config_path, cuda)
        checkpoints.append(checkpoint_folder)

    # Clean up the checkpoint folders
    for checkpoint_folder in checkpoints:
        fkdata.remove_dir(checkpoint_folder)


def simulate_training(config_path, cuda):
    # Read config file for training a model from scratch
    config_obj = YamlConfig(config_path)

    config_obj.config_dict["device"]["gpus"] = [0] if cuda else None

    # Parse the model's checkpoint
    checkpoint_folder = os.path.join(
        os.getcwd(), config_obj.get("paths", "checkpoint_folder"))

    # Clean up checkpoint folder before training starts
    fkdata.remove_dir(checkpoint_folder)

    # Train the model
    train_model(config_obj, os.getcwd())

    # Check checkpoint folder

    check_saved_files(checkpoint_folder, ["config.yaml", "model.latest",
                                          "tokenizer_dicts"])
    return checkpoint_folder


def setup_fake_video_data():
    # Make sure you have a clean start
    fkdata.remove_dir(fkdata.TMP_DIR)

    # Create fake data first
    fkdata.create_fake_video_data()


if __name__ == '__main__':

    setup_fake_video_data()

    test_on_device(cuda=False)

    if torch.cuda.is_available():
        test_on_device(cuda=True)

    # Clean up fake data
    fkdata.remove_dir(fkdata.TMP_DIR)
