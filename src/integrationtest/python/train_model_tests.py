# Code largely inspired by 20bn-rtorchn's repo

import os

import fake_data as fkdata
from rtorchn.data.preprocessing import CenterCropper
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import ptcap.data.preprocessing as prep
from ptcap.checkpointers import Checkpointer
from ptcap.data.annotation_parser import JsonParser
from ptcap.data.config_parser import YamlConfig
from ptcap.data.dataset import (NumpyVideoDataset)
from ptcap.data.tokenizer import Tokenizer
from ptcap.losses import SequenceCrossEntropy
from ptcap.model.captioners import *
from ptcap.trainers import Trainer

CONFIG_PATH = [os.path.join(os.getcwd(),
                            "src/main/configs/integration_test.yaml")]
CHECKPOINT_PATH = "model_checkpoints"


def check_saved_files(checkpoint_path, files_list):
    for file_name in files_list:
        if not os.path.exists(os.path.join(checkpoint_path, file_name)):
            raise FileNotFoundError


def simulate_training_script(config_obj, fake_dir):
    # Find paths to training, validation and test sets
    training_path = os.path.join(fake_dir,
                                 config_obj.get('paths', 'train_annot'))
    validation_path = os.path.join(fake_dir,
                                   config_obj.get('paths', 'validation_annot'))

    # Load Json annotation files
    json_dir = os.path.join(fake_dir, "json")
    training_parser = JsonParser(training_path, os.path.join(fake_dir,
                                 config_obj.get('paths', 'videos_folder')))
    validation_parser = JsonParser(validation_path, os.path.join(fake_dir,
                                   config_obj.get('paths', 'videos_folder')))

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(training_parser.get_captions())

    # Load attributes of config file
    num_epoch = config_obj.get('training', 'num_epochs')
    frequency_valid = config_obj.get('validation', 'frequency')
    verbose_train = config_obj.get('training', 'verbose')
    verbose_valid = config_obj.get('validation', 'verbose')
    teacher_force_train = config_obj.get('training', 'teacher_force')
    teacher_force_valid = config_obj.get('validation', 'teacher_force')
    gpus = config_obj.get("device", "gpus")
    checkpoint_folder = os.path.join(fake_dir, config_obj.get('paths',
                                     'checkpoint_folder'))
    pretrained_path = config_obj.get('paths', 'pretrained_path')
    pretrained_path = os.path.join(fake_dir, pretrained_path
                                   ) if pretrained_path else None
    # Clean up checkpoint folder before training starts
    fkdata.remove_dir(checkpoint_folder)

    preprocesser = Compose([prep.RandomCrop([24, 96, 96]),
                            prep.PadVideo([24, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    val_preprocesser = Compose([CenterCropper([24, 96, 96]),
                                prep.PadVideo([24, 96, 96]),
                                prep.Float32Converter(),
                                prep.PytorchTransposer()])

    training_set = NumpyVideoDataset(annotation_parser=training_parser,
                                     tokenizer=tokenizer,
                                     preprocess=preprocesser)

    validation_set = NumpyVideoDataset(annotation_parser=validation_parser,
                                       tokenizer=tokenizer,
                                       preprocess=val_preprocesser)

    dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
                            **config_obj.get('dataloaders', 'kwargs'))

    val_dataloader = DataLoader(validation_set, shuffle=True, drop_last=False,
                                **config_obj.get('dataloaders', 'kwargs'))

    captioner = CNN3dLSTM(vocab_size=tokenizer.get_vocab_size(),
                          go_token=tokenizer.encode_token(tokenizer.GO),
                          gpus=gpus)

    # Loss and Optimizer
    loss_function = SequenceCrossEntropy()
    params = list(captioner.parameters())

    optimizer = torch.optim.Adam(params,
                                 lr=config_obj.get('training', 'learning_rate'))

    # Prepare checkpoint directory and save config
    Checkpointer.save_meta(checkpoint_folder, config_obj, tokenizer)

    # Trainer
    pretrained_folder = config_obj.get("paths", "pretrained_path")
    trainer = Trainer(captioner, loss_function, optimizer, tokenizer,
                      checkpoint_folder, folder=pretrained_folder,
                      filename="model.best", gpus=gpus)

    # Train the Model
    trainer.train(dataloader, val_dataloader, num_epoch, frequency_valid,
                  teacher_force_train, teacher_force_valid, verbose_train,
                  verbose_valid)

    # Check checkpoint folder
    check_saved_files(checkpoint_folder, ["config.yaml", "model.best",
                                          "model.latest", "tokenizer_dicts"])
    # Clean up checkpoint folder
    fkdata.remove_dir(checkpoint_folder)


if __name__ == '__main__':
    # Make sure you have a clean start
    fkdata.remove_dir(fkdata.TMP_DIR)

    # Create fake data first
    fkdata.create_fake_video_data()

    # Training the model and check that it is saved
    config = YamlConfig(CONFIG_PATH[0])
    simulate_training_script(config, os.getcwd())

    # Remove everything
    fkdata.remove_dir(fkdata.TMP_DIR)
