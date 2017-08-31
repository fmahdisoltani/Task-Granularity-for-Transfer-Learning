import os

import ptcap.data.preprocessing as prep

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ptcap.checkpointers import Checkpointer
from ptcap.data.annotation_parser import JsonParser
from ptcap.data.config_parser import YamlConfig
from ptcap.data.dataset import (JpegVideoDataset, NumpyVideoDataset)
from ptcap.data.tokenizer import Tokenizer
from ptcap.losses import SequenceCrossEntropy
from ptcap.model.captioners import *
from ptcap.trainers import Trainer
from rtorchn.preprocessing import CenterCropper

if __name__ == '__main__':
    current_directory = os.getcwd()
    base_path_index = len(current_directory)
    for i in range(3):
        base_path_index = current_directory.rfind("/", 0, base_path_index)
    base_path = current_directory[:base_path_index]
    config_path = os.path.join(base_path,
                               "src/main/configs/integration_test.yaml")

    config_obj = YamlConfig(config_path)

    # Find paths to training, validation and test sets
    training_path = config_obj.get('paths', 'train_annot')
    training_path = os.path.join(base_path, training_path)
    validation_path = config_obj.get('paths', 'validation_annot')
    validation_path = os.path.join(base_path, validation_path)

    # Load Json annotation files
    training_parser = JsonParser(training_path, os.path.join(base_path,
                                 config_obj.get('paths', 'videos_folder')))
    validation_parser = JsonParser(validation_path, os.path.join(base_path,
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
    # use_cuda = config_obj.get('device', 'use_cuda')
    gpus = config_obj.get("device", "gpus")
    checkpoint_path = os.path.join(base_path,
                                   config_obj.get('paths', 'checkpoint_folder'))
    pretrained_path = config_obj.get('paths', 'pretrained_path')
    pretrained_path = os.path.join(base_path,
                                   pretrained_path) if pretrained_path else None

    preprocesser = Compose([prep.RandomCrop([24, 96, 96]),
                            prep.PadVideo([24, 96, 96]),
                            prep.Float32Converter(),
                            prep.PytorchTransposer()])

    val_preprocesser = Compose([CenterCropper([24, 96, 96]),
                                prep.PadVideo([24, 96, 96]),
                                prep.Float32Converter(),
                                prep.PytorchTransposer()])

    training_set = JpegVideoDataset(annotation_parser=training_parser,
                                     tokenizer=tokenizer,
                                     preprocess=preprocesser)

    validation_set = JpegVideoDataset(annotation_parser=validation_parser,
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
    Checkpointer.save_meta(config_obj, tokenizer)

    # Trainer
    trainer = Trainer(captioner, loss_function, optimizer, tokenizer,
                      checkpoint_path, pretrained_path=pretrained_path,
                        gpus=gpus)

    # Train the Model
    trainer.train(dataloader, val_dataloader, num_epoch, frequency_valid,
                  teacher_force_train, teacher_force_valid, verbose_train,
                  verbose_valid)