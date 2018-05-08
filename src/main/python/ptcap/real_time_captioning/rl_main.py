"""Training script.
Usage:
  train.py <config_path>
  train.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

import os
import torch
import copy

from docopt import docopt
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torch.nn as nn

import ptcap.data.preprocessing as prep
import ptcap.model.captioners


from ptcap.data.annotation_parser import V2Parser
from ptcap.data.config_parser import YamlConfig

from ptcap.checkpointers import Checkpointer
from ptcap.data.dataset import  GulpVideoDataset
from ptcap.data.tokenizer import Tokenizer
from ptcap.model.two_stream_encoders import TwoStreamEncoder
from ptcap.real_time_captioning.rl_trainer import RLTrainer
from ptcap.utils import DataParallelWrapper


if __name__ == "__main__":
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args["<config_path>"])
    caption_type = config_obj.get("targets", "caption_type")

    relative_path = ""
    training_path = os.path.join(relative_path,
                                 config_obj.get("paths", "train_annot"))
    videos_folder = config_obj.get("paths", "videos_folder")


    # Preprocess
    crop_size = config_obj.get("preprocess", "crop_size")
    scale = config_obj.get("preprocess", "scale")
    input_resize = config_obj.get("preprocess", "input_resize")

    preprocessor = Compose([prep.RandomCrop(crop_size),
                            prep.PadVideo(crop_size),
                            prep.Float32Converter(scale),
                            prep.PytorchTransposer()])

    training_parser = V2Parser(training_path, os.path.join(relative_path,
                               videos_folder),caption_type=caption_type)
    tokenizer = Tokenizer(**config_obj.get("tokenizer", "kwargs"))

    tokenizer.build_dictionaries(
        training_parser.get_captions_from_tmp_and_lbl())

    training_set = GulpVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer,
                                    preprocess=preprocessor,
                                    gulp_dir=videos_folder,
                                    size=input_resize)

    train_dataloader = DataLoader(training_set, shuffle=True, drop_last=False,
                                  **config_obj.get("dataloaders", "kwargs"))

    gpus = config_obj.get("device", "gpus")
    encoder_type = config_obj.get("model", "encoder")
    encoder_args = config_obj.get("model", "encoder_args")
    encoder_kwargs = config_obj.get("model", "encoder_kwargs")
    encoder_args = encoder_args or ()
    encoder_kwargs = encoder_kwargs or {}

    encoder = getattr(ptcap.model.two_stream_encoders, encoder_type)(
        *encoder_args,
        **encoder_kwargs)

    encoder = encoder if gpus is None else(
        DataParallelWrapper(encoder, device_ids=gpus).cuda(gpus[0])
    )

    pretrained_encoder = config_obj.get("pretrained", "pretrained_encoder")
    pretrained_encoder = os.path.join(relative_path, pretrained_encoder
                                     ) if pretrained_encoder else None
    pretrained_file = config_obj.get("pretrained", "pretrained_file")

    classif_layer = \
        nn.Linear(encoder.encoder_output_size, 178)

    checkpoint_folder = os.path.join(
                relative_path, config_obj.get("paths", "checkpoint_folder"))
    higher_is_better = config_obj.get("criteria", "higher_is_better")
    checkpointer = Checkpointer(checkpoint_folder, higher_is_better)


    init_state = checkpointer.load_model(encoder, classif_layer,
                                         None,
                                         folder=pretrained_encoder,
                                         filename=pretrained_file,
                                         load_encoder_only=True)
    _, encoder, _ = init_state

    rl_trainer = RLTrainer(encoder, classif_layer.cuda(), checkpointer, gpus=gpus)
    rl_trainer.train(train_dataloader)

