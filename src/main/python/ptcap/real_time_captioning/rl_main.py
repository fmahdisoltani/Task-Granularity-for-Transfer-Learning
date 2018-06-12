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

from ptcap.loggers import CustomLogger
from ptcap.data.annotation_parser import V2Parser
from ptcap.data.config_parser import YamlConfig
from ptcap.real_time_captioning.environment import Environment
from ptcap.real_time_captioning.agent import Agent

from ptcap.checkpointers import Checkpointer
from ptcap.data.dataset import  GulpVideoDataset
from ptcap.data.tokenizer import Tokenizer
from ptcap.model.two_stream_encoders import TwoStreamEncoder
from ptcap.real_time_captioning.rl_trainer import RLTrainer
from ptcap.tensorboardY import Seq2seqAdapter
from ptcap.utils import DataParallelWrapper
from rtorchn.data.preprocessing import CenterCropper


if __name__ == "__main__":
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args["<config_path>"])
    caption_type = config_obj.get("targets", "caption_type")

    relative_path = ""
    training_path = os.path.join(relative_path,
                                 config_obj.get("paths", "train_annot"))
    validation_path = os.path.join(relative_path,
                                   config_obj.get("paths", "validation_annot"))
    videos_folder = config_obj.get("paths", "videos_folder")

    valid_frequency = config_obj.get("validation", "frequency")



    # Preprocess
    crop_size = config_obj.get("preprocess", "crop_size")
    scale = config_obj.get("preprocess", "scale")
    input_resize = config_obj.get("preprocess", "input_resize")

    preprocessor = Compose([prep.RandomCrop(crop_size),
                            prep.PadVideo(crop_size),
                            prep.Float32Converter(scale),
                            prep.PytorchTransposer()])
    val_preprocessor = Compose([CenterCropper(crop_size),
                                prep.PadVideo(crop_size),
                                prep.Float32Converter(scale),
                                prep.PytorchTransposer()])

    training_parser = V2Parser(training_path, os.path.join(relative_path,
                               videos_folder),caption_type=caption_type)

    validation_parser = V2Parser(validation_path,
                                 os.path.join(relative_path,
                                              videos_folder),
                                 caption_type=caption_type)

    pretrained_encoder_path = config_obj.get("pretrained", "pretrained_encoder_path")
    pretrained_decoder_path = config_obj.get("pretrained", "pretrained_decoder_path")
    pretrained_path = config_obj.get("pretrained", "pretrained_path")

    # Build a tokenizer that contains all captions from annotation files
    tokenizer = Tokenizer(**config_obj.get("tokenizer", "kwargs"))
    if pretrained_decoder_path:
        tokenizer.load_dictionaries(pretrained_decoder_path)
        print("Inside pretrained", tokenizer.get_vocab_size())
        print("pretty fucked up")
    else:
        tokenizer.build_dictionaries(
            training_parser.get_captions_from_tmp_and_lbl())


    training_set = GulpVideoDataset(annotation_parser=training_parser,
                                    tokenizer=tokenizer,
                                    preprocess=preprocessor,
                                    gulp_dir=videos_folder,
                                    size=input_resize)

    validation_set = GulpVideoDataset(annotation_parser=validation_parser,
                                      tokenizer=tokenizer,
                                      preprocess=val_preprocessor,
                                      gulp_dir=videos_folder,
                                      size=input_resize)

    from ptcap.utils import CustomSubsetSampler

    sampler = CustomSubsetSampler(subset_size=50000, total_size=len(training_set))


    train_dataloader = DataLoader(training_set,  drop_last=False,
                                  sampler=sampler,
                                  **config_obj.get("dataloaders", "kwargs")
                                  )
    val_dataloader = DataLoader(validation_set, shuffle=False, drop_last=False,
                               **config_obj.get("dataloaders", "kwargs"))



    checkpoint_folder = os.path.join(
        relative_path, config_obj.get("paths", "checkpoint_folder"))
    higher_is_better = config_obj.get("criteria", "higher_is_better")
    checkpointer = Checkpointer(checkpoint_folder, higher_is_better)
    Checkpointer.save_meta(checkpoint_folder, config_obj, tokenizer)

    # reinforce stuff from config
    correct_w_reward = config_obj.get("reinforce","correct_w_reward")
    correct_r_reward = config_obj.get("reinforce", "correct_r_reward")
    incorrect_w_reward = config_obj.get("reinforce", "incorrect_w_reward")
    incorrect_r_reward = config_obj.get("reinforce", "incorrect_r_reward")


    gpus = config_obj.get("device", "gpus")
    encoder_type = config_obj.get("model", "encoder")
    encoder_args = config_obj.get("model", "encoder_args")
    encoder_kwargs = config_obj.get("model", "encoder_kwargs")
    encoder_args = encoder_args or ()
    encoder_kwargs = encoder_kwargs or {}

    encoder = getattr(ptcap.model.encoders, encoder_type)(
        *encoder_args,
        **encoder_kwargs)

    encoder = encoder if gpus is None else(
        DataParallelWrapper(encoder, device_ids=gpus).cuda(gpus[0])
    )

    classif_layer = \
        nn.Linear(encoder.module.encoder_output_size, 178)
    classif_layer = classif_layer if gpus is None else(
        DataParallelWrapper(classif_layer, device_ids=gpus).cuda(gpus[0])
    )

    decoder_type = config_obj.get("model", "decoder")
    decoder_args = config_obj.get("model", "decoder_args")
    decoder_kwargs = config_obj.get("model", "decoder_kwargs")
    decoder_args = decoder_args or ()
    decoder_kwargs["vocab_size"] = tokenizer.get_vocab_size()

    decoder = getattr(ptcap.model.decoders, decoder_type)(
        *decoder_args,
        **decoder_kwargs)


    if pretrained_encoder_path:
        _, encoder, _ = checkpointer.load_model(encoder, None,
                                        pretrained_path=pretrained_encoder_path,
                                        submodel="encoder")

        _, classif_layer, _ = checkpointer.load_model(classif_layer, None,
                                        pretrained_path=pretrained_encoder_path,
                                        submodel="classif_layer")

    env = Environment(encoder, decoder, classif_layer,  correct_w_reward, correct_r_reward,
                    incorrect_w_reward, incorrect_r_reward, tokenizer)
    agent = Agent()

    if pretrained_path:
        ckpt = torch.load(pretrained_path)
        print("loaded environment from ckpt")
        env_state_dict = ckpt["env"]
        env.load_state_dict(env_state_dict)

        print("loaded agent from ckpt")
        agent_state_dict = ckpt["agent"]
        agent.load_state_dict(agent_state_dict)

    if gpus is not None:
        env = env.cuda(gpus[0])
        agent = agent.cuda(gpus[0])

    # Setup the logger
    logger = CustomLogger(folder=checkpoint_folder, tokenizer=tokenizer)
    writer = Seq2seqAdapter(os.path.join(checkpoint_folder, "runs"),
                            config_obj.get("logging", "tensorboard_frequency"))

    rl_trainer = RLTrainer(env, agent,
                           checkpointer, logger, tokenizer, writer, gpus=gpus)
    rl_trainer.train(train_dataloader, val_dataloader,
                     criteria="classif_accuracy", valid_frequency=valid_frequency)


