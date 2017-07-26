import pprint

from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import JpegVideoDataset
from ptcap.data.config_parser import ConfigParser
from ptcap.data.annotation_parser import JsonParser

from torch.utils.data import DataLoader
from PIL import Image


if __name__ == '__main__':



    #Build a dictionary that contains fields of config file
    config_path = ("/Users/farzaneh/PycharmProjects/TwentyBN/pytorch-captioning/"
            "src/main/configs/video2caption.yaml")

    config_obj = ConfigParser(config_path)

    #Find paths to training, validation and test sets
    training_path = config_obj.config_dict['paths']['train_annot']

    # Load Json annotation files
    training_annot = JsonParser(training_path,
                            config_obj.config_dict['paths']['videos_folder'])

    #Build a tokenizer that contains all captions from annotation files
    training_captions = training_annot.get_captions()
    tokenizer_obj = Tokenizer(training_captions) #TODO: ask Roland

    training_set = JpegVideoDataset(annotation_obj=training_annot,
                                    tokenizer_obj=tokenizer_obj)

    dataloader = DataLoader(training_set, batch_size=2,
                            shuffle=True, num_workers=1)

    for it, sample_batch in enumerate(dataloader):
        video, string_caption, tokenized_caption = sample_batch