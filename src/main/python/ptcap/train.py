from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import VideoDataset
from ptcap.data.config_parser import ConfigParser
from ptcap.data.annotation_parser import AnnotationParser

from torch.utils.data import DataLoader

if __name__ == '__main__':

    #Build a dictionary that contains fields of config file
    ConfigParser.load_config_dict("/Users/farzaneh/PycharmProjects/TwentyBN"
                                  "/pytorch-captioning/src/main/configs"
                                  "/video2caption.yaml")

    #Find paths to training, validation and test sets
    training_path = ConfigParser.get_value(ConfigParser.TRAINING_SET)
    validation_path = ConfigParser.get_value(ConfigParser.VALIDATION_SET)
    test_path = ConfigParser.get_value(ConfigParser.TEST_SET)

    #Load Json annotation files
    training_annot = AnnotationParser.open_annotation(training_path)
    validation_annot = AnnotationParser.open_annotation(validation_path)
    test_annot = AnnotationParser.open_annotation(test_path)

    #Build a tokenizer that contains all captions from annotation files
    tokenizer_obj = Tokenizer([training_annot]) #TODO: ask Roland

    training_set = VideoDataset(training_annot, tokenizer_obj)

    dataloader = DataLoader(training_set, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())

