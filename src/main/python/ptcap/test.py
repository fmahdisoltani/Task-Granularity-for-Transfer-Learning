from ptcap.data.tokenizer import Tokenizer
from ptcap.data.dataset import VideoDataset
from ptcap.data.config_parser import ConfigParser
from ptcap.data.annotation_parser import AnnotationParser

from torch.utils.data import DataLoader

if __name__ == '__main__':

    #Build a dictionary that contains fields of config file
    ConfigParser.load_config_dict("path/to/config/file")

    #Find paths to test set
    test_path = ConfigParser.get_value(ConfigParser.TEST_SET)

    #Load Json annotation files
    test_annot = AnnotationParser.open_annotation(test_path)

    #TODO: Take the tokenizer that contains captions from annotation files

    #test_set = VideoDataset(test_annot, tokenizer_obj)