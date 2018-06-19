import os
import yaml

import ptcap.data.preprocessing
import ptcap.data.dataset

from torchvision.transforms import Compose


class YamlConfig(object):

    def __init__(self, path=None, config_dict=None):
        self.config_dict = {} if config_dict is None else config_dict
        if path:
            self.parse(path)

    def parse(self, path):
        with open(path, "r") as f:
            self.config_dict.update(yaml.load(f.read()))

    def get(self, *keys):
        output = self.config_dict
        for key in keys:
            output = output[key]
        return output

    def save(self, folder, filename="config.yaml"):
        with open(os.path.join(folder, filename), "w") as f:
            yaml.dump(self.config_dict, f)

    def get_preprocessor(self, key):
        """
        :param key: should be either "valid" or "train"
        """
        prep_list = []
        for prep_dict in self.get("preprocess", key):
            args = prep_dict["args"] or ()
            prep_list.append(
                getattr(ptcap.data.preprocessing, prep_dict["type"])(*args))

        preprocessor = Compose(prep_list)
        return preprocessor

    def get_dataset(self, key, parser, tokenizer, preprocessor):
        """
        :param key: should be either "train" or "valid"
        """
        dataset_type = self.get("dataset", key+"_dataset_type")
        dataset_kwargs = self.get("dataset", key+"_dataset_kwargs")
        dataset_kwargs = dataset_kwargs or {}
        dataset = getattr(ptcap.data.dataset, dataset_type)(
                          annotation_parser=parser,
                          tokenizer=tokenizer,
                          preprocess=preprocessor,
                          **dataset_kwargs)
        return dataset

    @classmethod
    def dump(cls, path, config_dict):
        with open(path, "w") as f:
            yaml.dump(config_dict, f)
