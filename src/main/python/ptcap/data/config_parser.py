import yaml
import os

class YamlConfig(object):

    def __init__(self, path=None, config_dict=None):
        self.config_dict = {} if config_dict is None else config_dict
        if path:
            self.parse(path)

    def parse(self, path):
        with open(path, 'r') as f:
            self.config_dict.update(yaml.load(f.read()))

    def get(self, *keys):
        output = self.config_dict
        for key in keys:
            output = output[key]
        return output

    def save(self, path):
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self.config_dict, f)

    @classmethod
    def dump(cls, path, config_dict):
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)
