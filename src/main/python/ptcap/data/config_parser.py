import yaml


class ConfigParser(object):
    config_dict = {}

    def __init__(self, path):
        self.load_config_dict(path)

    @classmethod
    def load_config_dict(cls, path):
        with open(path, 'r') as f:
            cls.config_dict = yaml.load(f.read())

    @classmethod
    def get_value(cls, key):
        """
        Return the value of the given key in the config dictionary
        """

        split_key = key.split(".")
        hdict = cls.config_dict
        for hkey in split_key:
            hdict = hdict[hkey]

        return hdict

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config_dict, f)
