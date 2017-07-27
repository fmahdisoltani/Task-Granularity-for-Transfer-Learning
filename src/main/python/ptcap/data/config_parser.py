import yaml


class ConfigParser(object):

    def __init__(self, path):
        self.config_dict = self.read_yaml(path)

    def read_yaml(self, path):
        with open(path, 'r') as f:
            config_dict = yaml.load(f.read())
        return config_dict

    def get_value(self, key):
        """
        Return the value of the given key in the config dictionary
        """

        split_key = key.split(".")
        hdict = self.config_dict
        for hkey in split_key:
            hdict = hdict[hkey]

        return hdict

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config_dict, f)
