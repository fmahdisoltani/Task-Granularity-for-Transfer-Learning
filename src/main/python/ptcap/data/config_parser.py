import yaml


class ConfigParser(object):
    TRAINING_SET = "train"
    VALIDATION_SET = "validation"
    TEST_SET = "test"
    EXPANDED_CAPTION = "label"
    config_dict = {}

    def load_config_dict(cls, path):
        with open(path, 'r') as f:
            cls.config_dict = yaml.load(f.read())

    @classmethod
    def get_value(cls, key):
        """
        Return the value of the given key in the config dictionary
        """
        pass

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config_dict, f)
