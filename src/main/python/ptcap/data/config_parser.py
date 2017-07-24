import yaml
import pprint

class ConfigParser(object):
    TRAINING_SET = "train"
    VALIDATION_SET = "validation"
    TEST_SET = "test"
    EXPANDED_CAPTION = "label"
    TEMPLATE = "template"
    config_dict = {}

    @classmethod
    def load_config_dict(cls, path):
        with open(path, 'r') as f:
            cls.config_dict = yaml.load(f.read())

        pprint.pprint(cls.config_dict)

    @classmethod
    def get_value(cls, key):
        """
        Return the value of the given key in the config dictionary
        """
        key = ""
        return cls.config_dict[key]

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config_dict, f)


