import os
import shutil
import unittest

from testfixtures import TempDirectory

from ptcap.data.config_parser import YamlConfig


def check_obj_dict_and_dict_equal(config_obj, config_dict):
    pass


class ConfigParserTest(unittest.TestCase):

    def setUp(self):
        self.config_path = os.path.join(TempDirectory().path, "temp_config")
        config_contents = ("path: some_path\n\nnested:\n  type: some_type\n  "
                           "kwargs: {kwarg_1: 1, kwarg_2: kwarg2}")
        self.config_dict = {
            "path": "some_path",
            "nested": {"type": "some_type",
                       "kwargs": {"kwarg_1": 1, "kwarg_2": "kwarg2"}}}
        with open(self.config_path, "w") as f:
            f.write(config_contents)

    def test_parse_config_file(self):
        config_obj = YamlConfig(self.config_path)
        config_dict = config_obj.config_dict
        self.assertEqual(config_dict, self.config_dict)

    def test_get_values_from_config_parser(self):
        config_obj = YamlConfig(self.config_path)
        check_obj_dict_and_dict_equal(config_obj, self.config_dict)

    def test_save_config_file(self):
        temp_dir = TempDirectory().path
        temp_file = "temp_config2"
        config_obj = YamlConfig(self.config_path)
        config_obj.save(temp_dir, temp_file)
        config_obj.parse(os.path.join(temp_dir, temp_file))
        self.assertEqual(config_obj.config_dict, self.config_dict)

    def tearDown(self):
        shutil.rmtree(self.config_path)
