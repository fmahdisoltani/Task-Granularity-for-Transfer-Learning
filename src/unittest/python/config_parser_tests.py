import os
import shutil
import unittest

from testfixtures import TempDirectory

from ptcap.data.config_parser import YamlConfig


class ConfigParserTest(unittest.TestCase):

    def setUp(self):
        self.config_path = os.path.join(TempDirectory().path, "temp_config")
        config_contents = ("path: some_path\n\nnested:\n  type: some_type\n  "
                           "kwargs: {kwarg_1: 1, kwarg_2: kwarg2}")
        with open(self.config_path, "w") as f:
            f.write(config_contents)

    def test_parse_config_file(self):
        config_obj = YamlConfig(self.config_path)
        config_dict = config_obj.config_dict
        self.assertEqual(config_dict["path"], "some_path")
        self.assertEqual(config_dict["nested"]["type"], "some_type")
        self.assertEqual(config_dict["nested"]["kwargs"]["kwarg_1"], 1)
        self.assertEqual(config_dict["nested"]["kwargs"]["kwarg_2"], "kwarg2")

    def test_get_values_from_config_parser(self):
        config_obj = YamlConfig(self.config_path)
        self.assertEqual(config_obj.get("path"), "some_path")
        self.assertEqual(config_obj.get("nested", "type"), "some_type")
        self.assertEqual(config_obj.get("nested", "kwargs", "kwarg_1"), 1)
        self.assertEqual(
            config_obj.get("nested", "kwargs", "kwarg_2"), "kwarg2")

    def tearDown(self):
        shutil.rmtree(self.config_path)
