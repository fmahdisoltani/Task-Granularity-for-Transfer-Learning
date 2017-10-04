import os
import unittest

from testfixtures import tempdir

from ptcap.data.config_parser import YamlConfig


class ConfigParserTest(unittest.TestCase):

    def setUp(self):
        config_path = "src/main/configs/config_parser.yaml"
        path = os.path.join(os.getcwd(), config_path)
        self.config_parser = YamlConfig(path)

    def test_config_dict(self):
        config_dict = self.config_parser.config_dict
        self.assertEqual(config_dict["path"], "some_path")
        self.assertEqual(config_dict["nested"]["type"], "some_type")
        self.assertEqual(config_dict["nested"]["kwargs"]["kwarg_1"], 1)
        self.assertEqual(config_dict["nested"]["kwargs"]["kwarg_2"], "kwarg2")

    def test_get_values_from_config_parser(self):
        self.assertEqual(self.config_parser.get("path"), "some_path")
        self.assertEqual(self.config_parser.get("nested", "type"), "some_type")
        self.assertEqual(
            self.config_parser.get("nested", "kwargs", "kwarg_1"), 1)
        self.assertEqual(
            self.config_parser.get("nested", "kwargs", "kwarg_2"), "kwarg2")
