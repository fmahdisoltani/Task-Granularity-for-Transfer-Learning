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

    @tempdir()
    def test_get_key_from_config_dict(self, temp_dir):
