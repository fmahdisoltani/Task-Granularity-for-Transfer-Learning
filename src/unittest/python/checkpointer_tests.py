import csv
import os
import unittest

import numpy as np
import torch

from unittest.mock import patch

from testfixtures import tempdir

from ptcap.checkpointers import Checkpointer
from ptcap.data.config_parser import YamlConfig
from ptcap.data.tokenizer import Tokenizer
from ptcap.model.mappers import FullyConnectedMapper


class CheckpointerTests(unittest.TestCase):

    def setUp(self):
        self.config_obj = YamlConfig(config_dict={"1": 1})
        self.epoch_num = 0
        self.model = FullyConnectedMapper(1,1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.score = None
        self.state_dict = {
            "epoch": self.epoch_num,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "score": self.score,
        }
        self.tokenizer = Tokenizer()
        self.tokenizer.build_dictionaries(["Just a dummy caption"])

    def check_loaded_objects(self, epoch_num, model, optimizer):
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)

    @tempdir()
    def test_set_best_score(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        scores = [None, 13.0, 17.0, 11.0, 23.0]
        expected = [np.Inf, 13.0, 13.0, 11.0, 11.0]
        for i, score in enumerate(scores):
            with self.subTest(score=score, expected=expected[i]):
                checkpointer.set_best_score(score)
                self.assertEqual(checkpointer.best_score, expected[i])

    @patch.object(torch, "load", autospec=True)
    @tempdir()
    def test_load_model_from_save_latest(self, mock_torch_load, temp_dir):
        mock_torch_load.return_value = self.state_dict
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.save_latest(self.state_dict)
        epoch_num, model, optimizer = checkpointer.load_model(
                                                self.model, self.optimizer,
                                                temp_dir.path, "model.latest")
        self.assertTrue(mock_torch_load.call_count == 1)
        self.check_loaded_objects(epoch_num, model, optimizer)

    @patch.object(torch, "load", autospec=True)
    @tempdir()
    def test_load_model_from_save_best(self, mock_torch_load, temp_dir):
        self.state_dict["score"] = 5.0
        mock_torch_load.return_value = self.state_dict
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.best_score = self.state_dict["score"] + 1
        checkpointer.save_best(self.state_dict)
        epoch_num, model, optimizer = checkpointer.load_model(
                                                    self.model, self.optimizer,
                                                    temp_dir.path, "model.best")
        self.assertTrue(mock_torch_load.call_count == 1)
        self.check_loaded_objects(epoch_num, model, optimizer)

    @patch.object(torch, "load")
    @tempdir()
    def test_load_model_from_no_checkpoint(self, mock_torch_load, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        epoch_num, model, optimizer = checkpointer.load_model(
                                                    self.model, self.optimizer,
                                                    temp_dir.path, "model.best")
        mock_torch_load.assert_not_called()
        self.check_loaded_objects(epoch_num, model, optimizer)

    @patch.object(torch, "load")
    @tempdir()
    def test_load_model_from_no_predefined_folder(self, mock_torch_load, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        epoch_num, model, optimizer = checkpointer.load_model(
            self.model, self.optimizer)
        mock_torch_load.assert_not_called()
        self.check_loaded_objects(epoch_num, model, optimizer)

    @patch.object(torch, "load")
    @tempdir()
    def test_load_model_from_predefined_folder(self, mock_torch_load, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        epoch_num, model, optimizer = checkpointer.load_model(
            self.model, self.optimizer, temp_dir.path)
        mock_torch_load.assert_not_called()
        self.check_loaded_objects(epoch_num, model, optimizer)

    @patch.object(torch, "load")
    @tempdir()
    def test_load_model_from_predefined_file(self, mock_torch_load, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        epoch_num, model, optimizer = checkpointer.load_model(
            self.model, self.optimizer, filename="model.latest")
        mock_torch_load.assert_not_called()
        self.check_loaded_objects(epoch_num, model, optimizer)

    @tempdir()
    def test_save_best_higher_is_better(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path, higher_is_better=True)
        scores = [3.0, 7.0, 2.0, 7.0, 11.0]
        best_epoch = [1, 2, 2, 2, 5]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=checkpointer.best_score):
                checkpointer.save_best(self.state_dict, temp_dir.path)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                self.assertEqual(self.state_dict["score"],
                                 checkpointer.best_score)
                self.assertEqual(self.state_dict["epoch"],
                                 best_epoch[epoch_num])

    @tempdir()
    def test_save_best_higher_is_not_better(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path, higher_is_better=False)
        scores = [3.0, 7.0, 3.0, 2.0, 11.0]
        best_epoch = [1, 1, 1, 4, 4]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=checkpointer.best_score):
                checkpointer.save_best(self.state_dict, temp_dir.path)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                self.assertEqual(self.state_dict["score"],
                                 checkpointer.best_score)
                self.assertEqual(self.state_dict["epoch"],
                                 best_epoch[epoch_num])

    @tempdir()
    def test_save_best_folder_None(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        scores = [3.0, 7.0, 2.0, 11.0]
        best_epoch = [1, 1, 3, 3]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=checkpointer.best_score):
                checkpointer.save_best(self.state_dict)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                self.assertEqual(self.state_dict["score"],
                                 checkpointer.best_score)
                self.assertEqual(self.state_dict["epoch"],
                                 best_epoch[epoch_num])

    @tempdir()
    def test_save_latest(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        scores = [3.0, 7.0, 2.0, 11.0]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score):
                checkpointer.save_latest(self.state_dict, temp_dir.path)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.latest"))
                self.assertEqual(self.state_dict["score"], score)
                self.assertEqual(self.state_dict["epoch"], epoch_num + 1)

    @tempdir()
    def test_save_latest_folder_None(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        scores = [3.0, 7.0, 2.0, 11.0]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score):
                checkpointer.save_latest(self.state_dict)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.latest"))
                self.assertEqual(self.state_dict["score"], score)
                self.assertEqual(self.state_dict["epoch"], epoch_num + 1)

    @tempdir()
    def test_save_meta(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.save_meta(temp_dir.path, self.config_obj, self.tokenizer)
        self.assertEqual(len(os.listdir(temp_dir.path)), 2)

    @tempdir()
    def test_save_meta_folder_does_not_exist(self, temp_dir):
        random_folder = temp_dir.path + "random_folder"
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.save_meta(random_folder, self.config_obj, self.tokenizer)
        self.assertEqual(len(os.listdir(random_folder)), 2)

    @tempdir()
    def test_save_value_csv(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        value = ["1","2","3"]
        filename = "checkpointer_test"
        checkpointer.save_value_csv(value, temp_dir.path, filename)
        read_value = []
        with open(os.path.join(temp_dir.path, filename)) as csvfile:
            read_csv_file = csv.reader(csvfile)
            for row in read_csv_file:
                read_value = row
        self.assertEqual(value, read_value)

    @tempdir()
    def test_save_value_csv_no_folder(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        value = ["1","2","3"]
        filename = "checkpointer_test"
        checkpointer.save_value_csv(value, filename=filename)
        read_value = []
        with open(os.path.join(temp_dir.path, filename)) as csvfile:
            read_csv_file = csv.reader(csvfile)
            for row in read_csv_file:
                read_value = row
        self.assertEqual(value, read_value)

    @tempdir()
    def test_save_best_after_loading_from_latest(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.save_latest(self.state_dict)
        checkpointer.load_model(self.model, self.optimizer, temp_dir.path,
                                "model.latest")
        self.state_dict["score"] = 3.0
        self.state_dict["epoch"] = 1
        checkpointer.save_best(self.state_dict)
        self.assertEqual(self.state_dict["score"], checkpointer.best_score)
        self.assertEqual(self.state_dict["epoch"], 1)
