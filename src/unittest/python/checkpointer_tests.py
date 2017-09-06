import csv
import torch
import os
import unittest

from testfixtures import tempdir

from ptcap.checkpointers import Checkpointer
from ptcap.data.config_parser import YamlConfig
from ptcap.data.tokenizer import Tokenizer
from ptcap.model.mappers import FullyConnectedMapper


class CheckpointerTests(unittest.TestCase):

    def setUp(self):
        self.epoch_num = 0
        self.model = FullyConnectedMapper(1,1)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                          lr=0.01)
        self.score = 5
        self.state_dict = {
            "epoch": self.epoch_num,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "score": self.score,
        }

    @tempdir()
    def test_load_model_from_save_latest(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.best_score = self.score - 1
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        checkpointer.save_latest(self.state_dict)
        epoch_num, model, optimizer, tokenizer_obj = checkpointer.load_model(
                                        self.model, self.optimizer, tokenizer,
                                        temp_dir.path, "model.latest")
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)
        self.assertEqual(tokenizer.caption_dict, tokenizer_obj.caption_dict)

    @tempdir()
    def test_load_model_from_save_best(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.best_score = self.score + 1
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        checkpointer.save_best(self.state_dict)
        epoch_num, model, optimizer, tokenizer_obj = checkpointer.load_model(
            self.model, self.optimizer, tokenizer, temp_dir.path, "model.best")
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)
        self.assertEqual(tokenizer.caption_dict, tokenizer_obj.caption_dict)

    @tempdir()
    def test_load_model_from_no_checkpoint(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.best_score = self.score - 1
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        checkpointer.save_best(self.state_dict)
        epoch_num, model, optimizer, tokenizer_obj = checkpointer.load_model(
            self.model, self.optimizer, tokenizer, temp_dir.path, "model.best")
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)
        self.assertEqual(tokenizer.caption_dict, tokenizer_obj.caption_dict)

    @tempdir()
    def test_load_model_from_no_predefined_folder(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        checkpointer.save_latest(self.state_dict)
        epoch_num, model, optimizer, tokenizer_obj = checkpointer.load_model(
            self.model, self.optimizer, tokenizer)
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)
        self.assertEqual(tokenizer, tokenizer_obj)

    @tempdir()
    def test_load_model_from_predefined_folder(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        checkpointer.save_latest(self.state_dict)
        epoch_num, model, optimizer, tokenizer_obj = checkpointer.load_model(
            self.model, self.optimizer, tokenizer, temp_dir.path)
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)
        self.assertEqual(tokenizer, tokenizer_obj)

    @tempdir()
    def test_load_model_from_predefined_file(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        checkpointer.save_latest(self.state_dict)
        epoch_num, model, optimizer, tokenizer_obj = checkpointer.load_model(
            self.model, self.optimizer, tokenizer, filename="model.latest")
        self.assertEqual(epoch_num, self.epoch_num)
        self.assertEqual(model, self.model)
        self.assertEqual(optimizer, self.optimizer)
        self.assertEqual(tokenizer, tokenizer_obj)

    @tempdir()
    def test_save_best_higher_is_better(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path, higher_is_better=True)
        checkpointer.save_best(self.state_dict, temp_dir.path)
        scores = [3, 7, 2, 11]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            best_score = checkpointer.best_score
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=best_score):
                checkpointer.save_best(self.state_dict, temp_dir.path)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                if score > best_score:
                    self.assertEqual(self.state_dict["score"], score)

    @tempdir()
    def test_save_best_higher_is_not_better(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path, higher_is_better=False)
        checkpointer.best_score = self.state_dict["score"]
        scores = [3, 7, 2, 11]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            best_score = checkpointer.best_score
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=best_score):
                checkpointer.save_best(self.state_dict, temp_dir.path)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                if score < best_score:
                    self.assertEqual(self.state_dict["score"], score)

    @tempdir()
    def test_save_best_folder_None(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.best_score = self.state_dict["score"]
        scores = [3, 7, 2, 11]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            best_score = checkpointer.best_score
            self.state_dict["score"] = score
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=best_score):
                checkpointer.save_best(self.state_dict)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                if score < best_score:
                    self.assertEqual(self.state_dict["score"], score)

    @tempdir()
    def test_save_latest(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.best_score = self.state_dict["score"]
        scores = [3, 7, 2, 11]
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
        checkpointer.best_score = self.state_dict["score"]
        scores = [3, 7, 2, 11]
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
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(temp_dir.path, config_obj, tokenizer)
        self.assertEqual(len(os.listdir(temp_dir.path)), 2)

    @tempdir()
    def test_save_meta_folder_does_not_exist(self, temp_dir):
        random_folder = temp_dir.path + "random_folder"
        checkpointer = Checkpointer(temp_dir.path)
        config_obj = YamlConfig(config_dict={"1": 1})
        tokenizer = Tokenizer()
        tokenizer.build_dictionaries(["Just a dummy caption"])
        checkpointer.save_meta(random_folder, config_obj, tokenizer)
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
