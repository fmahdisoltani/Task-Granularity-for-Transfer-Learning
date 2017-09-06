import torch
import os
import unittest

from testfixtures import tempdir

from ptcap.checkpointers import Checkpointer
from ptcap.model.mappers import FullyConnectedMapper

class CheckpointerTests(unittest.TestCase):

    def setUp(self):
        model = FullyConnectedMapper(1,1)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
        self.state_dict = {
            "epoch": 0,
            "model": model.state_dict(),
            "best_score": 5,
            "optimizer": optimizer.state_dict(),
        }

    def test_load_model(self):
        pass

    @tempdir()
    def test_save_best_higher_is_better(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path, higher_is_better=True)
        checkpointer.best_score = self.state_dict["best_score"]
        scores = [3, 7, 2, 11]
        for epoch_num, score in enumerate(scores):
            self.state_dict["epoch"] = epoch_num + 1
            best_score = self.state_dict["best_score"]
            with self.subTest(state_dict=self.state_dict, epoch_num=epoch_num,
                              score=score, best_score=best_score):
                checkpointer.save_best(self.state_dict, score, temp_dir.path)
                self.state_dict = torch.load(os.path.join(temp_dir.path,
                                                          "model.best"))
                if score > best_score:
                    self.assertEqual(self.state_dict["best_score"], score)
                else:
                    self.assertEqual(self.state_dict["best_score"], best_score)

    def test_save_best_folder_None(self):
        pass

    def test_save_latest(self):
        pass

    def test_save_latest_folder_None(self):
        pass

    def test_save_meta(self):
        pass

    def test_save_value_csv(self):
        pass