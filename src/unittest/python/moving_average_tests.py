import numpy as np
import unittest

from collections import OrderedDict

from ptcap.trainers import Trainer

class MovingAverageTests(unittest.TestCase):

    def setUp(self):
        self.input_dict = OrderedDict()
        self.num_epochs = 3

    def test_input_normal(self):
        # Test when the moving average is getting random input
        input_list = []
        all_scores_dict = OrderedDict()

        for count in range(self.num_epochs):
            random_num = np.random.rand()
            self.input_dict["loss"] = random_num
            input_list.append(random_num)
            all_scores_dict = Trainer.moving_average(self.input_dict,
                                                     all_scores_dict,
                                                     count + 1)
            expected = np.mean(input_list)
            self.assertEqual(all_scores_dict["loss"], self.input_dict["loss"])
            self.assertAlmostEqual(all_scores_dict["average_loss"],
                                   expected, 14)

    def test_input_same(self):
        # Test when the moving average is getting the same input (0 and 1 here)
        for value in range(2):
            all_scores_dict = OrderedDict()

            for count in range(self.num_epochs):
                self.input_dict["loss"] = value
                all_scores_dict = Trainer.moving_average(self.input_dict,
                                                         all_scores_dict,
                                                         count + 1)
                self.assertEqual(all_scores_dict["loss"], value)
                self.assertEqual(all_scores_dict["average_loss"], value)
