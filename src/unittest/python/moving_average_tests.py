import numpy as np
import unittest

from collections import OrderedDict

from ptcap.metrics import Metrics

class MovingAverageTests(unittest.TestCase):

    def setUp(self):
        self.input_dict = OrderedDict()
        self.input_dict["average_loss"] = 0
        self.num_epochs = 3

    def test_input_normal(self):
        # Test when the moving average is getting random input
        input_list = []
        all_scores_dict = OrderedDict()

        for count in range(self.num_epochs):
            random_num = np.random.rand()
            self.input_dict["loss"] = random_num
            input_list.append(random_num)
            all_scores_dict = Metrics.moving_average(self.input_dict,
                                                     count + 1)
            expected = np.mean(input_list)

            self.assertAlmostEqual(self.input_dict["average_loss"],
                                   expected, 14)

    def test_input_same(self):
        # Test when the moving average is getting the same input (0 and 1 here)
        for value in range(2):
            all_scores_dict = OrderedDict()

            for count in range(self.num_epochs):
                self.input_dict["loss"] = value
                all_scores_dict = Metrics.moving_average(self.input_dict,
                                                         count + 1)
                self.assertEqual(self.input_dict["average_loss"], value)
