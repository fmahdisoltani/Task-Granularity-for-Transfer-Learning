import numpy as np
import unittest

import torch

from collections import OrderedDict

from ptcap.metrics import Metrics


class MetricTests(unittest.TestCase):

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

            for count in range(self.num_epochs):
                self.input_dict["loss"] = value
                Metrics.moving_average(self.input_dict, count + 1)
                self.assertEqual(self.input_dict["average_loss"], value)

    def test_run_metrics(self):
        metrics = Metrics(OrderedDict([("add1", lambda x: x + 1)]))
        metrics.run_metrics([(1,)])
        self.assertEqual(metrics.metrics_dict["add1"], 2)

    def test_compute_metrics(self):
        metrics = Metrics(OrderedDict([("add1", lambda x: x + 1)]))
        for count in range(self.num_epochs):
            metrics.compute_metrics([(1,)], count + 1)
            self.assertEqual(metrics.metrics_dict["add1"], 2)
            self.assertEqual(metrics.metrics_dict["average_add1"], 2)


class TestTokenLevelAccuracy(unittest.TestCase):

    def test_all_elements_match(self):
        captions = torch.LongTensor([[1,2,3],[4,5,6]])
        predictions = torch.LongTensor([[1,2,3],[4,5,6]])
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                accuracy = Metrics.token_level_accuracy(captions, predictions,
                                                        num)
                self.assertEqual(accuracy, 100)

    def test_no_elements_match(self):
        captions = torch.LongTensor([[1,2,3],[4,5,6]])
        predictions = torch.LongTensor([[7,8,9],[10,11,12]])
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                accuracy = Metrics.token_level_accuracy(captions, predictions,
                                                        num)
                self.assertEqual(accuracy, 0)

    def test_some_elements_match(self):
        captions = torch.LongTensor([[1,2,3],[4,5,6]])
        predictions = torch.LongTensor([[1,20,30],[40,5,6]])
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                accuracy = Metrics.token_level_accuracy(captions, predictions,
                                                        num)
                self.assertEqual(accuracy, 50)

    def test_no_elements(self):
        captions = torch.LongTensor([])
        predictions = torch.LongTensor([])
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                with self.assertRaises(IndexError):
                    accuracy = Metrics.token_level_accuracy(captions,
                                                            predictions, num)