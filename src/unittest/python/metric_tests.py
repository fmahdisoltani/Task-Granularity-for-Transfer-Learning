import numpy as np
import unittest

import torch

from collections import OrderedDict

from torch.autograd import Variable

from ptcap.metrics import MetricsOperator
from ptcap.metrics import token_level_accuracy


class MetricTests(unittest.TestCase):

    def setUp(self):
        self.num_epochs = 3

    def test_input_normal(self):
        # Test when the moving average is getting random input
        metrics_operator = MetricsOperator(OrderedDict([("loss",
                                                         lambda x: x)]))
        input_list = []

        for count in range(self.num_epochs):
            random_num = np.random.rand()
            input_list.append(random_num)
            metrics_dict = metrics_operator.run_metrics(random_num)
            metrics_dict = metrics_operator.moving_average(metrics_dict,
                                                           count + 1)
            expected = np.mean(input_list)

            self.assertAlmostEqual(metrics_dict["average_loss"], expected, 14)

    def test_input_same(self):
        # Test when the moving average is getting the same input (0 and 1 here)
        metrics_operator = MetricsOperator(OrderedDict([("loss",
                                                         lambda x: x)]))

        for value in range(2):
            for count in range(self.num_epochs):
                metrics_dict = metrics_operator.run_metrics(value)
                metrics_dict = metrics_operator.moving_average(metrics_dict,
                                                               count + 1)
                self.assertEqual(metrics_dict["average_loss"], value)

    def test_run_metrics(self):
        metrics_operator = MetricsOperator(OrderedDict([("add1",
                                                         lambda x: x + 1)]))
        metrics_dict = metrics_operator.run_metrics(1)
        self.assertEqual(metrics_dict["add1"], 2)

    def test_compute_metrics(self):
        metrics_operator = MetricsOperator(OrderedDict([("add1",
                                                         lambda x: x + 1)]))
        for count in range(self.num_epochs):
            metrics_dict = metrics_operator.compute_metrics(1, count + 1)
            self.assertEqual(metrics_dict["add1"], 2)
            self.assertEqual(metrics_dict["average_add1"], 2)

    def test_get_average_metrics(self):
        metrics_operator = MetricsOperator(OrderedDict([("add1",
                                                         lambda x: x + 1)]))
        for count in range(self.num_epochs):
            metrics_operator.compute_metrics(1, count + 1)
        average_metrics = metrics_operator.get_average_metrics()
        self.assertEqual(list(average_metrics.keys()), ["average_add1"])


class TestTokenLevelAccuracy(unittest.TestCase):

    def test_all_elements_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                accuracy = token_level_accuracy(captions, predictions, num)
                self.assertEqual(accuracy, 100)

    def test_no_elements_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[7,8,9],[10,11,12]]))
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                accuracy = token_level_accuracy(captions, predictions, num)
                self.assertEqual(accuracy, 0)

    def test_some_elements_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[1,20,30],[40,5,6]]))
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                accuracy = token_level_accuracy(captions, predictions, num)
                self.assertEqual(accuracy, 50)

    def test_no_elements(self):
        captions = Variable(torch.LongTensor([]))
        predictions = Variable(torch.LongTensor([]))
        for num in (1, None):
            with self.subTest(captions=captions, predictions=predictions,
                              num=num):
                with self.assertRaises(IndexError):
                    accuracy = token_level_accuracy(captions, predictions, num)
