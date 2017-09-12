import numpy as np
import unittest

import torch

from collections import OrderedDict

from torch.autograd import Variable

from ptcap.scores import (ScoresOperator,
                          token_level_accuracy, caption_level_accuracy)


class ScoreTests(unittest.TestCase):

    def setUp(self):
        self.num_epochs = 3

    def test_input_normal(self):
        """
            Test when the moving average is getting random input.
        """
        scores_operator = ScoresOperator(
            OrderedDict([("loss", lambda x: x), ("Add1", lambda x: x + 1)]))
        input_list = []

        for count in range(self.num_epochs):
            random_num = np.random.rand()
            input_list.append(random_num)
            scores_dict = scores_operator.run_scores(random_num)
            scores_dict = scores_operator.update_moving_average(scores_dict,
                                                           count + 1)
            expected = np.mean(input_list)

            self.assertAlmostEqual(scores_dict["average_loss"], expected, 14)

    def test_input_same(self):
        """
            Test when the moving average is getting the same input (0 and 1
            here).
        """
        scores_operator = ScoresOperator(
            OrderedDict([("loss", lambda x: x), ("Add1", lambda x: x + 1)]))

        for value in range(2):
            for count in range(self.num_epochs):
                scores_dict = scores_operator.run_scores(value)
                scores_dict = scores_operator.update_moving_average(scores_dict,
                                                               count + 1)
                self.assertEqual(scores_dict["average_loss"], value)

    def test_run_scores(self):
        scores_operator = ScoresOperator(OrderedDict([("add1",
                                                         lambda x: x + 1)]))
        scores_dict = scores_operator.run_scores(1)
        self.assertEqual(scores_dict["add1"], 2)

    def test_compute_scores(self):
        scores_operator = ScoresOperator(OrderedDict([("add1",
                                                         lambda x: x + 1)]))
        for count in range(self.num_epochs):
            scores_dict = scores_operator.compute_scores(1, count + 1)
            self.assertEqual(scores_dict["add1"], 2)
            self.assertEqual(scores_dict["average_add1"], 2)

    def test_get_average_scores(self):
        scores_operator = ScoresOperator(OrderedDict([("add1",
                                                         lambda x: x + 1)]))
        for count in range(self.num_epochs):
            scores_operator.compute_scores(1, count + 1)
        average_scores = scores_operator.get_average_scores()
        self.assertEqual(list(average_scores.keys()), ["average_add1"])


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


class TestCaptionLevelAccuracy(unittest.TestCase):

    def test_all_captions_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))

        accuracy = caption_level_accuracy(captions, predictions)
        self.assertEqual(accuracy, 100)

    def test_no_captions_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[7,8,9],[10,11,12]]))

        accuracy = caption_level_accuracy(captions, predictions)
        self.assertEqual(accuracy, 0)

    def test_some_elements_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[1,20,30],[40,5,6]]))

        accuracy = caption_level_accuracy(captions, predictions)
        self.assertEqual(accuracy, 0)

    def test_no_captions(self):
        captions = Variable(torch.LongTensor([]))
        predictions = Variable(torch.LongTensor([]))

        with self.subTest(captions=captions, predictions=predictions):
            with self.assertRaises(ValueError):
                accuracy = caption_level_accuracy(captions, predictions)

    def test_some_captions_match(self):
        captions = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
        predictions = Variable(torch.LongTensor([[1,20,30],[4,5,6]]))

        accuracy = caption_level_accuracy(captions, predictions)
        self.assertEqual(accuracy, 50)