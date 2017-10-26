import numpy as np
import unittest

import torch

from torch.autograd import Variable

from ptcap.data.tokenizer import Tokenizer
from ptcap.scores import (LCS, ScoresOperator, fscore, gmeasure,
                          token_level_accuracy, caption_level_accuracy)


class ScoreTests(unittest.TestCase):

    def setUp(self):
        self.num_epochs = 3

    def test_input_normal(self):
        """
            Test when the moving average is getting random input.
        """
        scores_operator = ScoresOperator([lambda x: {"loss": x},
                                          lambda x: {"add1": x + 1}])
        input_list = []

        for count in range(self.num_epochs):
            random_num = np.random.rand()
            input_list.append(random_num)
            scores_dict = scores_operator.run_scores(random_num)
            scores_dict = scores_operator.update_moving_average(scores_dict,
                                                                count + 1)
            expected = np.mean(input_list)

            self.assertAlmostEqual(scores_dict["avg_loss"], expected, 14)

    def test_input_same(self):
        """
            Test when the moving average is getting the same input (0 and 1
            here).
        """
        scores_operator = ScoresOperator([lambda x: {"loss": x},
                                          lambda x: {"add1": x + 1}])

        for value in range(2):
            for count in range(self.num_epochs):
                scores_dict = scores_operator.run_scores(value)
                scores_dict = scores_operator.update_moving_average(scores_dict,
                                                                    count + 1)
                self.assertEqual(scores_dict["avg_loss"], value)

    def test_run_scores(self):
        scores_operator = ScoresOperator([lambda x: {"add1": x + 1}])
        scores_dict = scores_operator.run_scores(1)
        self.assertEqual(scores_dict["add1"], 2)

    def test_compute_scores(self):
        scores_operator = ScoresOperator([lambda x: {"add1": x + 1}])
        for count in range(self.num_epochs):
            scores_dict = scores_operator.compute_scores(1, count + 1)
            self.assertEqual(scores_dict["add1"], 2)
            self.assertEqual(scores_dict["avg_add1"], 2)

    def test_get_average_scores(self):
        scores_operator = ScoresOperator([lambda x: {"add1": x + 1}])
        for count in range(self.num_epochs):
            scores_operator.compute_scores(1, count + 1)
        average_scores = scores_operator.get_average_scores()
        self.assertEqual(list(average_scores.keys()), ["avg_add1"])


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


class TestLCS(unittest.TestCase):
    def setUp(self):
        self.captions = ["A B C", "B B B", "A B C", "A B C", "A B C", "A B C"]
        self.predictions = ["",   "A A A", "A A A", "A B B", "A B",   "A B C"]

        def test_function(x, y): return {"add": x + y}
        self.test_function = test_function
        self.lcs_obj = LCS([self.test_function], Tokenizer())

    def test_compute_lcs(self):

        expected_lcs = [0, 0, 1, 2, 2, 3]

        for i, (prediction, caption) in enumerate(zip(self.predictions,
                                                      self.captions)):
            with self.subTest(captions=self.captions[i],
                              predictions=self.predictions[i]):
                _, computed_lcs = LCS.compute_lcs(prediction.split(),
                                                  caption.split())
                self.assertEqual(computed_lcs, expected_lcs[i])

    def test_run_scores(self):

        expected_precision = [0, 0, 1.0/3, 2.0/3, 1, 1]
        expected_recall = [0, 0, 1.0/3, 2.0/3, 2.0/3, 1]

        for i, (prediction, caption) in enumerate(zip(self.predictions,
                                                      self.captions)):
            with self.subTest(captions=self.captions[i],
                              predictions=self.predictions[i]):
                scores_dict = self.lcs_obj.run_scores(prediction.split(),
                                                 caption.split())
                self.assertAlmostEqual(scores_dict["precision"],
                                       expected_precision[i], 8)
                self.assertAlmostEqual(scores_dict["recall"],
                                       expected_recall[i], 8)
                self.assertAlmostEqual(scores_dict["add"], self.test_function(
                    expected_precision[i], expected_recall[i])["add"], 8)

    def test_score(self):

        expected_batch_precision = 0.5
        expected_batch_recall = 4.0/9
        expected_sum = self.test_function(expected_batch_precision,
                                          expected_batch_recall)["add"]

        scores_dict = self.lcs_obj.score(self.predictions, self.captions)

        self.assertAlmostEqual(scores_dict["precision"],
                               expected_batch_precision, 8)
        self.assertAlmostEqual(scores_dict["recall"],
                               expected_batch_recall, 8)
        self.assertAlmostEqual(scores_dict["add"], expected_sum, 8)


class TestFscore(unittest.TestCase):
    def test_different_precision_and_recall(self):
        precision = [0, 0, 1.0/3, 1.0/3, 1, 1, 1]
        recall = [0, 1, 1, 0.2, 0.2, 1, 0]
        expected_fscore = [0, 0, 0.5, 0.25, 1.0/3, 1, 0]

        for i, (p, r) in enumerate(zip(precision, recall)):
            with self.subTest(precision=p, recall=r):
                self.assertAlmostEqual(fscore(p, r)["fscore"],
                                       expected_fscore[i], 8)

    def test_different_beta(self):
        beta_vals = [1, 2, 3]
        precision = 1.0/3
        recall = 0.2

        expected_fscores = [0.25, 5.0/23, 5.0/24]

        for i, beta in enumerate(beta_vals):
            with self.subTest(precision=precision, recall=recall, beta=beta):
                self.assertAlmostEqual(fscore(precision, recall, beta)["fscore"]
                                       , expected_fscores[i], 8)


class TestGmeasure(unittest.TestCase):
    def test_different_precision_and_recall(self):
        precision = [0, 0, 0.25, 0.1, 1, 1]
        recall = [0, 1, 1, 0.1, 0.01, 1]

        expected_gmeasure = [0, 0, 0.5, 0.1, 0.1, 1]

        for i, (p, r) in enumerate(zip(precision, recall)):
            with self.subTest(precision=p, recall=r):
                self.assertEqual(gmeasure(p, r)["gmeasure"],
                                 expected_gmeasure[i])
