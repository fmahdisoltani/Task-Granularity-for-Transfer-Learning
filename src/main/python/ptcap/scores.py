import torch

from collections import OrderedDict


def token_accuracy(outputs, num_tokens=None):
    return token_level_accuracy(outputs.captions, outputs.predictions,
                                num_tokens)


def loss_to_numpy(score_attr):
    return score_attr.loss.data.cpu().numpy()[0]


def first_token_accuracy(outputs):
    return token_accuracy(outputs, 1)


def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(
        predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
    return accuracy


def caption_accuracy(outputs):
    return caption_level_accuracy(outputs.captions, outputs.predictions)


def caption_level_accuracy(captions, predictions):
    _, caption_len = captions.size()
    equal_values = torch.sum(captions.eq(predictions), dim=1)
    accuracy = equal_values.eq(caption_len).float().mean().data.numpy()[0] * 100.0
    return accuracy


def classif_accuracy(outputs):
    _, class_index = torch.max(outputs.classif_probs, dim=1)
    equal_values = torch.mean(class_index.eq(outputs.classif_targets).float())
    accuracy = equal_values.float().data.numpy()[0] * 100.0
    return accuracy


class ScoresOperator(object):
    def __init__(self, functions_dict):
        """
            Initializes scores_dict and functions_dict.
        Args:
            functions_dict: An OrderedDict whose keys are strings and values are
                the functions that will be applied.
        """

        self.functions_dict = functions_dict
        self.scores_dict = OrderedDict({"avg_" + score: 0 for score in
                            self.functions_dict})

    def compute_scores(self, score_attr, count):
        """
            Computes all the scores provided by the functions_dict in __init__.
        Args:
            parameters_list: A list of arguments to be passed to be computed by
                functions_dict and stored in scores_dict.
            count: An int indicating the number of iterations.
        Returns:
            An OrderedDict containing the most recent scores as well as their
            moving average.
        """

        scores_dict = self.run_scores(score_attr)
        # Calculate and update the moving average of the scores.
        scores_dict = self.update_moving_average(scores_dict, count)
        return scores_dict

    def run_scores(self, score_attr):
        scores_dict = OrderedDict()
        for score, score_function in self.functions_dict.items():
            scores_dict[score] = score_function(score_attr)
        return scores_dict

    def get_average_scores(self):
        return {key: self.scores_dict[key] for key in self.scores_dict
                if "avg" in key}

    def update_moving_average(self, scores_dict, count):
        assert count > 0
        scores_dict = OrderedDict(scores_dict)
        scores_list = list(scores_dict.keys())
        for score in scores_list:
            average_score = "avg_" + score
            self.scores_dict[average_score] += (
                (scores_dict[score] - self.scores_dict[average_score])
                / count)
            scores_dict[average_score] = self.scores_dict[average_score]
        return scores_dict

############################################################################33

import sys

import numpy as np
import torch

from collections import OrderedDict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.fudge.fudge import Fudge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.metrics import MultiScorer
from pycocoevalcap.rouge.rouge import Rouge


def caption_accuracy(captions, predictions):
    _, caption_len = captions.size()
    equal_values = torch.sum(captions.eq(predictions), dim=1)
    mean_accuracy = equal_values.eq(caption_len).float().mean()
    accuracy = mean_accuracy.data.numpy()[0] * 100.0
    return accuracy


def token_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(
        predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
    return accuracy


def preprocess_name(name):
    """
    Preprocess the class name to have underscores between uppercase letters
    and then return it in lowercase.
    """
    prep_chars = ["_" + char if char.isupper() and i != 0 else char
                  for i, char in enumerate(name)]
    return "".join(prep_chars).lower()


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


class ScoreBase(object):
    @property
    def name(self):
        return preprocess_name(type(self).__name__)

    def __call__(self, outputs):
        raise NotImplementedError


class Accuracy(ScoreBase):
    def __call__(self, outputs):
        accuracy = token_accuracy(outputs.captions, outputs.predictions)
        return {self.name: accuracy}


class FirstAccuracy(ScoreBase):
    def __call__(self, outputs):
        first_token_accuracy = token_accuracy(outputs.captions,
                                              outputs.predictions, 1)
        return {self.name: first_token_accuracy}


class CaptionAccuracy(ScoreBase):
    def __call__(self, outputs):
        caption_accuracy_ = caption_accuracy(outputs.captions,
                                             outputs.predictions)
        return {self.name: caption_accuracy_}


class Loss(ScoreBase):
    def __call__(self, score_attr):
        return {self.name: score_attr.loss.data.cpu().numpy()[0]}


class LCSScoreBase(ScoreBase):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Fscore(LCSScoreBase):
    def __call__(self, precision, recall, beta=1):
        numerator = (1.0 + (beta ** 2)) * precision * recall
        denominator = ((beta ** 2) * precision) + recall
        return {self.name: safe_div(numerator, denominator)}


class Gmeasure(LCSScoreBase):
    def __call__(self, precision, recall):
        return {self.name: np.sqrt(precision * recall)}


class ScoresInterface(object):
    def __init__(self, metrics, tokenizer):
        self.metrics = metrics
        self.scores_operator = None
        self.scoring_functions = []
        self.tokenizer = tokenizer

    def get_score_operator(self):
        scores = []

        scores.extend(self._add_basic_scores())
        scores.extend(self._add_multiscorer())
        scores.extend(self._add_LCS())

        self.scores_operator = ScoresOperator(scores)

        return self.scores_operator

    def _add_basic_scores(self):
        scoring_objects = [scoring_object()
                           for scoring_object in ScoreBase.__subclasses__()
                           if scoring_object().name in self.metrics]

        relevant_scoring_objects = filter((lambda x: x.name in self.metrics),
                                          scoring_objects)

        return relevant_scoring_objects

    def _add_multiscorer(self):

        multiscore_adapter = []

        if "FUDGE" in self.metrics:
            multiscore_adapter = [MultiScoreAdapter(
                MultiScorer(aggregator=Fudge(), BLEU=Bleu(4), ROUGE_L=Rouge(),
                            METEOR=Meteor()), self.tokenizer)]

        else:
            multiscorers = [Bleu(4), Meteor(), Rouge()]

            relevant_multiscorers = list(filter((
                lambda x: x.method().upper() in self.metrics), multiscorers))

            if len(relevant_multiscorers) > 0:
                multiscorer_kwargs = list(map(lambda x: {x.method().upper(): x},
                                              relevant_multiscorers))[0]

                multiscore_adapter = [MultiScoreAdapter(
                    MultiScorer(**multiscorer_kwargs), self.tokenizer)]

        return multiscore_adapter

    def _add_LCS(self):

        lcs_scorers = [scoring_object()
                       for scoring_object in LCSScoreBase.__subclasses__()
                       if scoring_object().name in self.metrics]

        relevant_lcs_scorers = list(filter((lambda x: x.name in self.metrics),
                                           lcs_scorers))

        lcs_object = LCS(relevant_lcs_scorers, self.tokenizer)

        lcs_object_scorers = list(filter((lambda x: x in self.metrics),
                                         lcs_object.name))
        final_lcs_object = []
        if len(lcs_object_scorers) > 0:
            final_lcs_object = [LCS(relevant_lcs_scorers, self.tokenizer)]

        return final_lcs_object

    def get_average_scores(self):
        avg_scores = self.scores_operator.get_average_scores()
        scores = filter((lambda x: any(metric in x for metric in self.metrics)),
                        sorted(avg_scores.keys()))

        scores_dict = OrderedDict()
        for score in list(scores):
            scores_dict[score] = avg_scores[score]

        return scores_dict


class ScoresOperator(object):
    def __init__(self, objects_list):
        """
            Initializes scores_dict and functions_dict.
        Args:
            objects_list: A list of the functions that will be applied.
        """

        self.avg_keyword = "avg"
        self.objects_list = objects_list
        self.scores_dict = OrderedDict()

    def compute_scores(self, score_attr, count):
        """
            Computes all the scores provided by the functions_dict in __init__.
        Args:
            score_attr: The input passed as a NamedTuple to be computed by
                functions_dict and stored in scores_dict.
            count: An int indicating the number of iterations.
        Returns:
            An OrderedDict containing the most recent scores as well as their
            moving average.
        """

        scores_dict = self.run_scores(score_attr)
        # Calculate and update the moving average of the scores.
        scores_dict = self.update_moving_average(scores_dict, count)
        return scores_dict

    def run_scores(self, score_attr):
        scores_dict = OrderedDict()
        for score_function in self.objects_list:
            scores_dict.update(score_function(score_attr))
        return scores_dict

    def get_average_scores(self):
        return {key: self.scores_dict[key] for key in self.scores_dict
                if self.avg_keyword in key}

    def update_moving_average(self, scores_dict, count):
        assert count > 0
        scores_dict = OrderedDict(scores_dict)
        scores_list = list(scores_dict.keys())
        for score in scores_list:
            average_score = self.avg_keyword + "_" + score
            total_score = self.scores_dict.get(average_score, 0) * (count - 1)
            self.scores_dict[average_score] = (
                (scores_dict[score] + total_score) / count)
            scores_dict[average_score] = self.scores_dict[average_score]
        return scores_dict


class LCS(object):
    """
    The main functionality of this class is to compute the LCS (Lowest Common
    Subsequence) between a caption and prediction. By default, it returns the
    precision and recall values calculated based on the LCS between a prediction
    and a caption.
    """
    def __init__(self, objects_list, tokenizer):
        """
        Initializes functions_list and tokenizer.
        Args:
        functions_list: A list of the functions that will be applied on the
        precision and recall values calculated based on the LCS between a
        prediction and a caption.
        """

        self.name = [o.name for o in objects_list] + ["precision", "recall"]
        self.objects_list = objects_list
        self.scores_container = OrderedDict()
        self.scores_dict = OrderedDict()
        self.tokenizer = tokenizer

    def __call__(self, outputs):
        string_predictions = [self.tokenizer.get_string(str_pred.data.numpy())
                              for str_pred in outputs.predictions]
        return self.score_batch(string_predictions, outputs.string_captions)

    def collect_scores(self, batch_scores_dict, scores_dict):
        for metric, metric_value in scores_dict.items():
            if metric not in batch_scores_dict:
                batch_scores_dict[metric] = [metric_value]
            else:
                batch_scores_dict[metric].append(metric_value)
        return batch_scores_dict

    @classmethod
    def compute_lcs(cls, prediction, caption):
        num_rows = len(prediction)
        num_cols = len(caption)

        table = [[0] * (num_cols + 1) for _ in range(num_rows + 1)]
        for i in range(1, num_rows + 1):
            for j in range(1, num_cols + 1):
                if prediction[i - 1] == caption[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i][j - 1], table[i - 1][j])
        return table, table[num_rows][num_cols]

    def mean_scores(self, batch_scores_dict):
        for metric, metric_value in batch_scores_dict.items():
            batch_scores_dict[metric] = np.mean(metric_value)
        return batch_scores_dict

    def score_batch(self, predictions, captions):
        assert len(predictions) == len(captions)

        batch_scores_dict = OrderedDict()
        for count, (prediction, caption) in enumerate(zip(predictions,
                                                          captions)):
            scores_dict = self.score_sample(prediction.split(), caption.split())
            batch_scores_dict = self.collect_scores(batch_scores_dict,
                                                    scores_dict)

        batch_scores_dict = self.mean_scores(batch_scores_dict)
        return batch_scores_dict

    def score_sample(self, prediction, caption):
        scores_dict = OrderedDict()
        _, lcs_score = self.compute_lcs(prediction, caption)
        scores_dict["precision"] = safe_div(lcs_score, len(prediction))
        scores_dict["recall"] = safe_div(lcs_score, len(caption))

        for score_function in self.objects_list:
            scores_dict.update(score_function(scores_dict["precision"],
                                              scores_dict["recall"]))

        return scores_dict


class MultiScoreAdapter(object):
    def __init__(self, multiscorer, tokenizer):
        self.multiscorer = multiscorer
        self.tokenizer = tokenizer

    def __call__(self, score_attr):
        return self.multiscore(score_attr)

    def multiscore(self, outputs):
        string_predictions = [self.tokenizer.get_string(str_pred.data.numpy())
                              for str_pred in outputs.predictions]

        return self.multiscorer.score(outputs.string_captions,
                                      string_predictions)
