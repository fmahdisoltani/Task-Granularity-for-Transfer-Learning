import numpy as np
import torch

from collections import OrderedDict


def caption_accuracy(outputs):
    return caption_level_accuracy(outputs.captions, outputs.predictions)


def caption_level_accuracy(captions, predictions):
    _, caption_len = captions.size()
    equal_values = torch.sum(captions.eq(predictions), dim=1)
    accuracy = equal_values.eq(caption_len).float().mean().data.numpy()[0] * 100.0
    return accuracy


def first_token_accuracy(outputs):
    return token_accuracy(outputs, 1)


def fscore(precision, recall, beta=1):
    numerator = (1.0 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    return numerator/denominator


def gmeasure(precision, recall):
    return np.sqrt(precision * recall)


def loss_to_numpy(score_attr):
    return score_attr.loss.data.cpu().numpy()[0]


def token_accuracy(outputs, num_tokens=None):
    return token_level_accuracy(outputs.captions, outputs.predictions,
                                num_tokens)


def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(
        predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
    return accuracy


class LCS(object):
    def score(self, predictions, captions):
        num_predictions = len(predictions)
        assert num_predictions == len(captions)
        average_precision = 0
        average_recall = 0
        for i in range(num_predictions):
            _, lcs_score = self.compute_lcs(predictions[i], captions[i])
            precision = lcs_score/len(predictions[i])
            recall = lcs_score/len(captions[i])
            average_precision += (precision - average_precision)/(i + 1)
            average_recall += (recall - average_recall)/(i + 1)
        f1_score = fscore(average_precision, average_recall)
        g_measure = gmeasure(average_precision, average_recall)
        return average_precision, average_recall, f1_score, g_measure

    def compute_lcs(self, prediction, caption):
        num_rows = len(prediction)
        num_cols = len(caption)

        C = [[0] * (num_cols + 1) for _ in range(num_rows + 1)]
        for i in range(1, num_rows+1):
            for j in range(1, num_cols+1):
                if prediction[i-1] == caption[j-1]:
                    C[i][j] = C[i-1][j-1] + 1
                else:
                    C[i][j] = max(C[i][j-1], C[i-1][j])
        return C, C[num_cols][num_rows]


class ScoresOperator(object):
    def __init__(self, functions_dict):
        """
            Initializes scores_dict and functions_dict.
        Args:
            functions_dict: An OrderedDict whose keys are strings and values are
                the functions that will be applied.
        """

        self.average = "avg"
        self.functions_dict = functions_dict
        self.scores_dict = OrderedDict({self.average + "_" + score: 0 for score
                                        in self.functions_dict})

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
                if self.average in key}

    def update_moving_average(self, scores_dict, count):
        assert count > 0
        scores_dict = OrderedDict(scores_dict)
        scores_list = list(scores_dict.keys())
        for score in scores_list:
            average_score = self.average + "_" + score
            self.scores_dict[average_score] += (
                (scores_dict[score] - self.scores_dict[average_score]) / count)
            scores_dict[average_score] = self.scores_dict[average_score]
        return scores_dict


class MultiScorerOperator(ScoresOperator):
    def __init__(self, functions_dict, multiscorer, tokenizer):
        super().__init__(functions_dict)
        self.multiscorer = multiscorer
        self.scores_dict.update({self.average + "_" + scorer: 0 for scorer in
                                 self.multiscorer.scorers})
        self.expand_key("BLEU")
        self.tokenizer = tokenizer

    def expand_key(self, key):
        """
            Expand BLEU to BLEU@1, BLEU@2, BLEU@3, and BLEU@4.
        """

        self.scores_dict.pop(self.average + "_" + key)
        self.scores_dict.update({self.average + "_" + key + "@" + str(i + 1): 0
                                 for i in range(self.multiscorer.scorers[key]._n
                                                )})

    def run_scores(self, score_attr):
        scores_dict = ScoresOperator.run_scores(self, score_attr)
        multiscore_dict = self.multiscore(score_attr)
        return OrderedDict(multiscore_dict, **scores_dict)

    def multiscore(self, outputs):
        string_predictions = [self.tokenizer.get_string(str_pred.data.numpy())
                              for str_pred in outputs.predictions]

        return self.multiscorer.score(outputs.string_captions,
                                      string_predictions)