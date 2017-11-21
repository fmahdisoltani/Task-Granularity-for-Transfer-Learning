import numpy as np
import torch

from collections import OrderedDict


def caption_accuracy(outputs):
    caption_level_accuracy_ = caption_level_accuracy(outputs.captions,
                                                     outputs.predictions)
    return {"caption_accuracy": caption_level_accuracy_}


def caption_level_accuracy(captions, predictions):
    _, caption_len = captions.size()
    equal_values = torch.sum(captions.eq(predictions), dim=1)
    accuracy = equal_values.eq(caption_len).float().mean().data.numpy()[0] * 100.0
    return accuracy


def first_token_accuracy(outputs):
    first_token_accuracy_ = token_level_accuracy(outputs.captions,
                                                 outputs.predictions, 1)
    return {"first_accuracy": first_token_accuracy_}


def fscore(precision, recall, beta=1):
    numerator = (1.0 + (beta ** 2)) * precision * recall
    denominator = ((beta ** 2) * precision) + recall
    return {"fscore": safe_div(numerator, denominator)}


def gmeasure(precision, recall):
    return {"gmeasure": np.sqrt(precision * recall)}


def loss_to_numpy(score_attr):
    return {"loss": score_attr.loss.data.cpu().numpy()[0]}


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


def token_accuracy(outputs, num_tokens=None):
    token_accuracy_ = token_level_accuracy(outputs.captions,
                                           outputs.predictions, num_tokens)
    return {"accuracy": token_accuracy_}


def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
    return accuracy


class ScoresOperator(object):
    def __init__(self, functions_list):
        """
            Initializes scores_dict and functions_dict.
        Args:
            functions_list: A list of the functions that will be applied.
        """

        self.avg_keyword = "avg"
        self.functions_list = functions_list
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
        for score_function in self.functions_list:
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
    def __init__(self, functions_list, tokenizer):
        """
        Initializes functions_list and tokenizer.
        Args:
        functions_list: A list of the functions that will be applied on the
        precision and recall values calculated based on the LCS between a
        prediction and a caption.
        """

        self.functions_list = functions_list
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

        for score_function in self.functions_list:
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
