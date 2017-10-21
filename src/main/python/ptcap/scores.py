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
    numerator = (1.0 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
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


class ScoresBase(object):
    def __init__(self, keyword=""):
        """
            Initializes scores_dict and takes in a keyword.
        Args:
            keyword: A string used to highlight the entry of a key into
            scores_dict. For example, "avg" can be used as a keyword to retrieve
            the keys that contain "avg".
        """

        self.keyword = keyword
        self.scores_dict = OrderedDict()

    def update_moving_average(self, scores_dict, count):
        assert count > 0
        scores_dict = OrderedDict(scores_dict)
        scores_list = list(scores_dict.keys())
        for score in scores_list:
            average_score = self.keyword + "_" + score
            total_score = self.scores_dict.get(average_score, 0) * (count - 1)
            self.scores_dict[average_score] = (
                (scores_dict[score] + total_score) / count)
            scores_dict[average_score] = self.scores_dict[average_score]
        return scores_dict

    def get_keyword_scores(self):
        keyword_dict = {key: self.scores_dict[key] for key in self.scores_dict
                        if self.keyword in key}
        return OrderedDict(sorted(keyword_dict.items()))


class ScoresOperator(ScoresBase):
    def __init__(self, functions_list):
        """
            Initializes functions_list.
        Args:
            functions_list: A list of the functions that will be applied.
        """

        super().__init__("avg")
        self.functions_list = functions_list

    def compute_scores(self, score_attr, count):
        """
            Computes all the scores provided by the functions_list in __init__.
        Args:
            score_attr: The input passed as a NamedTuple to be computed by
                functions_list and stored in self.scores_dict.
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


class LCS(ScoresBase):
    def __init__(self, functions_list, tokenizer):
        """
            Initializes functions_list and tokenizer.
        Args:
            functions_list: A list of the functions that will be applied.
        """

        super().__init__("batch")
        self.functions_list = functions_list
        self.tokenizer = tokenizer

    def __call__(self, outputs):
        string_predictions = [self.tokenizer.get_string(str_pred.data.numpy())
                              for str_pred in outputs.predictions]
        return self.score(string_predictions, outputs.string_captions)

    @classmethod
    def compute_lcs(cls, prediction, caption):
        num_rows = len(prediction)
        num_cols = len(caption)

        C = [[0] * (num_cols + 1) for _ in range(num_rows + 1)]
        for i in range(1, num_rows + 1):
            for j in range(1, num_cols + 1):
                if prediction[i - 1] == caption[j - 1]:
                    C[i][j] = C[i - 1][j - 1] + 1
                else:
                    C[i][j] = max(C[i][j - 1], C[i - 1][j])
        return C, C[num_rows][num_cols]

    def score(self, predictions, captions):
        assert len(predictions) == len(captions)

        for count, (prediction, caption) in enumerate(zip(predictions,
                                                          captions)):
            scores_dict = self.run_scores(prediction.split(), caption.split())
            # Calculate and update the moving average of the scores.
            self.update_moving_average(scores_dict, count + 1)

        return self.get_keyword_scores()

    def run_scores(self, prediction, caption):
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
