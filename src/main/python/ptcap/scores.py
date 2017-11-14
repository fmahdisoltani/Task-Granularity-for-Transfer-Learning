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
