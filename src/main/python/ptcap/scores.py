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


def loss_to_numpy(score_attr):
    return {"loss": score_attr.loss.data.cpu().numpy()[0]}


def token_accuracy(outputs, num_tokens=None):
    token_accuracy_ = token_level_accuracy(outputs.captions,
                                           outputs.predictions, num_tokens)
    return {"accuracy": token_accuracy_}


def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
    return accuracy


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
        for score_function in self.functions_dict:
            scores_dict.update(score_function(score_attr))
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
            total_score = self.scores_dict.get(average_score, 0) * (count - 1)
            self.scores_dict[average_score] = (
                (scores_dict[score] + total_score) / count)
            scores_dict[average_score] = self.scores_dict[average_score]
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
