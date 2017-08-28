from collections import OrderedDict


def accuracy_namedtuple(outputs, num_tokens=None):
    return token_level_accuracy(outputs.captions, outputs.predictions,
                                num_tokens)


def compute_loss(named_tuple):
    return named_tuple.loss.data.cpu().numpy()[0]


def first_token_accuracy(outputs):
    return accuracy_namedtuple(outputs, 1)


def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(
        predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean() * 100.0
    return accuracy


class MetricsOperator(object):
    def __init__(self, functions_dict):
        """
            Initializes metrics_dict and functions_dict.
        Args:
            functions_dict: An OrderedDict of string keys and function values
                containing the functions and their corresponding names.
        """

        self.metrics_dict = OrderedDict()
        self.functions_dict = functions_dict
        for metric in self.functions_dict:
            self.metrics_dict["average_" + metric] = 0

    def compute_metrics(self, named_tuple, count):
        """
            Computes all the metrics provided by the functions_dict in __init__.
        Args:
            parameters_list: A list of arguments to be passed to be computed by
                functions_dict and stored in metrics_dict.
            count: An int indicating the number of iterations.
        Returns:
            An OrderedDict containing the most recent metrics as well as their
            moving average.
        """

        metrics_dict = self.run_metrics(named_tuple)
        # Calculate a moving average of the metrics.
        metrics_dict = self.moving_average(metrics_dict, count)
        return metrics_dict

    def run_metrics(self, named_tuple):
        metrics_dict = OrderedDict()
        for index, metric in enumerate(self.functions_dict):
            metrics_dict[metric] = self.functions_dict[metric](named_tuple)
        return metrics_dict

    def get_average_metrics(self):
        return {key: self.metrics_dict[key] for key in self.metrics_dict
                if "average" in key}

    def moving_average(self, metrics_dict, count):
        for metric in metrics_dict:
            average_metric = "average_" + metric
            self.metrics_dict[average_metric] += (
                (metrics_dict[metric] - self.metrics_dict[average_metric])
                / count)
            metrics_dict[average_metric] = self.metrics_dict[average_metric]
        return metrics_dict
