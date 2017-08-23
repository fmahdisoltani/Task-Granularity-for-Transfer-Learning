from collections import OrderedDict


class Metrics(object):
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
            self.metrics_dict[metric] = 0
        for metric in self.functions_dict:
            self.metrics_dict["average_" + metric] = 0

    def compute_metrics(self, parameters_list, count):
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

        self.run_metrics(parameters_list)
        # Calculate a moving average of the metrics.
        self.metrics_dict = self.moving_average(self.metrics_dict, count)
        return self.metrics_dict

    def run_metrics(self, parameters_list):
        for index, metric in enumerate(self.functions_dict):
            self.metrics_dict[metric] = self.functions_dict[metric](
                                                        *parameters_list[index])

    @classmethod
    def moving_average(cls, metrics_dict, count):
        for metric in metrics_dict:
            average_metric = "average_" + metric
            if average_metric in metrics_dict:
                metrics_dict[average_metric] += (
                    (metrics_dict[metric] - metrics_dict[average_metric])
                    / count)
        return metrics_dict

    @classmethod
    def token_level_accuracy(cls, captions, predictions, num_tokens=None):
        equal_values = captions[:, 0:num_tokens].eq(
                                                predictions[:, 0:num_tokens])
        accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
        return accuracy
