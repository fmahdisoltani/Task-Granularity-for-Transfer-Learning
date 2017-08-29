from collections import OrderedDict


def token_accuracy(outputs, num_tokens=None):
    return token_level_accuracy(outputs.captions, outputs.predictions,
                                num_tokens)


def loss_to_numpy(metric_attr):
    return metric_attr.loss.data.cpu().numpy()[0]


def first_token_accuracy(outputs):
    return token_accuracy(outputs, 1)


def token_level_accuracy(captions, predictions, num_tokens=None):
    equal_values = captions[:, 0:num_tokens].eq(
        predictions[:, 0:num_tokens])
    accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
    return accuracy


class MetricsOperator(object):
    def __init__(self, functions_dict):
        """
            Initializes metrics_dict and functions_dict.
        Args:
            functions_dict: An OrderedDict whose keys are strings and values are
                the functions that will be applied.
        """

        self.functions_dict = functions_dict
        self.metrics_dict = OrderedDict()
        for metric in self.functions_dict:
            self.metrics_dict["average_" + metric] = 0

    def compute_metrics(self, metric_attr, count):
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

        metrics_dict = self.run_metrics(metric_attr)
        # Calculate a moving average of the metrics.
        metrics_dict = self.moving_average(metrics_dict, count)
        return metrics_dict

    def run_metrics(self, metric_attr):
        metrics_dict = OrderedDict()
        for index, metric in enumerate(self.functions_dict):
            metrics_dict[metric] = self.functions_dict[metric](metric_attr)
        return metrics_dict

    def get_average_metrics(self):
        return {key: self.metrics_dict[key] for key in self.metrics_dict
                if "average" in key}

    def moving_average(self, metrics_dict, count):
        metrics_list = list(metrics_dict.keys())
        for metric in metrics_list:
            average_metric = "average_" + metric
            self.metrics_dict[average_metric] += (
                (metrics_dict[metric] - self.metrics_dict[average_metric])
                / count)
            metrics_dict[average_metric] = self.metrics_dict[average_metric]
        return metrics_dict
