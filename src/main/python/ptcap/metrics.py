from collections import OrderedDict


class Metrics(object):
    def __init__(self, funcions_dict):
        """
            Initializes the metrics_dict and stores the metrics of interest.
        Agrs:
            metrics: A list of strings containing the metrics of interest.
        """

        self.metrics_dict = OrderedDict()
        self.funcions_dict = funcions_dict
        for metric in self.funcions_dict:
            self.metrics_dict[metric] = 0
        for metric in self.funcions_dict:
            self.metrics_dict["average_" + metric] = 0

    def compute_metrics(self, parameters_list, count):
        """
            Computes all the metrics provided in __init__.
        Args:
            captions: A list of torch.variable containing the captions encoded
                by the Tokenizer.
            predictions: A list of torch.variable containing the encoded model
                predictions.
            loss: A torch.variable containing the value of loss.
            count: An int indicating the number of iterations.
        Returns:
            An ordered dictionary containing the metrics as well as their moving
            average.
        """

        self.run_metrics(parameters_list)
        # Calculate a moving average of the metrics.
        self.moving_average(count)
        return self.metrics_dict

    def run_metrics(self, parameters_list):
        for index, metric in enumerate(self.metrics_dict):
            self.metrics_dict[metric] = self.funcions_dict[metric](
                                                        *parameters_list[index])

    def moving_average(self, count):
        for metric in self.funcions_dict:
            average_metric = "average_" + metric
            self.metrics_dict[average_metric] += (
                (self.metrics_dict[metric] -
                 self.metrics_dict[average_metric]) / count)

    @classmethod
    def token_level_accuracy(cls, captions, predictions, num_tokens=None):
        equal_values = captions[:, 0:num_tokens].eq(
                                                predictions[:, 0:num_tokens])
        accuracy = equal_values.float().mean().data.numpy()[0] * 100.0
        return accuracy
