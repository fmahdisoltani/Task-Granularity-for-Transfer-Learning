import torch

from tensorboardX import SummaryWriter
from torch.autograd import Variable


def forward_hook_closure(master_dict, name, index=None, aggregate_steps=True):
    def forward_hook(module, input_tensor, output_tensor):
        forward_value = output_tensor
        if index is not None:
            forward_value = forward_value[index]
        if aggregate_steps is True:
            master_dict[name + "_forward"] = forward_value
        else:
            _, num_step, _ = forward_value.size()
            for i in range(num_step):
                master_dict[name + "_forward_" + str(i)] = forward_value[:, i]

    return forward_hook


class TensorboardAdapter(object):
    """
        An interface that uses tensorboard pytorch to visualize the contents of
        the pytorch model.
    """

    def __init__(self, log_dir=None):
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def add_gradients(self, model, global_step):
        """
            Adds a visualization of the model's gradients.
        """

        for param_name, param_value in model.named_parameters():
            self.summary_writer.add_histogram(param_name + "_grad",
                                              param_value.grad.cpu().data.numpy(),
                                              global_step)

    def add_graph(self, model, input_dims=None, model_output=None, **kwargs):
        """
            Adds a visualization of the model's computation graph.
        """

        if input_dims is not None:
            input_variable = Variable(torch.rand(*input_dims),
                                      requires_grad=True)
            model_output = model(input_variable, kwargs)
        if model_output is not None:
            self.summary_writer.add_graph(model, model_output)
        else:
            print("add_graph was not executed because model_output=None")

    def add_scalars(self, scalars_dict, global_step, is_training):
        """
            Visualizes the contents of scalars_dict which must be scalar.
        """

        pad = "_train" if is_training else "_valid"
        for key, value in scalars_dict.items():
            self.summary_writer.add_scalar(key + pad, value, global_step)

    def add_state_dict(self, model, global_step):
        """
            Visualizes the contents of model.state_dict().
        """

        model_state_dict = model.state_dict()
        for key, value in model_state_dict.items():
            self.summary_writer.add_histogram(key, value.cpu().numpy(), global_step)

    def add_activations(self, model, global_step):
        """
            Visualizes the variables in vars_dict_list.
        """

        vars_dict = model.activations
        for key, value in vars_dict.items():
            self.summary_writer.add_histogram(key, value.cpu().data.numpy(),
                                              global_step)

    def close(self):
        """
            Closes the summary_writer.
        """

        self.summary_writer.close()


class Seq2seqAdapter(TensorboardAdapter):
    """
        This inherited class is specifically designed for sequence-to-sequence
        models as the dynamic between the inputs and the labels is different
        compared to conventional feedforward classifiers.
    """

    def __init__(self, log_dir=None):
        super(Seq2seqAdapter, self).__init__(log_dir=log_dir)

    def add_graph(self, model, pair_dims=None, model_output=None, **kwargs):
        assert len(pair_dims) == 2
        if pair_dims is not None:
            label_variable = Variable(torch.zeros(*pair_dims[1]).long())
            model_output = model(
                (Variable(torch.zeros(*pair_dims[0]), requires_grad=True),
                 label_variable), **kwargs)
        if model_output is not None:
            self.summary_writer.add_graph(model, model_output)
        else:
            print("add_graph was not executed because model_output=None")
