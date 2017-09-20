import torch

from tensorboardX import SummaryWriter
from torch.autograd import Variable


def update_dict(master_dict, vars_tuple, num_step):
    for key, var in vars_tuple:
        [master_dict.update({key + "_" + str(i): var[:, i]}) for i in
         range(num_step)]


def register_grad(master_dict, vars_tuple, num_step=None):
    for key, value in vars_tuple:
        value.register_hook(save_grad(master_dict, key, num_step))


def save_grad(master_dict, name, num_step=None):
    def hook(grad):
        if num_step is not None:
            [master_dict.update({name + "_" + str(i) + "_grad": grad[:, i]})
             for i in range(num_step)]
        else:
            master_dict[name + "_grad"] = grad

    return hook


def fh(module, inp, out):
    print("******backward_pass BEGIN********")
    print("Number of parameters")
    print(len(list(module.parameters())))
    print("Number of named parameters")
    print(len(list(module.named_parameters())))
    # for name, value in module.named_parameters():
    #     print(name)
    #     print(value)
    print(inp.size())
    print(out.size())
    print("*******backward_pass END*********")


def bh(module, inp, out):
    print("******backward_pass BEGIN********")
    print("Number of parameters")
    print(len(list(module.parameters())))
    print("Number of named parameters")
    print(len(list(module.named_parameters())))
    # for name, value in module.named_parameters():
    #     print(name)
    #     print(value)
    print(inp.size())
    print(out.size())
    print("*******backward_pass END*********")


class TensorboardAdapter(object):
    """
        An interface that uses tensorboard pytorch to visualize the contents of
        the pytorch model.
    """

    def __init__(self, log_dir=None):
        self.summary_writer = SummaryWriter(log_dir=log_dir)

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

    def add_state_dict(self, model, global_step):
        """
            Visualizes the contents of model.state_dict().
        """

        model_state_dict = model.state_dict()
        for key, value in model_state_dict.items():
            self.summary_writer.add_histogram(key, value.numpy(),
                                              global_step)

    def add_variables(self, vars_dict, global_step):
        """
            Visualizes the variables in vars_dict_list.
        """

        for key, value in vars_dict.items():
            self.summary_writer.add_histogram(key, value.data.numpy(),
                                              global_step)

    def add_scalars(self, scalars_dict, global_step, is_training):
        """
            Visualizes the contents of scalars_dict which must be scalar.
        """

        pad = "train_" if is_training else "valid_"
        for key, value in scalars_dict.items():
            self.summary_writer.add_scalar(pad + key, value, global_step)

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
