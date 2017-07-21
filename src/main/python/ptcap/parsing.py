import os
from importlib import import_module

import torch.optim
import yaml
from torch.utils.data import DataLoader

import rtorchn.callbacks
import rtorchn.data
import rtorchn.metrics
import rtorchn.models.utils
import rtorchn.preparers
import rtorchn.preprocessing
from rtorchn.models.utils import Model
from rtorchn.preprocessing import PreprocessingOp
from rtorchn.utils import broadcast_to_devices, load_module_from_checkpoint


class YamlConfig(object):
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.configdict = yaml.load(f.read())

    def parse_model(self):
        """ 
        Build a Model (= net + optimizer + loss) from a config file.
        """
        if self.configdict['training']['resume']:
            return self.get_pretrained_model(self.configdict['training']['pretrained_model'])
        return self.get_raw_model()

    def parse_metrics(self, module=rtorchn.metrics):
        return dict([(name, getattr(module, name)) for name in self.configdict['metrics']])

    def parse_preparer(self, set, module=rtorchn.preparers):
        """
        Get a preparer for the given set of data.
        """
        video_preparation = PreprocessingOp(getattr(module, self.configdict['preparers']['video_preparation']['type']),
                                            self.configdict['preparers']['video_preparation']['args'])
        label_preparation = PreprocessingOp(getattr(module, self.configdict['preparers']['label_preparation']['type']),
                                            self.configdict['preparers']['label_preparation']['args'])
        json_file = self.get_annotation_file(set)
        return getattr(module, self.configdict['preparers']['type'])(json_file, video_preparation, label_preparation)

    def parse_dataset(self, set, data_module=rtorchn.data, prep_module=rtorchn.preprocessing):
        """
        Get a PyTorch dataset for the given set of data.
        """
        preparer = self.parse_preparer(set)
        video_prep_info = self.configdict['datasets']['video_preprocessing'][set]
        video_preprocess = getattr(prep_module, video_prep_info['type'])(*video_prep_info['args'])
        label_prep_info = self.configdict['datasets']['label_preprocessing'][set]
        label_preprocess = PreprocessingOp(getattr(prep_module, label_prep_info['type']), label_prep_info['args'])
        args = {'preparer': preparer, 'video_preprocessing': video_preprocess, 'label_preprocessing': label_preprocess}
        args = {**args, **self.configdict['datasets']['kwargs']}
        dataset = getattr(data_module, self.configdict['datasets']['type'])(**args)
        if 'gatherer' in self.configdict.keys():
            return getattr(data_module, self.configdict['gatherer']['type'])(dataset,
                                                                             *self.configdict['gatherer']['args'])
        return dataset

    def parse_dataloader(self, set, verbose=True):
        """
        Get a PyTorch dataloader for the given set of data.
        """
        loader_dict = self.configdict['dataloaders']
        # Get dataset
        dataset = self.parse_dataset(set)
        # Sampling strategy
        if set == 'train':
            nsamples_per_epoch = loader_dict['nb_batches_per_epoch'] * loader_dict['batch_size']
            sampler = rtorchn.data.SubsetRandomSampler(len(dataset), nsamples_per_epoch)
            shuffle = True
        else:
            sampler = None
            shuffle = False
        # Dataloader
        loader = DataLoader(dataset, **loader_dict['kwargs'], sampler=sampler, shuffle=shuffle)
        if verbose:
            print("%s data loaded." % set)
        return loader

    def parse_callbacks(self, model, module=rtorchn.callbacks):
        """
        Get a list of callbacks. By default, a BaseLogger callback is always used as the first callback.
        """
        callbacks_dict = self.configdict['callbacks']
        callbacks = [module.BaseLogger(model, callbacks_dict['print_every_nsteps'])]
        for t, arg in zip(callbacks_dict['type'], callbacks_dict['args']):
            callbacks.append(getattr(module, t)(*([model] + arg)))
        return callbacks

    def parse_loss(self, module=rtorchn.models.utils):
        return getattr(module, self.configdict['loss']['type'])(*self.configdict['loss']['args'])

    def parse_optimizer(self, model, module=torch.optim):
        """
        Parse optimizer. The optimizer is directly imported from the optim module of PyTorch.
        """
        return getattr(module, self.configdict['optimizer']['type'])(model.parameters(),
                                                                     **self.configdict['optimizer']['kwargs'])

    def get_annotation_file(self, set):
        """
        Return the path to the annotation file for the given set of data.
        """
        if set == "train":
            name = self.configdict['paths']['train_annot']
        elif set == "validation":
            name = self.configdict['paths']['validation_annot']
        elif set == "test":
            name = self.configdict['paths']['test_annot']
        else:
            raise Exception("'set' must be in [0, 1, 2] or ['train', 'validation', 'test']")
        return os.path.join(self.configdict['paths']['annot_path'], name)

    def get_raw_model(self, deepnet_module=None):
        """
        Build a model from scratch, and broadcast it to the appropriate device(s).
        """
        if deepnet_module is None:
            deepnet_module = import_module("rtorchn.models.%s" % self.configdict['model']['type'])
        # Build the network
        net = deepnet_module.DeepNet(*self.configdict['model']['args'],
                                     **self.configdict['model']['kwargs'])
        print(net)
        print("Model defined.")
        # Loss
        loss = self.parse_loss()
        # Load on appropriate devices
        net, loss = broadcast_to_devices(net, loss)
        # Optimizer
        optimizer = self.parse_optimizer(net)
        # Model instance
        model = Model(net, optimizer, loss, metrics=self.parse_metrics())
        # Weight initialization
        print("\nWeight initialization :")
        model.initialize()
        return model

    def get_pretrained_model(self, checkpoint_dir):
        """
        Build a model from a checkpoint directory, using the pretrained weights.
        """
        # Get module from saved python file
        deepnet_module = load_module_from_checkpoint(checkpoint_dir)
        # Get model from config
        model = self.get_raw_model(deepnet_module)
        # Load weights
        model.load_weights(os.path.join(checkpoint_dir, 'model.checkpoint'))
        return model

    def save(self, path):
        """
        Save the config file. 
        """
        with open(path, 'w') as f:
            yaml.dump(self.configdict, f)
