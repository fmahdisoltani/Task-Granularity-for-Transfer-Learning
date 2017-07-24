import yaml
import rtorchn
import os

class ConfigParser(object):
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.configdict = yaml.load(f.read())

    def parse_metrics(self, module=rtorchn.metrics):
        return dict([(name, getattr(module, name)) for name in self.configdict['metrics']])

    def parse_dataset(self, set, data_module=rtorchn.data, prep_module=rtorchn.preprocessing):
        pass

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


    def save(self, path):
        """
        Save the config file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.configdict, f)

