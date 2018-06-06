import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler


class DataParallelWrapper(nn.DataParallel):
    def __init__(self, model, device_ids):
        super().__init__(model, device_ids=device_ids)
#        self.encoder_output_size = self.module.encoder_output_size

    @property
    def activations(self):
        return self.module.activations

    def extract_features(self, x):
        return self.module.extract_features(x)

class CustomSubsetSampler(Sampler):

    def __init__(self, subset_size, total_size):
        self.subset_size = subset_size
        self.total_size = total_size
        self.reset_ids()

    def reset_ids(self):
        self.ids = torch.randperm(self.total_size)

    def __iter__(self):
        end_index = min(self.subset_size, len(self.ids))
        subset = self.ids[0:end_index]
        if len(self.ids) > self.subset_size:
            self.ids = self.ids[end_index:]
        else:
            self.reset_ids()
        print(subset)
        return iter(subset)

    def __len__(self):
        return self.subset_size


