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

    def __init__(self, subset_size, total_size, drop_last=False):
        self.subset_size = subset_size
        self.total_size = total_size
        self.drop_last = drop_last
        self.reset_inds()

    def reset_inds(self):
        self.inds = torch.randperm(self.total_size)

    def __iter__(self):
        if len(self.inds) < self.subset_size:
            old_inds = self.inds
            self.reset_inds()
            self.inds = list(old_inds) + list(self.inds)
        subset = self.inds[0:self.subset_size]
        self.inds = self.inds[self.subset_size:]

        # if len(self.inds) == 0:
        #    self.reset_inds()
        # end_index = min(self.subset_size, len(self.inds))
        # self.inds = self.inds[end_index:]
        # subset = self.inds[0:end_index]

        return iter(subset)


    def __len__(self):
        return self.subset_size


