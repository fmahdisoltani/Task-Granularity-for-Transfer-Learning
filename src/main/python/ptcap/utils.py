
import torch.nn as nn


class DataParallelWrapper(nn.DataParallel):
    def __init__(self, model, device_ids):
        super().__init__(model, device_ids=device_ids)
        #self.gpus = device_ids

    @property
    def activations(self):
        return self.module.activations

    def extract_features(self, x):
        return self.module.extract_features(x)