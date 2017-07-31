import numpy as np


class RandomCrop(object):

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, video):
        depth, height, width = self.cropsize
        bounds = np.array(video.shape[0:3]) - np.array([depth, height, width])
        bounds *= (bounds > 0).astype("int")
        d, h, w = [int(np.random.randint(0, bound + 1)) for bound in bounds]
        return video[d:d + depth, h:h + height, w:w + width]
