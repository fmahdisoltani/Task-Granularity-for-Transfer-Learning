import numpy as np
from rtorchn.data.preprocessing import pad_video


class RandomCrop(object):

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, video):
        depth, height, width = self.cropsize
        bounds = np.array(video.shape[0:3]) - np.array([depth, height, width])
        bounds *= (bounds > 0).astype("int")
        d, h, w = [int(np.random.randint(0, bound + 1)) for bound in bounds]
        return video[d:d + depth, h:h + height, w:w + width]


class PadVideo(object):

    def __init__(self, padsize):
        self.padsize = padsize

    def __call__(self, video):
        depth, height, width = self.padsize
        return pad_video(video, depth, height, width)


class Float32Converter(object):
    def __call__(self, x):
        return np.array(x, "float32")


class PytorchTransposer(object):
    """
    Pytorch models expect the channel axis to be the first.
    """

    def __call__(self, x):
        return x.transpose(3, 0, 1, 2)
