import numpy as np
from rtorchn.data.preprocessing import pad_video


class FixedSizeCrop1D(object):
    """
    FixedSizeCrop1D expects an input tensor of size T x C and returns a
    tensor of size self.cropsize x C. If T > self.cropsize, it extracts the
    centered features. If T < self.cropsize, it pads the input on both sides.
    """

    def __init__(self, cropsize, mode='edge'):
        self.cropsize = cropsize
        self.mode = mode

    def __call__(self, x):
        length = x.shape[0]
        if length < self.cropsize:
            left = int((self.cropsize - length) / 2)
            right = self.cropsize - length - left
            return np.pad(x, [(left, right), (0, 0)], mode=self.mode)
        start = int((length - self.cropsize) / 2)
        return x[start: start + self.cropsize]


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
    def __init__(self, scale=1.):
        self.scale = scale
        
    def __call__(self, x):

        return np.array(x, "float32") / self.scale


class PytorchTransposer(object):
    """
    Pytorch models expect the channel axis to be the first.
    """

    def __call__(self, x):
        return x.transpose(3, 0, 1, 2)
