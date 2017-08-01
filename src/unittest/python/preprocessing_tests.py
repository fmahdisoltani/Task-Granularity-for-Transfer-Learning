import unittest

import numpy as np

import ptcap.data.preprocessing as prep


def create_fake_video(size):
    depth, height, width, num_channels = size
    video = np.random.randint(0, 255, depth*height*width*num_channels)
    return video.reshape(depth, height, width, num_channels)


class TestRandomCrop(unittest.TestCase):

    def setUp(self):
        self.size = [24, 56, 31, 3]
        self.video = create_fake_video(self.size)

    def test_with_size_12x30x30(self):
        cropper = prep.RandomCrop([12, 30, 30])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, (12, 30, 30, 3))

    def test_with_size_96x96x96(self):
        cropper = prep.RandomCrop([96, 96, 96])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, self.video.shape)

    def test_with_size_12x30x100(self):
        cropper = prep.RandomCrop([12, 30, 100])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, (12, 30, 31, 3))

    def test_with_size_0x0x0(self):
        cropper = prep.RandomCrop([0, 0, 0])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, (0, 0, 0, 3))

    def test_with_size_24x56x31x3(self):
        cropper = prep.RandomCrop([24, 56, 31])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, self.video.shape)


class TestPadVideo(unittest.TestCase):

    def setUp(self):
        self.size = [24, 56, 31, 3]
        self.video = create_fake_video(self.size)

    def test_with_size_12x30x30(self):
        padder = prep.PadVideo([12, 30, 30])
        video_pad = padder(self.video)
        self.assertEqual(video_pad.shape, self.video.shape)

    def test_with_size_96x96x96(self):
        padder = prep.PadVideo([96, 96, 96])
        video_pad = padder(self.video)
        self.assertEqual(video_pad.shape, (96, 96, 96, 3))

    def test_with_size_12x30x100(self):
        padder = prep.PadVideo([12, 30, 100])
        video_pad = padder(self.video)
        self.assertEqual(video_pad.shape, (24, 56, 100, 3))

    def test_with_size_24x56x31x3(self):
        padder = prep.PadVideo([24, 56, 31])
        video_pad = padder(self.video)
        self.assertEqual(video_pad.shape, self.video.shape)

    def test_with_size_0x0x0(self):
        padder = prep.PadVideo([0, 0, 0])
        video_pad = padder(self.video)
        self.assertEqual(video_pad.shape, self.video.shape)


class TestFloat32Converter(unittest.TestCase):

    def setUp(self):
        self.size = [24, 56, 31, 3]
        self.video = create_fake_video(self.size)

    def test_dtype(self):
        float_converter = prep.Float32Converter()
        self.assertEqual(float_converter(self.video).dtype, 'float32')


class TestPytorchPermuter(unittest.TestCase):

    def setUp(self):
        self.size = [24, 56, 31, 3]
        self.video = create_fake_video(self.size)


    def test_dtype(self):
        pytorch_permuter = prep.PytorchTransposer()
        self.assertEqual(pytorch_permuter(self.video).shape, (3, 24, 56, 31))
