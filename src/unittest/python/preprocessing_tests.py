import unittest

import numpy as np

from ptcap.data.preprocessing import RandomCrop


def create_fake_video(size):
    depth, height, width, num_channels = size
    video = np.random.randint(0, 255, depth*height*width*num_channels)
    return video.reshape(depth, height, width, num_channels)


class TestRandomCrop(unittest.TestCase):

    def setUp(self):
        self.size = [24, 56, 31, 3]
        self.video = create_fake_video(self.size)

    def test_with_size_12x30x30(self):
        cropper = RandomCrop([12, 30, 30])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, (12, 30, 30, 3))

    def test_with_size_96x96x96(self):
        cropper = RandomCrop([96, 96, 96])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, self.video.shape)

    def test_with_size_12x30x100(self):
        cropper = RandomCrop([12, 30, 100])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, (12, 30, 31, 3))

    def test_with_size_0x0x0(self):
        cropper = RandomCrop([0, 0, 0])
        video_crop = cropper(self.video)
        self.assertEqual(video_crop.shape, (0, 0, 0, 3))