import unittest

import torch
from torch.autograd import Variable

from ptcap.data.dataset import VideoDataset
from ptcap.data.dataset import JpegVideoDataset

class TestVideoDataset(unittest.TestCase):

    def setUp(self):
        # Create a fake video along with a string caption
        pass

    # tests:
    # __getitem__ returns video, string, tokenized_string tuple
    # _getvideo returns the videos with the correct size
    # __len__ returns the correct number of videos

    def test_getitem(self):
        pass

    def test_len(self):
        pass


class TestJpegVideoDataset(unittest.TestCase):

    def setUp(self):
        pass

    def test_getvideo(self):
        pass