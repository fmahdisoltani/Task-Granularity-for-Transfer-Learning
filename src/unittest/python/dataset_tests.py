import unittest
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

from ptcap.data.annotation_parser import JsonParser
from ptcap.data.dataset import VideoDataset
from ptcap.data.dataset import (JpegVideoDataset, NumpyVideoDataset,
                                GulpVideoDataset)
from ptcap.data.tokenizer import Tokenizer

class TestVideoDataset(unittest.TestCase):

    FAKE_ANNOT = pd.read_json("""[{"id":1,"file":"11/vid1.mp4",
            "duration":"5.4","width":720,"height":480,
            "label":"ONE HAND AND THE OTHER ", 
            "template":"ONE HAND AND THE OTHER",
            "placeholders":["the table","coffee cup"],
            "external_worker_id":"A2YD53VKHR3BED"},
            {"id":2,"file":"21/vid2.mp4",
            "duration":"5.4","width":720,"height":480,
            "label":"Tilting the table with coffee cup on it",
            "template":"Tilting [something] with [something] on it",
            "placeholders":["the table","coffee cup"],
            "external_worker_id":"A2YD53VKHR3BED"},
            {"id":1,"file":"31/vid3.mp4",
            "duration":"5.4","width":720,"height":480,
            "label":"THREE HANDS ARE THERE",
            "template":"THREE HANDS ARE THERE",
            "placeholders":["the table","coffee cup"],
            "external_worker_id":"A2YD53VKHR3BED"}
            ]""")

    @mock.patch('ptcap.data.annotation_parser.JsonParser.open_annotation',
                return_value=FAKE_ANNOT)
    def setUp(self, mock_open_annot):
        # Create a fake video along with a string caption
        captions = [p for p in self.FAKE_ANNOT["label"]]
        self.tokenizer = Tokenizer(captions)
        self.annotation_parser = JsonParser('', '')
        self.dataset = VideoDataset(self.annotation_parser, self.tokenizer)

    @mock.patch('ptcap.data.dataset.VideoDataset._get_video',
                return_value= np.random.rand(5, 7, 11, 3))
    @mock.patch('ptcap.data.dataset.VideoDataset._get_tokenized_caption',
                return_value=np.random.rand(5, 7))
    def test_getitem(self, mock_get_video, mock_get_tokenized_caption):
        for i, target_caption in enumerate(self.annotation_parser.get_captions()):
            with self.subTest('check sample %d' % i):
                video, caption, tokenized_caption = self.dataset[i]
                self.assertEqual(caption, target_caption)
                self.assertEqual(video.shape, (5, 7, 11, 3))
                self.assertEqual(tokenized_caption.shape, (5, 7))

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)


class TestJpegVideoDataset(TestVideoDataset):
    PIL_IMAGE = Image.new("RGB", (256, 256), "white")

    @mock.patch('PIL.Image.open', return_value=PIL_IMAGE)
    @mock.patch('glob.glob', return_value=[0 for i in range(13)])
    def test_getvideo(self, mock_pil, mock_glob):
        dataset = JpegVideoDataset([128, 128], Image.BICUBIC,
                                   self.annotation_parser, self.tokenizer)
        video = dataset._get_video(0)
        self.assertEqual(video.shape, (13, 128, 128, 3))


class TestNumpyDataset(TestVideoDataset):
    NUMPY_VIDEO = {"arr_0": np.random.rand(11, 237, 237, 3)}

    @mock.patch('glob.glob', return_value=[0])
    @mock.patch('numpy.load', return_value=NUMPY_VIDEO)
    def test_getvideo(self, mock_np, mock_glob):
        dataset = NumpyVideoDataset(self.annotation_parser, self.tokenizer)
        video = dataset._get_video(0)
        self.assertEqual(video.shape, (11, 237, 237, 3))


class TestGulpVideoDataset(TestVideoDataset):
    GULP_VIDEO = np.random.rand(11, 237, 237, 3)
    GULP_FRAMES = [frame for frame in GULP_VIDEO]
    META_DICT = 0

    @mock.patch('gulpio.GulpDirectory.__getitem__',
                return_value=(GULP_FRAMES, META_DICT))
    def test_getvideo(self, mock_gulp):
        dataset = GulpVideoDataset(
                                   self.annotation_parser,
                                   self.tokenizer,
                                    '')
        video = dataset._get_video(0)
        self.assertEqual(video.shape, (11, 237, 237, 3))
        self.assertTrue(np.all(video==self.GULP_VIDEO))

    @mock.patch('gulpio.GulpDirectory.__getitem__',
                return_value=(GULP_FRAMES, META_DICT))
    def test_getvideo_with_size(self, mock_gulp):
        dataset = GulpVideoDataset(self.annotation_parser,
                                   self.tokenizer, '', size=[16, 16],
                                   )
        video = dataset._get_video(0)
        self.assertEqual(video.shape, (11, 16, 16, 3))
