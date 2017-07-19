import unittest

import torch
from torch.autograd import Variable

from ptcap import encoders


class TestDimensions(unittest.TestCase):

    arguments = {
        'FullyConnectedEncoder': (((3, 10, 12, 12), 4), {}),
    }

    def test_encoders(self):
        encoder_classes = encoders.Encoder.__subclasses__()
        video_batch = Variable(torch.zeros(5, 3, 10, 12, 12))
        for encoder_class in encoder_classes:
            with self.subTest(encoder_class=encoder_class):
                self.assertIn(encoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[encoder_class.__name__]

                encoder = encoder_class(*args, **kwargs)
                encoded = encoder(video_batch)

                self.assertEqual(encoded.size()[0], 5)
                self.assertEqual(len(encoded.size()), 2)

    def test_mappers(self):
        assert False

    def test_decoders(self):
        assert False
