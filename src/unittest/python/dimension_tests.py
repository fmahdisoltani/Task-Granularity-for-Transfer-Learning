import unittest

import torch
from torch.autograd import Variable

from ptcap import encoders
from ptcap import mappers
from ptcap import decoders



class TestDimensions(unittest.TestCase):

    arguments = {
        'FullyConnectedEncoder': (((3, 10, 12, 12), 4), {}),
        'FullyConnectedMapper': ((4, 10),{}),
        'FullyConnectedDecoder': ((10, (3, 5)),{}),
    }


    def test_encoders(self):
        encoder_classes = encoders.Encoder.__subclasses__()
        batch_size = 5
        video_batch = Variable(torch.zeros(batch_size, 3, 10, 12, 12))
        for encoder_class in encoder_classes:
            with self.subTest(encoder_class=encoder_class):
                self.assertIn(encoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[encoder_class.__name__]

                encoder = encoder_class(*args, **kwargs)
                encoded = encoder(video_batch)

                self.assertEqual(encoded.size()[0], batch_size)
                self.assertEqual(len(encoded.size()), 2)

    def test_mappers(self):
        mapper_classes = mappers.Mapper.__subclasses__()
        batch_size = 5
        video_batch = Variable(torch.zeros(batch_size, 4))
        for mapper_class in mapper_classes:
            with self.subTest(mapper_class=mapper_class):
                self.assertIn(mapper_class.__name__, self.arguments)

                args, kwargs = self.arguments[mapper_class.__name__]

                mapper = mapper_class(*args, **kwargs)
                mapped = mapper(video_batch)

                self.assertEqual(mapped.size()[0], batch_size)
                self.assertEqual(len(mapped.size()), 2)

    def test_decoders(self):
        decoder_classes = decoders.Decoder.__subclasses__()
        batch_size = 5
        data_batch = Variable(torch.zeros(batch_size, 10))
        for decoder_class in decoder_classes:
            with self.subTest(decoder_class=decoder_class):
                self.assertIn(decoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[decoder_class.__name__]

                decoder = decoder_class(*args, **kwargs)
                decoded = decoder(data_batch)

                self.assertEqual(decoded.size()[0], batch_size)
                self.assertEqual(decoded.size()[2], args[1][1])
                self.assertEqual(len(decoded.size()), 3)
