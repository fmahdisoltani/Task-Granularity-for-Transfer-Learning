import unittest

import torch
from torch.autograd import Variable

from ptcap.model import encoders
from ptcap.model import mappers
from ptcap.model import decoders


class TestDimensions(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.vocab_size = 5
        self.caption_len = 4
        self.arguments = {
            'FullyConnectedEncoder': (((3, 10, 12, 12), 4), {}),
            'FullyConnectedMapper': ((4, 10),{}),
            'FullyConnectedDecoder': ((10, self.caption_len, self.vocab_size),
                                      {}),
        }


    def test_encoders(self):
        encoder_classes = encoders.Encoder.__subclasses__()
        video_batch = Variable(torch.zeros(self.batch_size, 3, 10, 12, 12))
        for encoder_class in encoder_classes:
            with self.subTest(encoder_class=encoder_class):
                self.assertIn(encoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[encoder_class.__name__]

                encoder = encoder_class(*args, **kwargs)
                encoded = encoder(video_batch)

                self.assertEqual(encoded.size()[0], self.batch_size)
                self.assertEqual(len(encoded.size()), 2)

    def test_mappers(self):
        mapper_classes = mappers.Mapper.__subclasses__()
        feature_batch = Variable(torch.zeros(self.batch_size, 4))
        for mapper_class in mapper_classes:
            with self.subTest(mapper_class=mapper_class):
                self.assertIn(mapper_class.__name__, self.arguments)

                args, kwargs = self.arguments[mapper_class.__name__]

                mapper = mapper_class(*args, **kwargs)
                mapped = mapper(feature_batch)

                self.assertEqual(mapped.size()[0], self.batch_size)
                self.assertEqual(len(mapped.size()), 2)

    def test_decoders_with_teacher_forcing(self):
        decoder_classes = decoders.Decoder.__subclasses__()
        init_state_batch = Variable(torch.zeros(self.batch_size, 10))
        teacher_batch = Variable(torch.zeros(self.batch_size, self.caption_len,
                                             self.vocab_size))
        for decoder_class in decoder_classes:
            with self.subTest(decoder_class=decoder_class):
                self.assertIn(decoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[decoder_class.__name__]

                decoder = decoder_class(*args, **kwargs)
                decoded = decoder(init_state_batch, teacher_batch)

                self.assertEqual(decoded.size()[0], self.batch_size)
                self.assertEqual(decoded.size()[2], self.vocab_size)
                self.assertEqual(len(decoded.size()), 3)

    def test_decoders_without_teacher_forcing(self):
        decoder_classes = decoders.Decoder.__subclasses__()
        init_state_batch = Variable(torch.zeros(self.batch_size, 10))
        for decoder_class in decoder_classes:
            with self.subTest(decoder_class=decoder_class):
                self.assertIn(decoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[decoder_class.__name__]

                decoder = decoder_class(*args, **kwargs)
                decoded = decoder(init_state_batch)

                self.assertEqual(decoded.size()[0], self.batch_size)
                self.assertEqual(decoded.size()[2], self.vocab_size)
                self.assertEqual(len(decoded.size()), 3)
