import unittest

import torch

from torch.autograd import Variable

from ptcap.model import captioners
from ptcap.model import decoders
from ptcap.model import encoders
from ptcap.model import mappers


class TestDimensions(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.vocab_size = 5
        self.caption_len = 4
        self.num_features = 256
        self.arguments = {
            'FullyConnectedEncoder': (((3, 10, 96, 96), self.num_features), {}),
            'CNN3dEncoder': ((self.num_features,), {}),
            'CNN3dLSTMEncoder': ((self.num_features,), {}),

            'FullyConnectedMapper': ((4, 10), {}),

            'FullyConnectedDecoder': ((self.num_features, self.caption_len,
                                       self.vocab_size), {}),

            'RtorchnEncoderP': ((self.num_features,), {}),
            'LSTMDecoder': ((17, self.num_features, self.vocab_size, 23,), {}),
            'RtorchnCaptioner': ((self.vocab_size,), {}),
            'EncoderDecoder': ((encoders.CNN3dLSTMEncoder, decoders.LSTMDecoder,
                                (self.num_features,),
                                (17, self.num_features, self.vocab_size, 23,)),
                               {}),
        }

    def test_encoders(self):
        encoder_classes = encoders.Encoder.__subclasses__()
        video_batch = Variable(torch.zeros(self.batch_size, 3, 10, 96, 96))
        for encoder_class in encoder_classes:
            with self.subTest(encoder_class=encoder_class):
                self.assertIn(encoder_class.__name__, self.arguments)

                args, kwargs = self.arguments[encoder_class.__name__]

                encoder = encoder_class(*args, **kwargs)
                encoded = encoder(video_batch)

                self.assertEqual(encoded.size()[0], self.batch_size)
                self.assertEqual(encoded.size()[1], self.num_features)
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

    def test_decoders(self):
        decoder_classes = decoders.Decoder.__subclasses__()
        init_state_batch = Variable(
            torch.zeros(self.batch_size, self.num_features))
        teacher_batch = Variable(
            torch.zeros(self.batch_size, self.caption_len).long())
        for decoder_class in decoder_classes:
            for use_teacher_forcing in [True, False]:
                with self.subTest(decoder_class=decoder_class,
                                  use_teacher_forcing=use_teacher_forcing):
                    self.assertIn(decoder_class.__name__, self.arguments)

                    args, kwargs = self.arguments[decoder_class.__name__]

                    decoder = decoder_class(*args, **kwargs)
                    decoded = decoder(init_state_batch, teacher_batch,
                                      use_teacher_forcing)

                    self.assertEqual(decoded.size()[0], self.batch_size)
                    self.assertEqual(decoded.size()[2], self.vocab_size)
                    self.assertEqual(len(decoded.size()), 3)

    def test_captioners(self):
        captioner_classes = captioners.Captioner.__subclasses__()
        video_batch = (Variable(torch.zeros(self.batch_size, 3, 10, 96, 96)),
                       Variable(torch.zeros(self.batch_size,
                                            self.caption_len).long()))
        for use_teacher_forcing in [True, False]:
            for captioner_class in captioner_classes:
                with self.subTest(captioner_class=captioner_class):
                    self.assertIn(captioner_class.__name__, self.arguments)

                    args, kwargs = self.arguments[captioner_class.__name__]

                    captioner = captioner_class(*args, **kwargs)
                    captioned = captioner(video_batch, use_teacher_forcing)

                    self.assertEqual(captioned.size()[0], self.batch_size)
                    self.assertEqual(captioned.size()[2], self.vocab_size)
                    self.assertEqual(len(captioned.size()), 3)
