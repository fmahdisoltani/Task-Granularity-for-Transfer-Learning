import os
import unittest

import torch

from testfixtures import tempdir

from torch.autograd import Variable

from ptcap.checkpointers import Checkpointer
from ptcap.model.pretrained_encoders import PretrainedEncoder
from ptcap.model.mappers import FullyConnectedMapper


class TestPretrainedEncoders(unittest.TestCase):
    def setUp(self, temp_dir):
        input_size = 2
        self.model = FullyConnectedMapper(input_size, 3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.state_dict = {
            "epoch": 0,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "score": None,
        }
        self.input = Variable(torch.zeros(5, input_size))
        self.model_name = "test_model"

    @tempdir()
    def test_load_pretrained_encoder(self, temp_dir):
        torch.save(self.model.state_dict(),
                   os.path.join(temp_dir.path, self.model_name))
        encoder = PretrainedEncoder(
            encoder=FullyConnectedMapper,
            pretrained_path=os.path.join(temp_dir.path, self.model_name),
            encoder_args=(2, 3))

        encoded = encoder(self.input)
        expected_encoding = self.model(self.input)
        self.assertEqual((encoded - expected_encoding).sum().data.numpy(), 0)

    @tempdir()
    def test_load_pretrained_encoder_with_dict_attr(self, temp_dir):
        checkpointer = Checkpointer(temp_dir.path)
        checkpointer.save_latest(self.state_dict, filename=self.model_name)
        encoder = PretrainedEncoder(FullyConnectedMapper,
        pretrained_path=os.path.join(temp_dir.path, self.model_name),
        encoder_args=(2, 3), checkpoint_key='model')
        encoded = encoder(self.input)
        expected_encoding = self.model(self.input)
        self.assertEqual((encoded - expected_encoding).sum().data.numpy(), 0)
