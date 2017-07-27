import unittest

import pandas as pd

from ptcap.data.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.arguments = """[{"id":1,"file":"11/vid1.mp4",
                     "duration":"5.4","width":720,"height":480,
                     "label":"ONE HAND AND THE OTHER ",
                     "template":"ONE HAND AND THE OTHER",
                     "placeholders":["the table","coffee cup"],
                     "external_worker_id":"A2YD53VKHR3BED"},
                     {"id":2,"file":"21/vid2.mp4",
                     "duration":"5.4","width":720,"height":480,
                     "label":"Tilting the table with coffee cup on it slightly 
                     so it doesn’t fall down",
                     "template":"Tilting [something] with [something] on it 
                     slightly so it doesn’t fall down",
                     "placeholders":["the table","coffee cup"],
                     "external_worker_id":"A2YD53VKHR3BED"},
                     {"id":1,"file":"31/vid3.mp4",
                     "duration":"5.4","width":720,"height":480,
                     "label":"THREE HANDS ARE THERE",
                     "template":"THREE HANDS ARE THERE",
                     "placeholders":["the table","coffee cup"],
                     "external_worker_id":"A2YD53VKHR3BED"}
                   ]"""
        annot = self.arguments
        json_annot = pd.read_json(annot)
        self.captions = [p for p in json_annot["label"]]
        self.tokenizer = Tokenizer(self.captions)
        self.captions_vocab = {'THERE', 'HAND', 'ARE', 'AND', 'ON', 'TILTING',
                              'IT', 'TABLE', 'DOESNT', 'THREE', 'THE',
                              'COFFEE', 'CUP', 'SLIGHTLY', 'WITH', 'SO',
                              'HANDS', 'ONE', 'OTHER', 'FALL', 'DOWN', '<GO>',
                              '<END>', '<UNK>'}

    def test_set_captions_vocab(self):
        self.tokenizer.build_dictionaries()
        self.assertEqual(set(self.tokenizer.caption_dict.keys()),
                         self.captions_vocab)

    def test_set_captions_len(self):
        self.tokenizer.build_dictionaries()
        self.assertEqual(len(self.tokenizer.caption_dict),
                         len(self.captions_vocab))

    def test_encode_decode(self):
        phrase = "THERE IS ONE HAND"
        phrase_encoded = self.tokenizer.encode_caption(phrase)
        phrase_decoded = self.tokenizer.decode_caption(phrase_encoded)
        self.assertEqual(phrase_decoded[:6], [Tokenizer.GO,'THERE',
                                              Tokenizer.UNK, 'ONE',
                                             'HAND', Tokenizer.END])

    def test_max_len(self):
        phrase = "THERE IS ONE HAND"
        phrase_encoded = self.tokenizer.encode_caption(phrase)
        self.assertEqual(len(phrase_encoded), self.tokenizer.maxlen)
