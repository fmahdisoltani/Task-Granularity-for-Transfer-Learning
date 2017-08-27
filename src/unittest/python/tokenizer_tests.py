import unittest

import pandas as pd

from testfixtures import tempdir

from ptcap.data.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.phrase = "THERE IS ONE HAND"
        self.arguments = """[{"id":1,"file":"11/vid1.mp4",
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
        ]"""

        json_annot = pd.read_json(self.arguments)
        self.captions = [p for p in json_annot["label"]]
        self.tokenizer = Tokenizer(self.captions)
        self.captions_vocab = {'THERE', 'HAND', 'ARE', 'AND', 'ON', 'TILTING',
                               'IT', 'TABLE', 'THREE', 'THE', 'COFFEE', 'CUP',
                               'WITH', 'HANDS', 'ONE', 'OTHER', '<GO>', '<END>',
                               '<UNK>'}

    def test_set_captions_vocab(self):
        self.assertEqual(set(self.tokenizer.caption_dict.keys()),
                         self.captions_vocab)

    def test_set_captions_len(self):
        self.assertEqual(self.tokenizer.get_vocab_size(),
                         len(self.captions_vocab))

    def test_encode_decode(self):
        phrase_encoded = self.tokenizer.encode_caption(self.phrase)
        phrase_decoded = self.tokenizer.decode_caption(phrase_encoded)
        self.assertEqual(phrase_decoded[:5], ['THERE',
                                              Tokenizer.UNK, 'ONE',
                                             'HAND', Tokenizer.END])

    def test_max_len(self):
        phrase_encoded = self.tokenizer.encode_caption(self.phrase)
        self.assertEqual(len(phrase_encoded), self.tokenizer.maxlen)

    @tempdir()
    def test_save_load(self, temp_dir):
        self.tokenizer.save_dictionaries(temp_dir.path)
        loading_tokenizer = Tokenizer()
        loading_tokenizer.load_dictionaries(temp_dir.path)
        self.assertEqual(self.tokenizer.caption_dict,
                         loading_tokenizer.caption_dict)
        self.assertEqual(self.tokenizer.inv_caption_dict,
                         loading_tokenizer.inv_caption_dict)

    @tempdir()
    def test_tokenize_from_loaded_tokenizer(self, temp_dir):
        self.tokenizer.save_dictionaries(temp_dir.path)
        loading_tokenizer = Tokenizer()
        loading_tokenizer.load_dictionaries(temp_dir.path)
        phrase_encoded = loading_tokenizer.encode_caption(self.phrase)
        phrase_decoded = self.tokenizer.decode_caption(phrase_encoded)
        self.assertEqual(phrase_decoded[:5], ['THERE',
                                              Tokenizer.UNK, 'ONE',
                                             'HAND', Tokenizer.END])
