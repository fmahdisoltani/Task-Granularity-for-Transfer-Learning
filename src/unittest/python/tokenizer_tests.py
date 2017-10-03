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
        self.max_phraselen = 9
        json_annot = pd.read_json(self.arguments)
        self.captions = [p for p in json_annot["label"]]
        self.captions_vocab = {"THERE", "HAND", "ARE", "AND", "ON", "TILTING",
                               "IT", "TABLE", "THREE", "THE", "COFFEE", "CUP",
                               "WITH", "HANDS", "ONE", "OTHER", "<GO>", "<END>",
                               "<UNK>"}

    def test_captions_vocab(self):
        tokenizer = Tokenizer(self.captions)
        self.assertEqual(set(tokenizer.caption_dict.keys()),
                         self.captions_vocab)

    def test_captions_len(self):
        tokenizer = Tokenizer(self.captions)
        self.assertEqual(tokenizer.get_vocab_size(),
                         len(self.captions_vocab))

    def test_encode_decode(self):
        for value in [None, 5, 10]:
            with self.subTest(captions=self.captions, user_maxlen=value):
                tokenizer = Tokenizer(self.captions, value)
                phrase_encoded = tokenizer.encode_caption(self.phrase)
                phrase_decoded = tokenizer.decode_caption(phrase_encoded)
                expected_string = ["THERE", Tokenizer.UNK, "ONE", "HAND"]
                pad_length = self.max_phraselen if (value is None) or (
                    value >= self.max_phraselen) else value
                self.assertEqual(phrase_decoded[:pad_length], expected_string +
                                 [Tokenizer.END] * (pad_length -
                                                    len(expected_string)))

    def test_encode_decode_with_1_maxlen(self):
        tokenizer = Tokenizer(self.captions, 1)
        phrase_encoded = tokenizer.encode_caption(self.phrase)
        phrase_decoded = tokenizer.decode_caption(phrase_encoded)
        self.assertEqual(phrase_decoded, [Tokenizer.END])

    def test_encoding_length_equal_max_len(self):
        for value in [None, 1, 5, 10]:
            with self.subTest(captions=self.captions, user_maxlen=value):
                tokenizer = Tokenizer(self.captions, value)
                phrase_encoded = tokenizer.encode_caption(self.phrase)
                self.assertEqual(len(phrase_encoded), tokenizer.maxlen)

    def test_get_string(self):
        tokenizer = Tokenizer(self.captions, 5)
        first_chunk = ["THERE", tokenizer.UNK, "ONE", "HAND"]
        for remove_end in [True, False]:
            phrase_encoded = tokenizer.encode_caption(self.phrase)
            if remove_end:
                phrase_encoded[-1] = phrase_encoded[0]
                expected = " ".join(first_chunk + [first_chunk[0]])
            else:
                expected = " ".join(first_chunk)
            with self.subTest(remove_end=remove_end):
                string = tokenizer.get_string(phrase_encoded)
                self.assertEqual(expected, string)

    def test_user_maxlen(self):
        for value in [None, 1, 5, 10]:
            with self.subTest(user_maxlen=value):
                tokenizer = Tokenizer(user_maxlen=value)
                self.assertEqual(tokenizer.maxlen,
                                 None if value is None else value)

    def test_max_len(self):
        tokenizer = Tokenizer(self.captions)
        self.assertEqual(tokenizer.maxlen, self.max_phraselen)

    def test_user_maxlen_vs_caption_maxlen(self):
        for value in [None, 1, 5, 10]:
            with self.subTest(captions=self.captions, user_maxlen=value):
                tokenizer = Tokenizer(self.captions, value)
                self.assertEqual(tokenizer.maxlen, self.max_phraselen if (
                    value is None or value >= self.max_phraselen) else value)

    def test_assert_error_for_maxlen(self):
        tokenizer = Tokenizer()
        with self.assertRaises(AssertionError):
            tokenizer.set_maxlen(-1)

    @tempdir()
    def test_save_load(self, temp_dir):
        for value in [None, 1, 5, 10]:
            with self.subTest(captions=self.captions, user_maxlen=value):
                tokenizer = Tokenizer(self.captions, value)
                tokenizer.save_dictionaries(temp_dir.path)
                loading_tokenizer = Tokenizer()
                loading_tokenizer.load_dictionaries(temp_dir.path)
                self.assertEqual(tokenizer.caption_dict,
                                 loading_tokenizer.caption_dict)
                self.assertEqual(tokenizer.inv_caption_dict,
                                 loading_tokenizer.inv_caption_dict)

    @tempdir()
    def test_encode_decode_from_loaded_tokenizer(self, temp_dir):
        for value in [None, 5, 10]:
            with self.subTest(captions=self.captions, user_maxlen=value):
                tokenizer = Tokenizer(self.captions, value)
                tokenizer.save_dictionaries(temp_dir.path)
                loading_tokenizer = Tokenizer()
                loading_tokenizer.load_dictionaries(temp_dir.path)
                phrase_encoded = loading_tokenizer.encode_caption(self.phrase)
                phrase_decoded = tokenizer.decode_caption(phrase_encoded)
                expected_string = ["THERE", Tokenizer.UNK, "ONE", "HAND"]
                pad_length = self.max_phraselen if (value is None) or (
                    value >= self.max_phraselen) else value
                self.assertEqual(phrase_decoded[:pad_length], expected_string +
                                 [Tokenizer.END] * (pad_length -
                                                    len(expected_string)))

    @tempdir()
    def test_encode_decode_from_loaded_tokenizer_with_1_maxlen(self, temp_dir):
        tokenizer = Tokenizer(self.captions, 1)
        tokenizer.save_dictionaries(temp_dir.path)
        loading_tokenizer = Tokenizer()
        loading_tokenizer.load_dictionaries(temp_dir.path)
        phrase_encoded = loading_tokenizer.encode_caption(self.phrase)
        phrase_decoded = tokenizer.decode_caption(phrase_encoded)
        self.assertEqual(phrase_decoded, [Tokenizer.END])
