import pickle
import re
import numpy as np
from ptcap.data import config_parser


class Tokenizer(object):

    def __init__(self, annotations):
        """
        Build captions from all the expanded labels in all annotation files
        Args:
            annotations: list of paths to annotation files
        """

        self.captions = []
        for annot in annotations:
            self.captions +=[p for p in annot[ConfigParser.EXPANDED_CAPTION]]
        self.set_captions()

    def set_captions(self):
        self.maxlen = np.max([len(caption.split()) for caption in self.captions]) + 1
        print('\nBuilding dictionary for captions...')
        extra_tokens = ['<GO>', '<END>']
        all_tokens = [self.tokenize(p) for p in self.captions]
        all_tokens = [item for sublist in all_tokens for item in sublist]
        tokens = extra_tokens + sorted(list(set(all_tokens)))
        print('Nb of different tokens:', len(tokens))
        self.caption_dict = {k: idx for idx, k in enumerate(tokens)}
        self.inv_caption_dict = {idx: k for k, idx in self.caption_dict.items()}

    def tokenize(self, caption):
        tokenize_regex = re.compile('[^A-Z\s]')
        return [x for x in tokenize_regex.sub(
            '', caption.upper()).split(" ") if x is not ""]

    def encode_caption(self, caption):
        tokenized_caption = self.tokenize(caption)
        encoded_caption = [self.caption_dict[token] for token in tokenized_caption]
        encoded_caption += [self.caption_dict['<END>']] * (self.maxlen - len(tokenized_caption))
        return encoded_caption

    def decode_caption(self, indices):
        return [self.inv_caption_dict[index] for index in indices]

    def save_dictionaries(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.caption_dict, f)
            pickle.dump(self.inv_caption_dict, f)
