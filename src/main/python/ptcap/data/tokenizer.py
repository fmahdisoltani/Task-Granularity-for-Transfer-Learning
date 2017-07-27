import pickle
import re
import numpy as np


class Tokenizer(object):

    GO = '<GO>'
    END = '<END>'
    UNK = '<UNK>'

    def __init__(self, captions):
        """
            Build captions from all the expanded labels in all annotation files
        Args:
            annotations: list of paths to annotation files
        """
        self.build_dictionaries(captions)

    def build_dictionaries(self):
        """
            Builds two dictionaries: One that maps from tokens to ints, and
            another that maps from ints back to tokens.
        """

        self.maxlen = np.max([len(caption.split())
                              for caption in self.captions]) + 1
        print('\nBuilding dictionary for captions...')
        extra_tokens = [self.GO, self.END, self.UNK]
        tokens = [self.tokenize(p) for p in self.captions]
        tokens = [item for sublist in tokens for item in sublist]
        all_tokens = extra_tokens + list(set(tokens))
        print('Number of different tokens: ', len(all_tokens))
        self.caption_dict = {k: idx for idx, k in enumerate(all_tokens)}
        self.inv_caption_dict = {idx: k for k, idx in self.caption_dict.items()}
        print(self.caption_dict)
        print(self.inv_caption_dict)

    def tokenize(self, caption):
        tokenize_regex = re.compile('[^A-Z\s]')
        return [x for x in tokenize_regex.sub(
            '', caption.upper()).split(" ") if x is not ""]

    def encode_caption(self, caption):
        tokenized_caption = self.tokenize(caption)
        encoded_caption = [self.caption_dict[self.GO]]
        encoded_caption += [self.encode_token(token)
                            for token in tokenized_caption]
        return self.pad_with_end(encoded_caption)

    def encode_token(self, token):
        return self.caption_dict[token] if token in self.caption_dict else \
            self.caption_dict[self.UNK]

    def decode_caption(self, indices):
        return [self.inv_caption_dict[index] for index in indices]

    def pad_with_end(self, encoded_caption):
        num_end = self.maxlen - len(encoded_caption)
        return encoded_caption + num_end * [self.caption_dict[self.END]]

    def save_dictionaries(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.caption_dict, f)
            pickle.dump(self.inv_caption_dict, f)
