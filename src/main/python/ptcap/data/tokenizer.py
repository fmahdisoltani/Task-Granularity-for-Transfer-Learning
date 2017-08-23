import pickle
import re
import numpy as np
import os


class Tokenizer(object):

    GO = '<GO>'
    END = '<END>'
    UNK = '<UNK>'

    def __init__(self, captions=None, user_maxlen=None):
        """
            Build captions from all the expanded labels in all annotation files
        Args:
            annotations: list of paths to annotation files
        """

        if captions:
            self._build_dictionaries(captions, user_maxlen)

    def _build_dictionaries(self, captions, user_maxlen):
        """
            Builds two dictionaries: One that maps from tokens to ints, and
            another that maps from ints back to tokens.
        """

        self.maxlen = np.max([len(caption.split())
                              for caption in captions]) + 1

        if user_maxlen:
            self.maxlen = np.min([self.maxlen, user_maxlen])

        print('\nBuilding dictionary for captions...')
        extra_tokens = [self.GO, self.END, self.UNK]
        tokens = [self.tokenize(p) for p in captions]
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
        if len(tokenized_caption) > self.maxlen:
            tokenized_caption = tokenized_caption[0:self.maxlen - 1]
        encoded_caption = [self.encode_token(token)
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

    def get_vocab_size(self):
        return len(self.caption_dict)

    def get_string(self, predictions):
        output_tokens = self.decode_caption(predictions)
        if self.END in output_tokens:
            end_index = output_tokens.index(self.END)
        else:
            end_index = len(predictions)
        return " ".join(output_tokens[:end_index]).lower()

    def load_dictionaries(self, path):
        with open(os.path.join(path, "tokenizer_dicts"), "rb") as f:
            self.caption_dict, self.inv_caption_dict = pickle.load(f)

    def save_dictionaries(self, path):
        with open(os.path.join(path, "tokenizer_dicts"), "wb") as f:
            pickle.dump((self.caption_dict, self.inv_caption_dict), f)
