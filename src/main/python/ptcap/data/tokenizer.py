import numpy as np
import os
import pickle
import re

from collections import Counter


class Tokenizer(object):

    GO = "<GO>"
    END = "<END>"
    UNK = "<UNK>"

    def __init__(self, captions=None, user_maxlen=None, cutoff=0):
        """
            Build captions from all the expanded labels in all annotation files.
        Args:
            captions: list of paths to annotation files.
            user_maxlen: the maximum length of the captions set by the user.
        """

        self.extra_tokens = ["<GO>", "<END>", "<UNK>"]
        self.maxlen = None if user_maxlen is None else user_maxlen
        self.cutoff = cutoff
        if captions:
            self.build_dictionaries(captions)

    def build_dictionaries(self, captions):
        """
            Builds two dictionaries: One that maps from tokens to ints, and
            another that maps from ints back to tokens.
        """

        maxlen = np.max([len(caption.split()) for caption in captions]) + 1
        self.set_maxlen(maxlen)

        print("\nBuilding dictionary for captions...")
        tokens = self.get_all_tokens(captions)
        tokens = self.filter_tokens(tokens)
        unique_tokens = self.extra_tokens + sorted(set(tokens))

        self.caption_dict = {k: idx for idx, k in enumerate(unique_tokens)}
        self.inv_caption_dict = {idx: k for k, idx in self.caption_dict.items()}

        print("Number of different tokens: ", len(unique_tokens))

    def get_all_tokens(self, captions):

        tokens = [self.tokenize(p) for p in captions]
        tokens = [item for sublist in tokens for item in sublist]
        return tokens

    def get_token_freqs(self, captions):
        tokens = self.get_all_tokens(captions)
        unk_freq = self.get_weight_for_unk(Counter(tokens))
        tokens = self.filter_tokens(tokens)
        tokens_freq = Counter([self.caption_dict[t] for t in tokens])
        max_frequency = max(tokens_freq.values())
        for extra_token in self.extra_tokens:
            tokens_freq[self.caption_dict[extra_token]] = max_frequency

        tokens_freq[self.caption_dict[self.UNK]] = unk_freq


        return tokens_freq

    def get_weight_for_unk(self, tokens_freq):
        unk_freq = 0
        for token in tokens_freq:

            unk_freq += \
                tokens_freq[token] if tokens_freq[token] < self.cutoff else 0
        print("unk freq is {}".format(unk_freq))

        return unk_freq


    def tokenize(self, caption):
        tokenize_regex = re.compile("[^A-Z\s]")
        return [x for x in tokenize_regex.sub(
            "", caption.upper()).split(" ") if x is not ""]

    def filter_tokens(self, tokens):
        tokens_count = Counter(tokens)
        return [tok for tok in tokens if tokens_count[tok] > self.cutoff]

    def encode_caption(self, caption):

        tokenized_caption = self.tokenize(caption)
        if len(tokenized_caption) >= self.maxlen:
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
        return " ".join(output_tokens[:end_index]).upper()

    def set_maxlen(self, maxlen):
        assert maxlen >= 1
        if self.maxlen is None:
            self.maxlen = maxlen
        else:
            self.maxlen = np.min([self.maxlen, maxlen])

    def load_dictionaries(self, path):
        # TODO: save and load token_freqs if necessary
        with open(os.path.join(path, "tokenizer_dicts"), "rb") as f:
            (self.maxlen, self.caption_dict,
             self.inv_caption_dict) = pickle.load(f)

    def save_dictionaries(self, path):
        with open(os.path.join(path, "tokenizer_dicts"), "wb") as f:
            pickle.dump((self.maxlen, self.caption_dict,
                         self.inv_caption_dict), f)
