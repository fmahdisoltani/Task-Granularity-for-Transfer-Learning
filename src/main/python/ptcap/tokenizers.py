import pickle
import re
import numpy as np


class Tokenizer(object):
    def __init__(self, phrases):
        self.maxlen = np.max([len(phrase.split()) for phrase in phrases]) + 1
        print('\nBuilding dictionary for phrases...')
        extra_tokens = ['<GO>', '<END>']
        all_tokens = [self.tokenize(p) for p in phrases]
        all_tokens = [item for sublist in all_tokens for item in sublist]
        tokens = extra_tokens + sorted(list(set(all_tokens)))
        print('Nb of different tokens:', len(tokens))
        self.phrase_dict = {k: idx for idx, k in enumerate(tokens)}
        self.inv_phrase_dict = {idx: k for k, idx in self.phrase_dict.items()}

    def tokenize(self, phrase):
        tokenize_regex = re.compile('[^A-Z\s]')
        return [x for x in tokenize_regex.sub(
            '', phrase.upper()).split(" ") if x is not ""]

    def encode_phrase(self, phrase):
        """
        Convert captions to integer sequence by mapping every word to its index using self.phrase_dict.
        """
        tokenized_phrase = self.tokenize(phrase)
        encoded_phrase = [self.phrase_dict[token] for token in tokenized_phrase]
        encoded_phrase += [self.phrase_dict['<END>']] * (self.maxlen - len(tokenized_phrase))
        return encoded_phrase

    def decode_phrase(self, indices):
        """
        Convert the indices output by the captioning model to tokens.
        :param indices: Indices output by the captioning model
        :return: list of tokens that correspond to the indices
        """

        return [self.inv_phrase_dict[index] for index in indices]

    def save_dictionaries(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.phrase_dict, f)
            pickle.dump(self.inv_phrase_dict, f)
