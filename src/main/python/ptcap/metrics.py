from torch import nn


def accuracy(output, target):
    return output == target


def bleu(output, target):
    return output == target