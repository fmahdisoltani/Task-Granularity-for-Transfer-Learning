from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    '/home/waseem/Models/GoogleNews-vectors-negative300.bin', binary=True)

dog = model['dog']

model.wmdistance("hi my name is waseem".split(), "not my problem".split())
