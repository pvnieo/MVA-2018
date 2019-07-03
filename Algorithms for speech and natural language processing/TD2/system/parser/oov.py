# stdlib
import pickle
from os.path import join, dirname, realpath
from operator import itemgetter
import re
# 3p
import numpy as np
# project
from utils import cosine_distance, levenshtein_distance


class OOV:
    def __init__(self):
        # load pyglot
        with open(join(dirname(realpath(__file__)), 'polyglot-fr.pkl'), 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.words, self.embeddings = u.load()

        # Special tokens
        self.Token_ID = {"<UNK>": 0, "<S>": 1, "</S>": 2, "<PAD>": 3}
        self.ID_Token = {v: k for k, v in self.Token_ID.items()}

        # Map words to indices and vice versa
        self.word_id = {w: i for (i, w) in enumerate(self.words)}
        self.id_word = dict(enumerate(self.words))

        # Noramlize digits by replacing them with #
        self.DIGITS = re.compile("[0-9]", re.UNICODE)

    def case_normalizer(self, word, dictionary):
        w = word
        lower = (dictionary.get(w.lower(), 1e12), w.lower())
        upper = (dictionary.get(w.upper(), 1e12), w.upper())
        title = (dictionary.get(w.title(), 1e12), w.title())
        results = [lower, upper, title]
        results.sort()
        index, w = results[0]
        if index != 1e12:
            return w
        return word

    def normalize(self, word, word_id):
        """ Find the closest alternative in case the word is OOV."""
        if word not in self.word_id:
            word = self.DIGITS.sub("#", word)
        if word not in word_id:
            word = self.case_normalizer(word, word_id)

        if word not in word_id:
            return None
        return word

    def cos_nearest(self, word_index, k):
        """Sorts words according to their Cosinus distance."""

        e = self.embeddings[word_index]
        distances = []
        for line in self.embeddings:
            distances.append(1 - cosine_distance(e, line))
        sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
        return zip(*sorted_distances[:k])

    def knn(self, word, k=1):
        word = self.normalize(word, self.word_id)
        if not word:
            return
        word_index = self.word_id[word]
        indices, distances = self.cos_nearest(word_index, k)
        neighbors = [self.id_word[idx] for idx in indices]
        return zip(neighbors, distances)

    def closest_to_tokens(self, word, tokens):
        # return the close word (cosine similarity in pyglot dataset) that is part of tokens
        candidates = self.knn(word, k=100)
        if candidates:
            for word, _ in candidates:
                if word in tokens:
                    return word

        # if we don't find any close word, return random word
        return list(tokens)[0]
