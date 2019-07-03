# stdlib
from os.path import join, dirname, realpath
from time import time
from collections import defaultdict
from itertools import product
# 3p
import numpy as np
# project
from tree import Tree
from utils import levenshtein_distance
from oov import OOV


class PCFG:
    def __init__(self):
        self.oov = OOV()
        self.train = []
        self.test = []
        self.poses = set()
        # ex// Un: 56
        self.tokens = defaultdict(int)
        # ex// (A, B): 41
        self.count_grammar = defaultdict(int)
        # ex// (A, a): 11
        self.count_lexicon = defaultdict(int)
        # ex// Un: [N, NP]
        self.token_to_pos = defaultdict(set)
        # ex// (B, C) : [A1, A2, A3]
        self.right_to_pos = defaultdict(set)
        # ex// A: 22
        self.preterminals_pos = defaultdict(int)
        # ex// (N, Un): 0.23
        self.prob_pos_to_token = defaultdict(int)
        # ex// (A, (B, C)): 56
        self.count_left_to_right = defaultdict(int)
        # ex// (A, (B, C)): 0.11
        self.prob_left_to_right = defaultdict(int)

    def from_path(self, path):
        """load and create dataset from a treebank in path"""
        dataset = open(join(dirname(realpath(__file__)), path), 'r').read().splitlines()
        np.random.shuffle(dataset)
        sep1, sep2 = int(len(dataset) * 0.8), int(len(dataset) * 0.9)
        self.train, self.test = dataset[:sep2], dataset[sep2:]

    def count_occurences(self):
        """count occurences for the different grammar rules, and compute probabilities"""
        for line in self.train:
            new_tree = Tree()
            new_tree.fit(line)
            for pos, _dict in new_tree.count_rules.items():
                self.poses.add(pos)
                for left, count in _dict.items():
                    self.count_grammar[(pos, left)] += count
                    self.right_to_pos[left].add(pos)
            for pos, _dict in new_tree.count_lexicon.items():
                for token, count in _dict.items():
                    self.count_lexicon[(pos, token[0])] += count
                    self.tokens[token[0]] += 1
                    self.token_to_pos[token[0]].add(pos)
        # compute proba for A --> token
        for (pos, token), count in self.count_lexicon.items():
            self.preterminals_pos[pos] += count
        for (pos, token), count in self.count_lexicon.items():
            self.prob_pos_to_token[(pos, token)] = count / self.preterminals_pos[pos]
        # compute proba for A --> BC
        for (pos, _), count in self.count_grammar.items():
            self.count_left_to_right[pos] += count
        for (pos, right_side), count in self.count_grammar.items():
            self.prob_left_to_right[(pos, right_side)] = count / self.count_left_to_right[pos]

    def fit(self):
        """Compute grammar probabilities"""
        self.count_occurences()
        self.proba_grammar = {**self.prob_pos_to_token, **self.prob_left_to_right}
        self.non_terminal = set([x[0] for x in self.proba_grammar.keys()])
        self.pos_2_ind = {pos: i for i, pos in enumerate(self.non_terminal)}
        self.ind_2_pos = {v: k for k, v in self.pos_2_ind.items()}

    def pcky(self, tokens):
        """Probabilistic CYK algorithm"""
        since = time()

        # normalize input: OOV module
        words = self.normalize_tokens(tokens)

        N = len(words)
        V = len(self.non_terminal)

        table = np.zeros((N+1, N+1, V))
        back = np.zeros((N+1, N+1, V), dtype=tuple)
        for j in range(1, N+1):
            for A in self.token_to_pos[words[j-1]]:
                table[j-1, j, self.pos_2_ind[A]] = self.proba_grammar[(A, words[j-1])]

            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    ind_B = np.nonzero(table[i, k, :] > 0)[0]
                    B_list = [self.ind_2_pos[x] for x in ind_B]
                    ind_C = np.nonzero(table[k, j, :] > 0)[0]
                    C_list = [self.ind_2_pos[x] for x in ind_C]
                    prod = product(B_list, C_list)
                    for BC in prod:
                        for A in self.right_to_pos[BC]:
                            indA, indB, indC = self.pos_2_ind[A], self.pos_2_ind[BC[0]], self.pos_2_ind[BC[1]]
                            value = (self.proba_grammar[(A, BC)]) * (table[i, k, indB]) * (table[k, j, indC])
                            if (table[i, j, indA]) < value:
                                table[i, j, indA] = value
                                back[i, j, indA] = (k, *BC)

        print("Took {}s".format(int(time() - since)))
        if not back[0, N, self.pos_2_ind["SENT"]]:
            return None
        tree = self.build_tree(tokens, back, 0, N, "SENT")
        return " ".join(self.debinarize(tree.split()))

    def build_tree(self, words, back, i, j, pos):
        """Transform the output of CYK to the form of a parsed sentence with parentheses"""
        n = j - i
        if n == 1:
            return " ( " + pos + " " + words[i] + " ) "
        else:
            k, B, C = back[i, j, self.pos_2_ind[pos]]
            return "( " + pos + " " + self.build_tree(words, back, i, k, B) + " " + self.build_tree(words, back, k, j, C) + ") "

    def debinarize(self, s):
        """Reverse Chomsky binarisation"""
        for i, x in enumerate(s):
            if "$" in x and s[i-1] == "(":
                c = 1
                for j, y in enumerate(s[i+1:]):
                    if y == '(':
                        c += 1
                    elif y == ")":
                        c -= 1
                    if c == 0:
                        return self.debinarize(s[:i-1] + s[i+1:i+1+j] + s[i+1+j+1:])
        return s

    def predict(self, line):
        """predict the parser of line from training dataset"""
        new = self.line_to_tokens(line)
        return self.pcky(new)

    def line_to_tokens(self, line):
        """transform a line from dataset to a list of tokens"""
        tokenized = line.replace("(", " ( ").replace(")", " ) ").split()[1:-1]
        remove = False
        new = []
        for i, x in enumerate(tokenized):
            if tokenized[i] == "(" and tokenized[i+1] != "(":
                remove = True
            elif tokenized[i] == "(":
                new.append(x)
            else:
                if not remove:
                    new.append(x)
                else:
                    remove = False
        new = list(filter(lambda x: x not in [')', '('], new))
        return new

    def prepare_line_for_prediction(self, line):
        """Tokenize a line from dataset"""
        tokenized = line.replace("(", " ( ").replace(")", " ) ").split()[1:-1]
        new = []
        for i, x in enumerate(tokenized):
            if "-" in x and tokenized[i-1] == "(":
                new.append(x.split("-")[0])
            else:
                new.append(x)
        return " ".join(new)

    def normalize_word(self, word):
        """OOV module, 1st: compute levenshtein_distance, if not, return closest word using cosinus similarity"""
        if word in self.tokens.keys():
            return word
        lv_distances = defaultdict(list)
        for token in self.tokens.keys():
            distance = levenshtein_distance(word, token)
            for i in range(1, 3):
                if distance == i:
                    lv_distances[i].append(token)
                    break
        for i in range(1, 3):
            if lv_distances[i]:
                return lv_distances[i][0]

        return self.oov.closest_to_tokens(word, self.tokens.keys())

    def normalize_tokens(self, tokens):
        """apply self.normalize_word to a list of tokens"""
        return [self.normalize_word(token) for token in tokens]
