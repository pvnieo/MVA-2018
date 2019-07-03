# stdlib
from collections import defaultdict
# 3p
# uncomment to enable drawing
# from graphviz import Digraph


class Node:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        if self.name.count("@") == 2:
            return self.name.split('@')[1]
        else:
            return self.name


class Tree:
    def __init__(self):
        self.edges = []
        self.POS = []
        self.tokens = []
        self.rules = defaultdict(list)
        self.count_lexicon = defaultdict(dict)
        self.count_rules = defaultdict(dict)

    def line_to_list(self, line):
        """Transform a line into a nested lists of the different words
           ex: (SENT(NP ok)) ==> ['SENT', ['NP', 'ok']]

           Input:
           ------
           line: str
                string of input

            output:
            -------
            list: output list
           """
        nodes = []
        count = 0
        tokenized = line.replace("(", " ( ").replace(")", " ) ").split()[1:-1]
        functional = False
        for i, token in enumerate(tokenized):
            if token == "(":
                count += 1
                tokenized[i] = '['
                functional = True
            elif token == ')':
                count -= 1
                tokenized[i] = '],'
                functional = False
            else:
                if functional:
                    token = token.split("-")[0]
                    token = "{}@{}".format(count, token)
                    s = sum([token in x for x in nodes])
                    token += '@' + str(s+1)
                    nodes.append(token)
                if token == '"':
                    tokenized[i] = "'\"',"
                elif token == "'":
                    tokenized[i] = '"\'",'
                else:
                    tokenized[i] = '"{}",'.format(token)
                functional = False
        return eval("".join(tokenized)[:-1])

    def get_edges(self, _tree, edges=[], node_name="S0"):
        """Get all the edges of the tree corresponding to list returned by self.line_to_list"""
        if isinstance(_tree, list) and _tree:
            if isinstance(_tree[0], list):
                self.get_edges(_tree[0][0], edges, node_name)
                self.get_edges(_tree[1:], edges, node_name)
                node_name = _tree[0][0] if isinstance(_tree[0][0], str) else node_name
                for _list in _tree[0][1:]:
                    self.get_edges(_list, edges, node_name)
            elif isinstance(_tree[0], str):
                edges.append((Node(node_name), Node(_tree[0])))
                return self.get_edges(_tree[1:], edges, _tree[0])
        elif isinstance(_tree, str):
            edges.append((Node(node_name), Node(_tree)))
        return edges

    def get_nodes(self):
        """Create two sets, one with only leafs of the tree(tokens), and the second is with the rest of nodes(POS)"""
        self.POS = set([x[0].name for x in self.edges])
        child_nodes = set([x[1].name for x in self.edges])
        self.tokens = child_nodes - self.POS

    def fit(self, line):
        """Construct the tree from a given line input"""
        listed_line = self.line_to_list(line)
        self.edges = self.get_edges(listed_line, [])[1:]
        self.get_nodes()
        self.get_rules()

    def get_rules(self):
        """Apply Chomsky normalization by removing production units, and BINs,
        and add extracted grammar rules to self.count_rules"""
        for n1, n2 in self.edges:
            self.rules[n1.name].append(n2.name)
        # CNF: Eliminate BIN
        self.eliminate_bin()
        # CNF: Eliminate unit
        self.eliminate_unit()
        # Normalize names and get count
        for left, right in self.rules.items():
            norm_left = self.normalize_name(left)
            norm_right = self.normalize_list(right)
            if norm_left in self.count_rules.keys() and len(norm_right) == 2:
                self.count_rules[norm_left][tuple(norm_right)] += 1
            elif norm_left in self.count_lexicon.keys() and len(norm_right) == 1:
                self.count_lexicon[norm_left][tuple(norm_right)] += 1
            else:
                new_right = defaultdict(int)
                new_right[tuple(norm_right)] += 1
                if len(norm_right) == 1:
                    self.count_lexicon[norm_left] = new_right
                elif len(norm_right) == 2:
                    self.count_rules[norm_left] = new_right
                else:
                    raise Exception("CNF not done correctly!")

    def sort_nodes(self, nodes):
        """Sort nodes based on their names, NOT USED"""
        return sorted(nodes, key=lambda x: x.split("@")[1])

    def normalize_name(self, name):
        """Remove from name all attributes indicating repeatition and level"""
        splitted = name.split("$")
        for i, x in enumerate(splitted):
            if x.count('@') == 2:
                splitted[i] = x.split("@")[1]
            else:
                splitted[i] = x
        return "$".join(splitted)

    def normalize_list(self, _list):
        """apply self.normalize_name on a list of names"""
        return [self.normalize_name(x) for x in _list]

    def eliminate_bin(self):
        """Chomsky normalization: Eliminate right-hand sides with more than 2 non-terminals"""
        repeat = False
        new_rules = defaultdict(list)
        for left, right in self.rules.items():
            if len(right) == 2:
                # new_rules[left] = self.sort_nodes(right)
                new_rules[left] = right
            elif len(right) > 2:
                # sorted_nodes = self.sort_nodes(right)
                sorted_nodes = right
                new_lefts = [left] + ["$".join(sorted_nodes[i:]) for i in range(1, len(right)-1)]
                new_rights = [[sorted_nodes[i], "$".join(sorted_nodes[i+1:])] for i in range(len(right) - 1)]
                for i in range(len(right) - 1):
                    new_rules[new_lefts[i]] = new_rights[i]
            else:
                new_rules[left] = right
        self.rules = new_rules
        if repeat:
            self.eliminate_bin()

    def eliminate_unit(self):
        """Chomsky normalization: Eliminate unit rules"""
        repeat = False
        new_rules = {}
        for left, right in self.rules.items():
            if isinstance(right, str):
                continue
            elif len(right) == 1 and right[0] not in self.tokens:
                new_rules[left] = self.rules[right[0]]
                self.rules[right[0]] = ""
                repeat = True
            else:
                new_rules[left] = right
        self.rules = new_rules
        if repeat:
            self.eliminate_unit()

    # to use if graphiz is installed
    """
    def draw(self):
        dot = Digraph()
        for x, y in self.edges:
            dot.edge(x.name, y.name)
        return dot
    """
