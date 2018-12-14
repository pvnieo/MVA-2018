# stdlib
import numpy as np


class Action:
    def __init__(self, name, state, directions, probabilities, reward):
        self.name = name
        self.state = state
        self.directions = directions
        self.probabilities = probabilities
        self.reward = reward


class MDP():
    def __init__(self, actions, discount_factor=0.95):
        self.discount_factor = discount_factor
        self._construct_proba_and_reward_matrices(actions)

    def _construct_proba_and_reward_matrices(self, actions):
        self.states = sorted(list(set([a.state for a in actions])))
        self.actions = sorted([a.name for a in actions])
        self.n_actions = len([1 for a in actions if a.state == self.states[0]])
        self.n_states = len(self.states)
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        for action in actions:
            i_a = self.actions.index(action.name) % self.n_actions
            i_s = self.states.index(action.state)
            for i, direction in enumerate(action.directions):
                i_dir = self.states.index(direction)
                self.P[i_a, i_s, i_dir] = action.probabilities[i]
                self.R[i_s, i_a] = action.reward

    def apply_bellman_operator(self, W):
        W = W.reshape(-1, 1)
        return self.R + self.discount_factor * self.P.dot(W).T.reshape(self.n_states, self.n_actions)

    def return_greedy_policy(self, v):
        tv = self.apply_bellman_operator(v)
        return np.argmax(tv, axis=1)

    def apply_optimal_bellman_operator(self, W):
        tW = self.apply_bellman_operator(W)
        return np.max(tW, axis=1)

    def value_iterate(self, v0=None, eps=1e-2):
        v = np.random.rand(self.n_states) if v0 is None else v0
        hist = [v]
        i = 1
        while True:
            tv = self.apply_optimal_bellman_operator(v)
            hist.append(tv)
            if np.max(np.abs(tv - v)) < eps:
                v = tv
                break
            v = tv
            i += 1
        print("Number of iteration for epsilon = {} : {} iterations".format(eps, i))
        return self.return_greedy_policy(v), hist

    def policy_evaluation(self, pi):
        term2 = self.R[np.arange(len(self.R)), pi]
        term1 = np.eye(len(pi)) - self.discount_factor * self.P[pi, np.arange(self.n_states), :]
        return np.linalg.inv(term1).dot(term2)

    def exact_policy_iterate(self, pi0):
        pi = np.random.rand(self.n_states) if pi0 is None else pi0
        hist = [pi]
        vo = self.policy_evaluation(pi)
        pi = self.return_greedy_policy(vo)
        hist.append(pi)
        i = 1
        while True:
            vn = self.policy_evaluation(pi)
            if (vn == vo).all():
                break
            pi = self.return_greedy_policy(vn)
            hist.append(pi)
            i += 1
            vo = vn
        print("Number of iterations for pi_0 = {} : {}".format(hist[0], i))
        return pi, hist
