from time import time
import numpy as np
from .utils import collect_episodes

# Updates


def constant_update(theta, lr=0.01):
    return lr * theta


class GaussianPolicy(object):
    """Gaussian policy with parametrized mean (theta*state) and fixed standard
    deviation.
    """

    def __init__(self, theta, sigma=0.5):
        """
        Parameters
        ----------
        theta : float
            LGQ parameter.
        sigma : float, optional
            Standard deviation.
        """
        self.theta = theta
        self.sigma = sigma

    def draw_action(self, state):
        """Draw an action for a state."""
        self.mu = self.theta*state
        return np.random.normal(self.mu, self.sigma)

    def grad_J(self, paths, discounts, n_ep, T):
        l_1 = []
        for n in range(n_ep):
            l_2 = []
            for t in range(T):
                action = paths[n]["actions"][t]
                state = paths[n]["states"][t]
                self.mu = self.theta * state
                temp1 = ((action - self.mu) * state) / (self.sigma**2)

                temp2 = np.dot(discounts[t:], paths[n]["rewards"][t:])
                l_2.append(temp1 * temp2)
            l_1.append(sum(l_2))
        dJ = int(sum(l_1))
        dJ /= n_ep
        return dJ


class Reinforce:
    """
        Implement Reinforce algorithm with Gaussian policy model

        Parameters
        ----------
        sim : object
            Simulator.
        bonus : int, choices=[0, 1]
            If 1, Apply MBIE-BE
        update_rule : string, choices = ["constant"]
            Update rule to apply
        lr: float
            Learning rate for update
        n_episodes : int, optional
            Number of episodes per iteration.
        T : int, optional
            Trajectory horizon
        n_itr : int, optional
            Number of iteration of the algorithm.
        gamma : float, optional
            Discount factor.
        sigma: float
            Standard deviation for the gaussian policy.
        """

    def __init__(self, sim, bonus=0, update_rule="constant", lr=0.01, n_ep=100, T=100, n_itr=100, gamma=0.9, sigma=0.4):
        self.sim = sim
        if update_rule == "constant":
            self.update_rule = constant_update

        self.lr = lr
        self.n_episodes = n_ep
        self.T = T
        self.n_itr = n_itr
        self.gamma = gamma
        self.sigma = sigma
        self.bonus = bonus

        self.theta = 0

        self.discounts = np.array([gamma**t for t in range(T)])
        self.theta_history = []
        self.avg_returns = []

    def _compute_performance(self, paths):

        J = 0
        for p in paths:
            df = 1
            sum_r = 0.
            for r in p["rewards"]:
                sum_r += df * r
                df *= self.gamma
            J += sum_r
        return J / self.n_episodes

    def compute_optimal_policy(self):
        """Compute the optimal policy"""

        self.theta_history.append(self.theta)

        since = time()
        for it in range(self.n_itr):
            print("lr: {} | Iteration N: {} \r".format(self.lr, it), end="")

            self.policy = GaussianPolicy(self.theta, self.sigma)

            # Simulate N trajectories
            paths = collect_episodes(
                self.sim, policy=self.policy, horizon=self.T, n_episodes=self.n_episodes)

            avg_return = self._compute_performance(paths=paths)
            self.avg_returns.append(avg_return)

            # Gradient update
            self.theta += self.update_rule(self.policy.grad_J(
                paths, self.discounts, n_ep=self.n_episodes, T=self.T), lr=self.lr)

            # History update
            self.theta_history.append(self.theta)

        # print("\nTook {}s".format(round(time() - since, 2)))
        print("lr: {} | Iteration N: {} | Took: {}s".format(self.lr, self.n_itr, round(time() - since, 2)))
