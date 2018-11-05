from matrix_tools import argmax
from math import log, sqrt
import pyximport
pyximport.install()
import rd_fast


class EpsilonGreedy:
    """
    Determine whether to explore or exploit using epsilon-greedy approach.
    Which basically means choosing small constant of exploration.
    """
    def __init__(self, eps=.05, alpha='classic'):
        self.eps = eps
        if (type(alpha) is float and 0 < alpha < 1) or alpha == 'classic':
            self.alpha = alpha
        else:
            raise ValueError("Unknown alpha strategy:", alpha)

    def pick(self):
        explore = rd_fast.rand() < self.eps
        pick = argmax(self.values)
        if explore:
            pick = rd_fast.randint(len(self.values))
        return pick

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        if self.alpha == 'classic':
            new_value = (value * (n - 1) + reward) / n
        else:  # constant
            new_value = value - self.alpha * (value - reward)
        self.values[chosen_arm] = new_value

    def initialize(self, n_arms):
        self.values = [0] * n_arms  # estimated expected return from each arm
        self.counts = [0] * n_arms  # track number of pulls of each arm

    def __str__(self):
        return "EpsilonGreedy (eps={}, alpha={})".format(self.eps, self.alpha)


class EpsilonDecay(EpsilonGreedy):
    """
    EpsilonDecay is expected to perform worse than AnnealingEpsilon
    because of the linear decay, instead of logarithmic. This leads to
    slower convergence and poorer performance when compared to other
    explore-exploit techniques.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.t = 1

    def initialize(self, n_arms):
        super().initialize(n_arms)

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        self.eps = 1 / self.t
        self.t += 1

    def __str__(self):
        return "EpsilonDecay"


class AnnealingEpsilonGreedy(EpsilonGreedy):
    def __init__(self, *args):
        super().__init__(*args)
        self.t = 1

    def pick(self):
        eps = 1 / log(self.t + 1)
        self.t += 1
        pick = argmax(self.values)
        z = rd_fast.rand()
        if z < eps:
            pick = rd_fast.randint(len(self.values))
        return pick

    def __str__(self):
        return "AnnealingEpsilonGreedy"


class OptimisticInitialValues(EpsilonGreedy):
    """
    Optimistic Initial Values introduces bias to
    values estimation. This encourages agent to explore more.
    As it receives low rewards, it gets discouraged and converges
    to optimal arm.
    """

    def initialize(self, n_arms):
        super().initialize(n_arms)
        # this is important step for calculating mean
        self.counts = [1] * len(self.counts)
        # we specify optimistic expected values here
        self.values = [150] * len(self.values)

    def pick(self):
        return argmax(self.values)

    def __str__(self):
        return "OptimisticInitialValues"


class UCB1(OptimisticInitialValues):
    def initialize(self, n_arms):
        super().initialize(n_arms)
        self.total_count = 0
        self.ucb = lambda x: sqrt(2 * log(self.total_count + 1)/(self.counts[x] + .1))

    def pick(self):
        # x - index of the arm
        upper_bounds = [_ + self.ucb(i) for i, _ in enumerate(self.values)]
        return argmax(upper_bounds)

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        self.total_count += 1

    def __str__(self):
        return "UCB1"


class BernoulliTS(EpsilonGreedy):
    """
    Bernoulli Thomson Sampling strategy assumes each arm reward to be
    Bernoulli distributed with success probability Theta.
    Theta is estimated by using the expected value of Beta distribution,
    which tracks # of successes and # of failures for each arm.
    """

    def initialize(self, n_arms):
        super().initialize(n_arms)
        self.a = [1 for _ in range(n_arms)]
        self.b = [1 for _ in range(n_arms)]
        self.values = [a/(a+b) for a, b in zip(self.a, self.b)]

    def update(self, chosen_arm, reward):
        if reward >= 1:
            self.a[chosen_arm] += 1
        else:
            self.b[chosen_arm] += 1
        self.values[chosen_arm] = self.a[chosen_arm] / (self.a[chosen_arm] + self.b[chosen_arm])
        self.counts[chosen_arm] += 1
        #  scale a and b estimates proportionally, to make sampling from beta
        #  work moderately fast
        if self.b[chosen_arm] > 10 or self.a[chosen_arm] > 10:
            sum_ab = self.b[chosen_arm] + self.a[chosen_arm]
            self.a[chosen_arm] /= sum_ab / 2
            self.b[chosen_arm] /= sum_ab / 2

    def pick(self):
        """
        For each parameter, we draw a sample from it's assumed distribution
        """
        # beta distribution sampling works only when a >= 1, b >= 1
        # (there is another algorithm for 0 <= a <= 1.0, 0 <= b <= 1.0)
        return argmax([rd_fast.beta(a, b) for a, b in zip(self.a, self.b)])

    def __str__(self):
        return "Bernoulli Thomson Sampling"


class GaussianTS(EpsilonGreedy):
    """
    Thomson Sampling with assumption of Normal distribution
    """
    def initialize(self, n_arms):
        super().initialize(n_arms)
        self.t0 = 1
        self.sigmas = [1 for _ in range(n_arms)]
        self.sums = [0 for _ in range(n_arms)]
        self.lambdas = [1 for _ in range(n_arms)]
        self.values = [0 for _ in range(n_arms)]

    def update_mu(self, chosen_arm, reward):
        mu = self.values[chosen_arm]
        n = self.counts[chosen_arm]
        new_mu = (mu * (n - 1) + reward) / n
        self.values[chosen_arm] = new_mu
        return mu

    def update_sigma(self, chosen_arm, reward, old_mu):
        sigma = self.sigmas[chosen_arm]
        new_mu = self.values[chosen_arm]
        n = self.counts[chosen_arm]
        new_sigma = sqrt(sigma ** 2 + ((reward - old_mu)*(reward - new_mu) - sigma ** 2)/n)
        self.sigmas[chosen_arm] = new_sigma

    def update(self, chosen_arm, reward):
        """
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        Page 3
        :param chosen_arm:
        :param reward:
        :return:
        """
        self.counts[chosen_arm] += 1
        self.lambdas[chosen_arm] += self.t0
        self.sums[chosen_arm] += reward
        self.values[chosen_arm] = self.sums[chosen_arm] / (1 + self.lambdas[chosen_arm])
        self.update_sigma(chosen_arm, reward, self.values[chosen_arm])

    def pick(self):
        """
        The idea is to use our estimates of distribution parameters to generate
        random sample from them as if they were our target distributions and
        make our choice based on this samples.
        :return: id of Bandit with maximum estimated expected value
        """
        return argmax([rd_fast.gauss(mu, l) for mu, l in zip(self.values, self.sigmas)])

    def __str__(self):
        return "Gaussian Thomson Sampling"
