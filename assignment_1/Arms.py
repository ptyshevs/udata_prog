import pyximport
pyximport.install()
import rd_fast


class Arm(object):
    def __init__(self):
        """
        Arm of a slot-machine
        """

    def __repr__(self):
        par_str = ', '.join([f"{par}={val}" for par, val in self.__dict__.items()])
        return self.__class__.__name__ + f" ({par_str})"

    def draw(self):
        """
        Draw a sample from this slot-machine
        :return: reward
        """
        pass


class BernoulliArm(Arm):
    def __init__(self, p=.5, r=1):
        """
        Bernoulli Arm is a Bernoulli process, parametrized by success probability
        :param p: Probability of success
        :param r: Reward associated with success
        """
        super().__init__()
        self.p = p
        self.r = r

    def draw(self):
        """
        Draw a sample z from uniform distribution [0, 1).
        If z < p, return a reward, otherwise no reward.
        :return: reward
        """
        return self.r if rd_fast.rand() < self.p else 0


class GaussianArm(Arm):
    def __init__(self, mu=0.0, sigma=1.0):
        """
        Gaussian Arm is a univariate Gaussian random variable
        :param mu: mean
        :param sigma: SD
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def draw(self):
        """
        Draw a sample from Gaussian distribution.
        :return: reward received (can be negative)
        """
        return rd_fast.gauss(self.mu, self.sigma)


class NonstationaryArm(GaussianArm):
    def __init__(self, mu=0.0, sigma=1.0, eps=.001):
        """
        Non-stationary Gaussian process
        :param mu: mean
        :param sigma: SD
        :param eps: change constant
        """
        super().__init__(mu, sigma)
        self.n = 0
        self.eps = eps

    def draw(self):
        self.n += self.eps
        return rd_fast.gauss(self.mu + self.n, self.sigma)


class ExponentialArm(Arm):
    def __init__(self, scale=1.0):
        """
        Draw a reward from exponential distribution
        :param scale: inverse of rate parameter
        """
        super().__init__()
        self.scale = scale

    def draw(self):
        return rd_fast.exponential(self.scale)


if __name__ == '__main__':
    b = GaussianArm()
    print([b.draw() for _ in range(100)])
    print(b)
