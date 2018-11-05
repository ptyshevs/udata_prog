import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from matrix_tools import argmax
from Arms import BernoulliArm, GaussianArm, ExponentialArm, NonstationaryArm
from Strategy import EpsilonGreedy, EpsilonDecay, AnnealingEpsilonGreedy, \
                     OptimisticInitialValues, UCB1, GaussianTS, BernoulliTS
import array


class MultiarmBandit:
    def __init__(self, bandits):
        """
        MultiarmBandit is a collection of Armed bandits of heterogeneous classes
        :param bandits: list of *Arm bandits
        """
        self.bandits = bandits
        bandits_expected_values = []
        for bandit in bandits:
            if type(bandit) is BernoulliArm:
                bandits_expected_values.append(bandit.p)
            elif type(bandit) is GaussianArm:
                bandits_expected_values.append(bandit.mu)
            elif type(bandit) is ExponentialArm:
                bandits_expected_values.append(bandit.scale)
        self.best_arm = argmax(bandits_expected_values)
        self.strategy = None

    def plot(self):
        """
        Plot true and estimated PDF for each bandit

        Note: works only for GaussianArm objects
        :return:
        """
        plot_type = None
        if all([type(bandit) is GaussianArm for bandit in self.bandits]):
            plot_type = 'gaussian'
        elif all([type(bandit) is BernoulliArm for bandit in self.bandits]):
            plot_type = 'bernoulli'
        elif all([type(bandit) is ExponentialArm for bandit in self.bandits]):
            plot_type = 'exponential'
        if plot_type is None:
            print("Plot is unavailable for this configuration of MultiarmBandit")
            return
        if plot_type == 'gaussian':
            for i, bandit in enumerate(self.bandits):
                mu_est, sigma_est = self.strategy.values[i], self.strategy.sigmas[i]
                x = np.arange(-10, 20, .001)
                true_pdf = scipy.stats.norm.pdf(x, bandit.mu, bandit.sigma)
                estimated_pdf = scipy.stats.norm.pdf(x, self.strategy.values[i], bandit.sigma)
                ax1 = plt.plot(x, true_pdf, alpha=.5,
                               label=f'True d{i+1}: (μ={bandit.mu:.3f}, σ^2={bandit.sigma ** 2:.3f})')
                ax2 = plt.plot(x, estimated_pdf, linestyle='--', alpha=.6,
                               label=f'Est. d{i+1}: (μ={mu_est:.3f}), σ^2={sigma_est ** 2:.3f}')
                ax2[0].set_color(ax1[0].get_color())
        elif plot_type == 'bernoulli':
            n_bandits = list(range(len(self.bandits)))
            true_p = [bandit.p for bandit in self.bandits]
            true_pos = [_ - .1 for _ in n_bandits]
            est_p = self.strategy.values
            est_pos = [_ + .1 for _ in n_bandits]
            plt.bar(true_pos, true_p, width=.3, hatch="/", edgecolor='black',
                    label=f'True probability success', )
            plt.bar(est_pos, est_p, width=.3, hatch="//", edgecolor='black',
                    label=f'Estimated probability success')
            plt.xticks(n_bandits, n_bandits)
            plt.title("True vs. Estimated success probability of Bernoulli Arms")
            plt.xlabel("Slot-machine")
            plt.ylabel("Probability of success")
        elif plot_type == 'exponential':
            raise NotImplementedError()
        plt.legend(loc='upper left')
        plt.show()
        
    def simulate(self, num_simulations, time_horizon: int):
        """
        Simulate <num_simulations> games, each with <time_horizon> trials

        Probability of selecting the best arm is a good measure
        of algorithm performance for a couple of reasons:
        1) It doesn't depend on the magnitude of reward
        2) It doesn't depend on the number of trials
        """
        all_rewards = np.zeros((num_simulations, time_horizon))
        prob_best = np.zeros(num_simulations)
        for n in range(num_simulations):
            self.strategy.initialize(len(self.bandits))
            for i in range(time_horizon):
                idx = self.strategy.pick()
                if idx == self.best_arm:
                    prob_best[n] += 1
                reward = self.bandits[idx].draw()
                if type(reward) is array.array:
                    reward = reward[0]
                all_rewards[n, i] = reward
                self.strategy.update(idx, reward)
            prob_best[n] /= time_horizon
        avg_total_reward = all_rewards.sum(axis=1).mean()
        return prob_best.mean(), avg_total_reward
    
    def compare(self, *strategies):
        avg_prob_best = np.zeros(len(strategies))
        avg_total_reward = np.zeros(len(strategies))
        for i, strategy in enumerate(strategies):
            self.strategy = strategy
            avg_prob_best[i], avg_total_reward[i] = self.simulate(100, 2000)
            print(f"{strategy}: optimal choice prob: {avg_prob_best[i]},"
                  f" avg reward: {avg_total_reward[i]}")


if __name__ == '__main__':
    # ma = MultiarmBandit([BernoulliArm(_) for _ in [.2, .9]])
    ma = MultiarmBandit([GaussianArm(1, 1.2), GaussianArm(-1, 1.2), GaussianArm(3, 2), GaussianArm(15, 15)])
    # ma = MultiarmBandit([NonstationaryArm(1, 1.2), NonstationaryArm(3, 2), NonstationaryArm(15, 15)])
    # ma.strategy = GaussianTS()
    # print(ma.simulate(1, 1000))
    # print(ma.strategy.values)
    # ma.plot()
    ma.compare(EpsilonGreedy(), EpsilonGreedy(alpha=.05),
               AnnealingEpsilonGreedy(), OptimisticInitialValues(alpha=.05),
               UCB1(),
               BernoulliTS(), GaussianTS())
