# Multi-arm bandit problem

This document describes specification of the problem, outline of the solutions
and implementation details.

## Problem statement
For this assignment, we are required to solve Multi-arm bandit problem. It's
specification is as follows:
> You have `m` slot-machines ("bandits"), each with one arm. When you pull an arm,
you get a reward, chosen randomly from some underlying distribution. You have `k` trials,
for each trial you select one slot-machine out of `m` and pull it's arm, receiving the reward.
Your task is to maximize the cumulative reward for `k` trials.

## Solutions outline

Key insight that I've made is that our tasks boils down to making estimation of the
Expected Value -- long-run average reward of each Arm. For this we're keeping track
of how many times have we pulled a particular arm, and calculate the running average
$$\mu_n = \frac{\mu_{n-1}(n-1) + reward}{n}$$.

## Drawing samples from known distributions
In order to generate random rewards from different distributions,
we were required to write our own routines (`rd_fast.pyx`).

One of the most important algorithms is to generate random number from
uniform distribution in [0, 1) range (`rand`). For this we declare 3 big
positive numbers `a`, `c`, and `m`, and iteratively calculate
$$X_n = (aX_{n-1} + c) % m$$
To scale `X_n` to the required interval, we further divide my `m` - this
guarantees us [0, 1) range, since we've taken modulo `Z % m`, thus the
resulting number is strictly smaller then `m`.

This routine is further used in drawing samples from Gaussian distribution.
As proposed in Box-Muller transform, we generate two samples from `rand`
and consider them as a coordinates somewhere inside the unit circle on the
Cartesian plane. By performing non-linear transformation, we obtain samples
from Standard Normal distribution, scaling it if required.

There are several other routines implemented: `randint`, `exponential`, and `beta`.
I've chosen Cython, because this is computational-intensive task. As a
result, all routines outperform their counterparts from `random` library
on the scale of tens times as fast.

More information about the subject you can find [here](https://www.springer.com/cda/content/document/cda_downloaddocument/9780387781648-c1.pdf?SGWID=0-0-45-733854-p173882714).
Both naive and cythoned implementations can be found in `sampling.ipynb`,
as well as PDF plots and execution time measurement.

## Arm simulation
In `Arms.py` you can find classes that implement the particular distribution
of the reward. They are `BernoulliArm`, `GaussianArm`, `ExponentialArm`,
and `NonstationaryArm`.

## Strategies
In `Strategy.py` there are implementations of the most common strategies
that solve the multi-arm bandit problem. Here is a quick review:

#### `EpsilonGreedy`
Before we discuss this strategy, let's reformulate our problem a bit. As
we've said, every time we pull a particular bandit, we update the estimate
of the expected value of the reward. If our current estimate is correct,
we can maximize the reward simply by playing bandit with the highest associated
expected value. However, there is a chance that we're making suboptimal
choice and if we've explored other options more, we could find a better
slot-machine to play. This problem is known as explore-exploit trade-off.
The most common strategy is to set some small constant probability of
exploration, while exploiting current best arm in the rest of cases.
This is implemented in `EpsilonGreedy` class. Apart from exploration
probability `eps`, there is another parameter `alpha` which controls
the calculation of running average. If we suspect Arm to be non-stationary,
constant alpha will prevent us from convergence to suboptimal expected
value estimate.

#### `EpsilonDecay` and `AnnealingEpsilonGreedy`
We can further relax the constraint of constant probability of exploration
by decaying it over time (`EpsilonDecay`) at a constant rate, or as a
reciprocal of logarithm (`AnnealingEpsilonGreedy`).

#### `OptimisticInitialValues`
The idea is to make optimistic initial estimates of the expected values
of each arm. Optimism in the fact of uncertainty will encourage
exploration early during a game, converging to purely greedy strategy later on.

#### `UCB1`
This approach uses Upper Confidence Bound on the expected value as an
estimate we're maximizing over. Explained in great detail [here](https://jeremykun.com/2013/10/28/optimism-in-the-face-of-uncertainty-the-ucb1-algorithm/).

#### `BernoulliTS` and `GaussianTS`
[Thomson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling) is another
approach towards solving the explore-exploit dilemma. It does this by making
an assumption about the shape of the distribution underlying the reward function.
We're sampling from the distributions with the estimates we've built so far, and
then update our parameter estimates based on the outcome. To update parameters,
we must use Bayes rule and know conjugate priors. This results in surprisingly
accurate estimate of true parameters.

## `MultiarmBandit`

In order to collect the fruits of our hard labor, we create class
`MultiarmBandit`, which takes as a parameter a list of (possibly heterogeneous)
arms. We can then specify strategy using `strategy` attribute and perform
a simulation using `simulate` method. To plot the estimates (in case of `BernoulliArm`'s
or `GaussianArm`'s), use `plot` method.

To compare various strategies use `compare` method, passing it a list of
strategy objects.
