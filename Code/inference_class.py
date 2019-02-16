import numpy as np


class Inference(object):
    def __init__(self, mdp, env, beta, reward_space_true, reward_space_proxy):
        """
        :param env: Environment object or subclass
        :param beta: Rationality constant for proxy reward selection
        :param reward_space_proxy: List of proxy reward functions (which should be 1-dim np arrays)
        :param reward_space_true: List of true reward functions (which should be 1-dim np arrays)
        """
        self.mdp = mdp
        self.env = env
        self.beta = beta
        self.reward_space_proxy = reward_space_proxy
        self.reward_space_true = reward_space_true
        self.true_reward_matrix = self.reward_space_true
        self.reset_prior()
        self.make_reward_to_index_dict()

    def update_prior(self, query, answer, true_log_posterior=None):
        """Calculates posterior for given query and answer and replaces prior with the outcome. Deletes prior_avg.
        If true_posterior is given, it replaces the prior directly and updates the prior_avg."""
        if true_log_posterior is not None:
            self.log_prior = true_log_posterior
            self.prior = np.exp(true_log_posterior)
        # TODO(rohinmshah): Can the elif case be removed? (soerenmind): It would break if query is None
        elif len(query) == 0: # Do nothing for empty query
            return
        else:
            raise ValueError('inference.get_full_posterior shouldnt be used')
            self.prior = self.get_full_posterior(query, answer)

    def reset_prior(self):
        '''Resets to uniform prior'''
        num_rewards = len(self.reward_space_true)
        self.log_prior = np.tile(-np.log(num_rewards), num_rewards)
        self.prior = np.exp(self.log_prior)

    def make_reward_to_index_dict(self):
        """Creates dictionary from proxy rewards (1D arrays) to their index in proxy space. If index_true_space, it does the
        same for the true reward space."""
        self.reward_index_proxy = {}
        for i, proxy in enumerate(self.reward_space_proxy):
            self.reward_index_proxy[tuple(proxy)] = i
