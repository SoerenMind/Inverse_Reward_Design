import numpy as np
from agent_runner import run_agent
from utils import Distribution
from scipy.misc import logsumexp
from gradient_descent_test import get_likelihoods_from_feature_expectations



class Inference:
    def __init__(self, agent, mdp, env, beta, reward_space_true, reward_space_proxy, num_traject=1, prior=None):
        """
        :param agent: Agent object or subclass
        :param env: Environment object or subclass
        :param beta: Rationality constant for proxy reward selection
        :param reward_space_proxy: List of proxy reward functions (which should be 1-dim np arrays)
        :param reward_space_true: List of true reward functions (which should be 1-dim np arrays)
        :param num_traject: Number of trajectories the agent samples in planning (default 1)
        :param prior: A dictionary from tuple(reward) to probability for reward in reward_space_true. None=uniform.
        """
        # Or input env_type+s_terminal+etc
        # self.env = env
        self.agent = agent
        self.mdp = mdp
        self.env = env
        self.beta = beta
        # self.reward_space_proxy = reward_space_proxy
        self.reward_space_proxy = reward_space_proxy
        self.reward_space_true = reward_space_true
        self.true_reward_matrix = np.array(self.reward_space_true)
        self.num_traject = num_traject
        self.reset_prior()
        self.feature_expectations_dict = {}
        self.avg_reward_dict = {}
        self.make_reward_to_index_dict()
    # # @profile
    def get_prior(self, true_reward):
        '''Gets prior(true_reward) from prior vector.'''
        # if self.prior is None:
        #     num_rewards = len(self.reward_space_true)
        #     self.prior = np.ones(num_rewards) / num_rewards
        #     return np.true_divide(1, len(self.reward_space_true))
        # else: return self.prior[tuple(true_reward)]
        # else:
        # TODO: Compare performance to having 2D reward space array and doing reward_space.tolist.index(reward).
        index = self.reward_index[tuple(true_reward)]
        return self.prior[index]

    def update_prior(self, query, chosen_proxy, true_posterior=None):
        """Calculates posterior for given query and answer and replaces prior with the outcome. Deletes prior_avg.
        If true_posterior is given, it replaces the prior directly and updates the prior_avg."""
        if true_posterior is not None:
            self.prior = true_posterior
            try: del self.prior_avg
            except: pass
        else:
            if len(query) == 0: # Do nothing for empty query
                return
            try: del self.prior_avg
            except: pass
            self.prior = self.calc_and_save_posterior(chosen_proxy, query)

    def reset_prior(self):
        '''Resets to uniform prior'''
        try: del self.prior_avg
        except: pass
        num_rewards = len(self.reward_space_true)
        self.prior = np.ones(num_rewards) / num_rewards


    # @profile
    def calc_and_save_posterior(self, proxy_given, reward_space_proxy):
        """
        Calculates and caches all likelihoods and the normalizer for a given reward space and proxy_given.
        """
        self.likelihoods = {tuple(true_reward): self.get_likelihood(true_reward, proxy_given, reward_space_proxy)
                            for true_reward in self.reward_space_true}
        self.evidence = np.sum([lhood * self.get_prior(true_reward)
                                for true_reward, lhood in self.likelihoods.items()])
        likelihoods_vec = np.array([self.likelihoods[tuple(true_reward)] for true_reward in self.reward_space_true])
        posterior_vec = likelihoods_vec * self.prior / self.evidence
        return posterior_vec

    # def calc_and_save_feature_expectations(self,reward_space_proxy):
    #     """Currently not used"""
    #     for proxy in reward_space_proxy:
    #         self.get_feature_expectations(proxy)

    # @profile
    def get_avg_reward_for_post_averages(self,post_averages):
        """Write description"""
        N = post_averages.shape[0]
        feature_dim = len(self.reward_space_true[0])

        # Make feature_exp matrix
        feature_exp_matrix = np.zeros([N,feature_dim])
        for n, proxy in enumerate(post_averages):
            feature_exp_matrix[n,:] = self.get_feature_expectations(proxy)

        avg_reward_matrix = np.matmul(feature_exp_matrix, self.true_reward_matrix.T)

        return avg_reward_matrix

    def cache_lhoods(self):
        # TODO: Idea: Change reward_space_proxy in later iterations based on the posterior.
        N = len(self.reward_space_proxy)
        feature_dim = len(self.reward_space_true[0])
        K = len(self.reward_space_true)

        # Make feature_exp matrix
        self.feature_exp_matrix = np.zeros([N,feature_dim])
        for n, proxy in enumerate(self.reward_space_proxy):
            self.feature_exp_matrix[n,:] = self.get_feature_expectations(proxy)

        self.avg_reward_matrix = np.matmul(self.feature_exp_matrix, self.true_reward_matrix.T)
        self.log_lhood_numerator_matrix = self.beta * self.avg_reward_matrix
        self.lhood_numerator_matrix = np.exp(self.log_lhood_numerator_matrix)

        # Cache results for true rewards
        num_true_rewards = len(self.reward_space_true)
        self.feature_exp_matrix_true_rewards = np.zeros([num_true_rewards,feature_dim])
        for n, true_reward in enumerate(self.reward_space_true):
            self.feature_exp_matrix_true_rewards[n,:] = self.get_feature_expectations(true_reward)
        self.true_reward_avg_reward_matrix = np.matmul(self.feature_exp_matrix_true_rewards, self.true_reward_matrix.T)
        self.true_reward_avg_reward_vec = self.true_reward_avg_reward_matrix.diagonal()

        # log_Z_w, log_P_q_z, P_q_z, sum_to_1, Z_q, posterior, log_Z_q, post_ent, post_sum_to_1, log_post_ent, log_posterior \
        #     = get_likelihoods_from_feature_expectations(self.feature_exp_matrix, self.true_reward_matrix,
        #                                               self.beta, self.prior, self.feature_exp_matrix_true_rewards)


    # @profile
    def calc_posterior(self, query, get_entropy=False):
        """Returns a K x N array of posteriors and K x M array of posterior averages, where
        K is the query_size, N is the size of reward_space_true and M the number of features.

        :return: posteriors, posterior_averages
        """

        if len(query) == 0:
            ent_w = -np.dot(self.prior, np.log2(self.prior))
            return self.prior, ent_w, None

        idx = [self.reward_index_proxy[tuple(reward)] for reward in query]   # indexes of rewards in query
        log_Z = logsumexp(self.beta * self.avg_reward_matrix[idx, :], axis=0)   # log normalizer of likelihood
        log_numerators = self.log_lhood_numerator_matrix[idx, :]
        log_probs = log_numerators - log_Z
        probs = np.exp(log_probs)
        assert probs.sum(axis=0).round(3).all()  # Sum to 1?
        evidence = np.dot(probs, self.prior)
        posteriors = probs * self.prior / evidence.reshape([-1, 1])
        assert posteriors.sum(axis=1).round(3).all()    # Sum to 1?

        # Calculate entropies and return
        if get_entropy:
            ent_q = -np.dot(evidence, np.log(evidence))
            ent_w = -np.dot(self.prior, np.log(self.prior))
            # Conditional entropies
            # cond_ent_per_w = - np.dot(probs * self.prior, np.log(probs))
            cond_ent_per_w = - (probs * self.prior * np.log(probs)).sum(-1)
            cond_ent = cond_ent_per_w.sum()
            ent_w_given_q = cond_ent - ent_q + ent_w
            if ent_w_given_q < 0:
                pass
            return posteriors, ent_w_given_q, evidence

        # Calculate posterior averages
        # Do outside this function with returned posterior?
        post_averages = np.matmul(posteriors, self.true_reward_matrix)

        return posteriors, post_averages, evidence

        # Calculate log posterior
        # log_prior = np.log(self.prior)
        # log_evidence = ?
        # log_probs + log_prior - log_evidence

        # @profile
    def get_likelihood(self, true_reward, proxy, reward_space_proxy):
        """NOT IN USE!
        Calculates likelihood of proxy reward given true reward.
        Proxy selection is assumed to be Boltzman rational."""
        """Main vectorization
        -Cache feature_exp for whole proxy space in a matrix (cache self.matrix)
        -Multiply by true reward matrix to get all avg_rewards (cache self.matrix)
        -Take exp(avg_rewards)
        -Cache lhoods = exp(beta * reward_matrix)


        -Cache log_lhoods = np.log(lhoods)
            -Calculate on demand per query
            -Function: inference.get_likelihoods(true, query):  # Gets likelihoods for all proxies in query
                -Or: Function: inference.get_likelihoods_for_query:
                    -Calculate log_Z
                    -Calculate each get_likelihood(log_Z=given)
                -log_Z = logsumexp(lhoods[i,{j in q}])
                -log_P_ijq = log_lhoods[ij] - log_Z
                -Cache logsumexp for a query and then substract it from the log_lhood[ij]
        -Function: calc_and_save_posterior(proxy, query):
            -Do not cache (self.)likelihoods unless you use the same query again at some point
            -Calculate log_Z
            -Make probabilities-vector with get_likelihood
                -Use indexes for rewards (or an index map)
            -return posterior_vec = likelihoods_vec * self.prior / self.evidence
            -Cache post_avg?
        -What about posterior averages?
            -We can already calculate the likelihood of proxy choices and the posteriors.
            -We need post_reward but we've only got avg_reward matrix for proxy/true pairs.
                (-Precompute all the post averages | proxy, query? No because queries are unknown.)
                    -But maybe precompute for all queries you know you'll consider next. Make set_of_queries and do
                     calc_and_save_posterior for each answer. Get and cache feature_exp for all the post_averages.
                     Multiply with true_reward matrix to get avg_rewards given all true rewards.
                     -Get feature_exp for true_rewards too and get diagonal of multiplication with true_reward_matrix.
                -Don't need to get avg rewards / likelihoods for the post_averages!
        """
        # TODO: Cache sum of numerators for each query?
        expected_true_reward = self.get_avg_reward(proxy, true_reward)
        numerator = np.exp(self.beta * expected_true_reward)
        # Make sure floats don't become too large. Use log-likelihoods?
        Z_summands = [np.exp(self.beta * self.get_avg_reward(proxy_i, true_reward)) for proxy_i in reward_space_proxy]
        lhood = np.true_divide(numerator, np.sum(Z_summands))
        return lhood

    # def get_Z_constant(self, true_reward):
    #     Z_normalization = 0
    #     for proxy in self.reward_space_proxy:
    #         Z_normalization += self.get_likelihood(true_reward, np.array(proxy)) \
    #                            * self.get_prior(proxy)
    #     return Z_normalization

    # # @profile
    def get_posterior(self, true_reward):
        '''Just Bayes' rule'''
        # TODO (efficiency): Cache to save up to 10% of the time in get_exp_regret
        lhood = self.likelihoods[tuple(true_reward)]
        # lhood = self.get_likelihood(true_reward, proxy)
        # Z = self.get_Z_constant(true_reward)
        prior = self.get_prior(true_reward)
        if self.evidence == 0:
            print('Warning: evidence=0')
        return np.true_divide(lhood, self.evidence) * prior
    # # @profile
    def get_avg_reward(self, proxy, true_reward):
        """Calculates average true reward over num_runs trajectories when the agent optimizes the proxy reward."""
        # TODO (efficiency): Cache result for pairs
        """How to make more efficient?
        -First cache feature expectations (already doing) and then also avg_reward for proxy/true pairs.
            -Make sure to have proxy/true_reward as tuples already. Changing takes time.
        """
        reward = self.avg_reward_dict.get((tuple(proxy), tuple(true_reward)))   # Make sure these are already tuples
        if reward is None:
            feature_expectations = self.get_feature_expectations(proxy)
            reward = np.dot(feature_expectations, true_reward)
            self.avg_reward_dict[tuple(proxy), tuple(true_reward)] = reward
        # reward = self.mdp.get_reward_from_features(feature_expectations, true_reward)
        return reward

    def get_posterior_avg(self):
        return sum([true_reward * self.get_posterior(true_reward) for true_reward in self.reward_space_true])

    def get_prior_avg(self):
        """Calculates and returns prior average reward if it's not cached. Cached version is deleted in self.update_prior."""
        try:
            return self.prior_avg
        except:
            # Vectorize:
            self.prior_avg = np.array(sum([self.get_prior(true_reward) * true_reward
                                  for true_reward in self.reward_space_true]))
            return self.prior_avg

    # @profile
    def get_feature_expectations(self, proxy):
        """Given a proxy reward, calculates feature_expectations and returns them. Also stores them in a dictionary
        and reuses the result if it has previously been calculated.
        """
        try:
            feature_expectations = self.feature_expectations_dict[tuple(proxy)]
        except:
            self.mdp.change_reward(proxy)
            # self.mdp.set_feature_weights(proxy)
            iters = self.agent.set_mdp(self.mdp)  # Does value iteration
            trajectories = [run_agent(self.agent, self.env) for _ in range(self.num_traject)]
            # trajectory = [trajectories[0][t][0] for t in range(20)]
            # print trajectory
            # print(run_agent(self.agent, self.env, episode_length=10))
            for traject in trajectories:
                for t, tup in enumerate(traject):
                    traject[t] = tuple(list(tup)+[self.agent.gamma**t])
            feature_expectations = self.mdp.get_feature_expectations_from_trajectories(trajectories)
            # feature_expectations = np.true_divide(np.ones(shape=proxy.shape), len(proxy))
            self.feature_expectations_dict[tuple(proxy)] = feature_expectations
            num_plannings_done = len(self.feature_expectations_dict.items())
            if num_plannings_done % 25 == 0:
                print('Done planning for {num} proxies'.format(num=num_plannings_done))
                print iters
        return feature_expectations

    # # @profile
    def get_prob_proxy_choice(self, proxy, reward_space_proxy):
        """Calculates P(proxy) for a query by integrating P(proxy | true_reward) * P(true_reward)
        over reward_space_true."""
        # TODO: Test if this sums to 1 over proxies in Q
        p_proxy_chosen = 0
        for true_reward in self.reward_space_true:
            p_true_reward = self.get_prior(true_reward)
            p_proxy_given_true = self.likelihoods[tuple(true_reward)]
            p_proxy_chosen += p_true_reward * p_proxy_given_true
        return p_proxy_chosen

    def get_proxy_from_query(self, query, true_reward):
        """Chooses and returns a proxy."""
        if len(query) == 0:
            return None
        # idx = [self.reward_index_proxy[tuple(proxy)] for proxy in query]

        # Old approach
        lhoods = [self.get_likelihood(true_reward, proxy, query) for proxy in query]    # TODO: Use cached ones.
        # Replace with vector
        # lhoods = self.likelihood_dict[true_reward, query]
        d = {i: lhood for i, lhood in enumerate(lhoods)}
        # try:
        #     chosen_proxy_number = Distribution(d).sample()
        # except:
        #     chosen_proxy_number = np.array(lhoods).argmax()  # Replaces argmax with sampling
        chosen_proxy_number = Distribution(d).sample()
        chosen_proxy = query[chosen_proxy_number]
        return chosen_proxy

    def reset(self, reset_mdp=True):
        """Resets feature expecations, likelihoods, prior and (if chosen) MDP-features."""
        # Reset feature expectations, likelihoods, etc
        self.feature_exp_matrix = None
        self.avg_reward_matrix = None
        self.lhood_numerator_matrix = None
        self.log_lhood_numerator_matrix = None
        self.feature_exp_matrix_true_rewards = None
        self.true_reward_avg_reward_matrix = None
        self.true_reward_avg_reward_vec = None
        # Reset vars from old posterior calculation
        self.feature_expectations_dict = {}
        self.avg_reward_dict = {}
        self.likelihoods = {}
        self.reset_prior()
        if reset_mdp: self.mdp.populate_features()
        # Reset other cached variables if new ones added!

    def make_reward_to_index_dict(self):
        """This dictionary is used to find the cached posterior or prior of a reward function."""
        self.reward_index = {}
        self.reward_index_proxy = {}
        for i, true_reward in enumerate(self.reward_space_true):
            self.reward_index[tuple(true_reward)] = i
        for i, proxy in enumerate(self.reward_space_proxy):
            self.reward_index_proxy[tuple(proxy)] = i


def test_inference(inference, rfunc_proxy_given, reward_space):
    '''Tests if posterior adds up to 1 by calculating it for every possible true reward function.'''
    cum_post = 0
    # for true_reward in itertools.product([0,1],repeat=num_states):
    for true_reward in reward_space:
        post = inference.get_posterior(true_reward, rfunc_proxy_given)
        cum_post += post
    return cum_post



#
#
# if __name__=='__main__':
#     rewards = [0, 1, 2, 3, 4]
#     mdp = NStateMdp(num_states=5, rewards=rewards, start_state=0, preterminal_states=[3])
#     env = GridworldEnvironment(mdp)
#     # default_action = 1
#     # agent = DirectionalAgent(default_action)
#     agent = ImmediateRewardAgent()
#     agent.set_mdp(mdp)
#     print(run_agent(agent, env, episode_length=float(6)))
