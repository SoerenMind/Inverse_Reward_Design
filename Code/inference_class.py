import numpy as np
from agent_runner import run_agent




class Inference:
    def __init__(self, agent, env, beta, reward_space_true, num_traject=1, prior=None):
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
        self.env = env
        self.beta = beta
        # self.reward_space_proxy = reward_space_proxy
        self.reward_space_true = reward_space_true
        self.num_traject = num_traject
        self.prior = prior
        self.feature_expectations_dict = {}

    def get_prior(self, true_reward):
        if self.prior is None:
            return np.true_divide(1, len(self.reward_space_true))
        else: return self.prior[tuple(true_reward)]

    def calc_and_save_posterior(self, proxy_given, reward_space_proxy):
        """
        Calculates and caches all likelihoods and the normalizer for a given reward space and proxy_given.
        """
        """Todo: reward spaces:
        -Pass reward_space_proxy into calc_and_save_posterior
        -Or chance self.reward_space_proxy
        -Make sure the right space is passed in run_IRD
        """
        """Todo: Efficiency
        -Likelihood(proxy | true) gets evaluated over all true rewards. Make this efficient.
        -The normalizer in the likelihood contains get_avg_reward(true | proxy) over all proxies. That's hard to cut.
        Plan:
            -Memoize all feature_expectations(proxy \in reward_space_proxy)
            -Calculate likelihood for each true reward
                -Inside the likelihood, for each proxy: 1) get cached feature expecations 2) dot them with the true reward
                -Or cache feature expectations and use them in get_avg_reward
            -The inefficiency is this: You loop through all the same proxies for each likelihood so you can reuse the feature expectations.
        """
        """
        Todo: make current self.proxy and self.query fixed.
        Todo: make
        """
        self.likelihoods = {tuple(true_reward): self.get_likelihood(true_reward, proxy_given, reward_space_proxy)
                            for true_reward in self.reward_space_true}
        self.evidence = np.sum([lhood * self.get_prior(true_reward)
                                for true_reward, lhood in self.likelihoods.items()])

    def calc_and_save_feature_expectations(self,reward_space_proxy):
        for proxy in reward_space_proxy:
            self.get_feature_expectations(proxy)

    def get_likelihood(self, true_reward, proxy, reward_space_proxy):
        """Calculates likelihood of proxy reward given true reward.
        Proxy selection is assumed to be Boltzman rational."""
        expected_true_reward = self.get_avg_reward(proxy, true_reward)  # TODO: Inefficient to calculate twice
        numerator = np.exp(self.beta * expected_true_reward)
        # Make sure floats don't become too large. Use log-likelihoods?
        Z_summands = [np.exp(self.beta * self.get_avg_reward(proxy_i, true_reward)) for proxy_i in reward_space_proxy]
        lhood = np.true_divide(numerator, np.sum(Z_summands))
        return lhood
        # return np.exp(self.beta*expected_true_reward)

    def get_Z_constant(self, true_reward):
        Z_normalization = 0
        for proxy in self.reward_space_proxy:
            Z_normalization += self.get_likelihood(true_reward, np.array(proxy)) \
                               * self.get_prior(proxy)
        return Z_normalization

    def get_posterior(self, true_reward):
        '''Just Bayes' rule'''
        lhood = self.likelihoods[tuple(true_reward)]
        # lhood = self.get_likelihood(true_reward, proxy)
        # Z = self.get_Z_constant(true_reward)
        prior = self.get_prior(true_reward)
        if self.evidence == 0:
            print('Warning: evidence=0')
        return np.true_divide(lhood, self.evidence) * prior

    def get_avg_reward(self, proxy, true_reward):
        """Calculates average true reward over num_runs trajectories when the agent optimizes the proxy reward."""
        feature_expectations = self.get_feature_expectations(proxy)
        return self.agent.mdp.get_reward_from_features(feature_expectations, true_reward)

    def get_posterior_avg(self):
        return sum([true_reward * self.get_posterior(true_reward) for true_reward in self.reward_space_true])

    def get_feature_expectations(self, proxy):
        """Given a proxy reward, calculates feature_expectations and returns them. Also stores them in a dictionary
        and reuses the result if it has previously been calculated.
        """
        try: feature_expectations = self.feature_expectations_dict[tuple(proxy)]
        except:
            self.agent.mdp.change_reward(proxy)
            trajectories = [run_agent(self.agent, self.env) for _ in range(self.num_traject)]
            feature_expectations = self.agent.mdp.get_feature_expectations_from_trajectories(trajectories)
            # feature_expectations = np.true_divide(np.ones(shape=proxy.shape), len(proxy))
            self.feature_expectations_dict[tuple(proxy)] = feature_expectations
            num_plannings_done = len(self.feature_expectations_dict.items())
            if num_plannings_done % 20 == 0:
                print('Done planning for {num} proxies'.format(num=num_plannings_done))
        return feature_expectations


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


def test_inference(inference, rfunc_proxy_given, reward_space):
    '''Tests if posterior adds up to 1 by calculating it for every possible true reward function.'''
    cum_post = 0
    # for true_reward in itertools.product([0,1],repeat=num_states):
    for true_reward in reward_space:
        post = inference.get_posterior(true_reward, rfunc_proxy_given)
        cum_post += post
    return cum_post



# class InferenceOld:
# def __init__(self,agent,beta,reward_space):
#         # Or input env_type+s_terminal+etc
#         # self.env = env
#         self.agent = agent
#         self.beta = beta
#         self.reward_space = reward_space
#     def get_prior(self, true_reward):
#         return np.true_divide(1, len(self.reward_space))
#     def get_likelihood(self, true_reward, rfunc_proxy):
#         self.agent.add_rfunc(rfunc_proxy)
#         expected_avg_reward = self.agent.get_avg_true_reward(true_reward)
#         return np.exp(self.beta*expected_avg_reward)
#     def get_Z_constant(self, true_reward):
#         Z_normalization = 0
#         for rfunc_proxy in self.reward_space:
#             Z_normalization += self.get_likelihood(true_reward, np.array(rfunc_proxy)) \
#                                * self.get_prior(rfunc_proxy)
#         return Z_normalization
#     def get_posterior(self, true_reward, rfunc_proxy):
#         '''Just Bayes' rule'''
#         lhood = self.get_likelihood(true_reward, rfunc_proxy)
#         Z = self.get_Z_constant(true_reward)
#         prior = self.get_prior(true_reward)
#         if Z == 0: print('Warning: Z=0')
#         return np.true_divide(lhood,Z) * prior
#
# class Determ_Inference(InferenceOld):
#     '''for testing purposes'''
#     def get_likelihood(self, true_reward, rfunc_proxy):
#         self.agent.add_rfunc(rfunc_proxy)
#         expected_avg_reward = self.agent.get_avg_true_reward(true_reward)
#
#
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