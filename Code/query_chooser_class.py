from itertools import combinations

def choose_regret_minimizing_proposal(set_of_proposal_sets, reward_space_true, prior, inference, cost_of_asking = 0.01):
    '''Chooses a proposal reward sub space by minimizing the expected regret after asking.'''
    """Todo:
    -Functions for looping through sets, exp_post_regret(proxy, omega), get_post_avg(posterior), get_posterior(proxy, omega)
    """
    prior_avg = sum([prior[tuple(reward)] * reward for reward in reward_space_true]) # Over whole reward_space (change?)
    best_r_set = []
    best_exp_exp_post_regret = np.inf
    best_regret_plus_cost = np.inf
    set_of_proposal_sets = list(set_of_proposal_sets)[::-1]
    for omega in set_of_proposal_sets:
        # TODO: Restricted to length 2!
        if not len(omega)==2: continue
        # if not omega == list(set_of_proposal_sets)[2]: continue

        cost_omega = cost_of_asking*len(omega)
        prior_regret = 0
        exp_exp_post_regret = 0
        for proxy in omega:
            # TODO: Extremely costly to get posterior for all proxy choices. Save repeated computations?
            # Do I have to do (and thus save) the planning for every proxy here?
            inference.calc_and_save_posterior(proxy, reward_space_proxy=omega)    # Do only once per reward
            posterior = dict([(tuple(true_reward), inference.get_posterior(true_reward))
                              for true_reward in reward_space_true])
            post_avg = sum([posterior[tuple(reward)] * reward for reward in reward_space_true]) # Over whole reward_space
            # Calculate expected regret from optimizing post_avg (expectation over posterior true rewards)
            exp_post_regret = sum([posterior[tuple(true_reward)]  # multiply by regret
                     * (inference.get_avg_reward(true_reward,true_reward) - inference.get_avg_reward(post_avg,true_reward)) for true_reward in reward_space_true])
            # sum_true posterior(true | prox)* (avg_reward(true | true) - avg_reward(proxy | true))
            exp_exp_post_regret += exp_post_regret
        regret_plus_cost = exp_exp_post_regret + cost_omega
        if regret_plus_cost < best_regret_plus_cost:
            best_regret_plus_cost = regret_plus_cost
            best_exp_exp_post_regret = exp_exp_post_regret
            best_r_set = omega
            best_posterior = posterior
            best_post_avg = post_avg
    return best_r_set, best_exp_exp_post_regret, best_regret_plus_cost, best_posterior, best_post_avg

class Query_Chooser(object):
    def __init__(self,inference, reward_space_proxy, prior=None, cost_of_asking=0.01):
        self.inference = inference
        self.reward_space_proxy = reward_space_proxy
        self.prior = prior
        self.cost_of_asking = cost_of_asking

    def cache_feature_expectations(self):
        """Calculates feature expectations for all proxies and stores them in the dictionary
        inference.feature_expectations_dict. This function is only needed if you want to front-load these computations."""
        for proxy in self.reward_space_proxy:
            feature_expectations = self.inference.get_feature_expectations(proxy)

    def find_regret_minimizing_query(self, query_size=4):
        """Todo: Avoid double calculations
        -Calculate feature expectations for every proxy reward in advance

        :param query_size: number of reward functions in each query considered
        :return: best query
        """
        best_r_set = []
        best_exp_exp_post_regret = self.get_exp_exp_post_regret(best_r_set)
        best_regret_plus_cost = best_exp_exp_post_regret
        # Find query with minimal regret
        set_of_queries = combinations(self.reward_space_proxy, query_size)
        for query in set_of_queries:
            exp_exp_post_regret = self.get_exp_exp_post_regret(query)
            query_cost = self.cost_of_asking * len(query)
            regret_plus_cost = exp_exp_post_regret + query_cost
            if regret_plus_cost < best_regret_plus_cost:
                best_regret_plus_cost = regret_plus_cost
                best_exp_exp_post_regret = exp_exp_post_regret
                best_query = query
        return best_query, best_regret_plus_cost

    def get_exp_exp_post_regret(self,query):
        """Calculates the expected regret after getting query answered. This measure should be minimized over queries.
        The calculation is done by calculating the probability of each answer and then the regret conditioned on it."""
        for proxy in query:
            # p_proxy_chosen =