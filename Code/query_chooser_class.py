from itertools import combinations
from random import choice


class Query_Chooser(object):
    def __init__(self):
        pass

class Regret_Minimizing_Query_Chooser(Query_Chooser):
    def __init__(self, inference, reward_space_proxy, prior=None, cost_of_asking=0):
        super(Query_Chooser, self).__init__()
        self.inference = inference
        self.reward_space_proxy = reward_space_proxy
        # self.prior = prior
        self.cost_of_asking = cost_of_asking

    def cache_feature_expectations(self):
        """Calculates feature expectations for all proxies and stores them in the dictionary
        inference.feature_expectations_dict. This function is only needed if you want to front-load these computations."""
        for proxy in self.reward_space_proxy:
            feature_expectations = self.inference.get_feature_expectations(proxy)

    def generate_set_of_queries(self, query_size=4):
        set_of_queries = combinations(self.reward_space_proxy, query_size)
        return list(set_of_queries)

    def find_regret_minimizing_query(self, set_of_queries, test_before_query=True):
        """Calculates the expected posterior regret after asking for each query of size query_size. Returns the query
        that minimizes this quantity plus the cost for asking (which is constant for fixed-size queries).
        :param query_size: number of reward functions in each query considered
        :return: best query
        """
        self.cache_feature_expectations()
        best_query = []
        if test_before_query:
            best_exp_exp_post_regret = self.get_exp_exp_post_regret([])
        else: best_exp_exp_post_regret = float("inf")
        best_regret_plus_cost = best_exp_exp_post_regret
        # Find query with minimal regret
        for query in set_of_queries:
            exp_exp_post_regret = self.get_exp_exp_post_regret(query)
            query_cost = self.cost_of_asking * len(query)
            regret_plus_cost = exp_exp_post_regret + query_cost
            if regret_plus_cost < best_regret_plus_cost:
                best_regret_plus_cost = regret_plus_cost
                best_exp_exp_post_regret = exp_exp_post_regret
                best_query = query
        return best_query, best_exp_exp_post_regret, best_regret_plus_cost

    def find_best_query_greedy(self, query_size=4):
        """Finds query of size query_size that minimizes expected regret by starting with a random proxy and greedily
        adding more proxies to the query.
        Not implemented: Query size could be chosen adaptively based on cost vs gain of bigger queries.
        :param query_size: int
        :return: best_query, corresponding regret and regret plus cost of asking (which grows with query size).
        """
        cost_of_asking = self.cost_of_asking    # could use this to decide query length
        self.cache_feature_expectations()
        best_exp_exp_post_regret = float("inf")
        best_regret_plus_cost = best_exp_exp_post_regret
        # set_of_size_2_queries = self.generate_set_of_queries(query_size=2)
        # TODO: Compare with choosing from size 2 queries first
        # Find query with minimal regret
        best_query = [choice(self.reward_space_proxy)]  # Initialize randomly
        while len(best_query) < query_size:
            found_new = False
            # TODO: Use a function for this
            for proxy in self.reward_space_proxy:
                query = best_query+[proxy]
                exp_exp_post_regret = self.get_exp_exp_post_regret(query)
                query_cost = self.cost_of_asking * len(query)
                regret_plus_cost = exp_exp_post_regret + query_cost
                if regret_plus_cost <= best_regret_plus_cost:
                    best_regret_plus_cost = regret_plus_cost
                    best_exp_exp_post_regret = exp_exp_post_regret
                    best_query_new = query
                    found_new = True
            best_query = best_query_new
            try: assert found_new # If no better query was found the while loop will go forever. Use <= instead.
            except:
                assert found_new
        return best_query, best_exp_exp_post_regret, best_regret_plus_cost


    def get_exp_exp_post_regret(self, query):
        """Returns the expected regret after getting query answered. This measure should be minimized over queries.
        The calculation is done by calculating the probability of each answer and then the regret conditioned on it."""

        if len(query) == 0:
            return self.get_exp_regret(no_query=True)

        exp_exp_post_regret = 0
        for proxy in query:
            self.inference.calc_and_save_posterior(proxy, query)
            p_proxy_chosen = self.inference.get_prob_proxy_choice(proxy, query)
            exp_regret = self.get_exp_regret()
            exp_exp_post_regret += p_proxy_chosen * exp_regret
        return exp_exp_post_regret

    def get_exp_regret(self, no_query=False):
        """Returns the expected regret if proxy is the answer chosen from query.
        Proxy and query are not used here because the likelihoods in self.inference are already conditioned on them!
        Calculates the posterior over true rewards. Assumes that the posterior average will be optimized (greedy assumption).
        Then calculates the expected regret which is the expected difference between the reward if optimizing the true
        reward and the reward if optimizing the posterior average.
        """
        # inference has to have cached the posterior for the right proxy & query here.
        exp_regret = 0

        # If no query, return prior regret
        if no_query:
            prior_avg = sum([self.inference.get_prior(true_reward) * true_reward
                             for true_reward in self.inference.reward_space_true])
            for true_reward in self.inference.reward_space_true:
                p_true_reward = self.inference.get_prior(true_reward)
                optimal_reward = self.inference.get_avg_reward(true_reward,true_reward)
                prior_reward = self.inference.get_avg_reward(prior_avg, true_reward)
                regret = optimal_reward - prior_reward
                exp_regret += regret * p_true_reward
            return exp_regret

        # Get expected regret for query
        post_avg = self.inference.get_posterior_avg()   # uses cached posterior
        for true_reward in self.inference.reward_space_true:    # Vectorize
            p_true_reward = self.inference.get_posterior(true_reward)
            optimal_reward = self.inference.get_avg_reward(true_reward, true_reward)    # Cache to save 9%
            # True reward for optimizing post_avg
            post_reward = self.inference.get_avg_reward(post_avg, true_reward)
            regret = optimal_reward - post_reward
            exp_regret += regret * p_true_reward
        return exp_regret
        # posterior = {tuple(true_reward): self.inference.get_posterior(true_reward)
        #                   for true_reward in self.inference.reward_space_true}
        # exp_regret = self.get_regret(posterior, query)