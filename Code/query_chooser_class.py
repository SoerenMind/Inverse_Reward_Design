from itertools import combinations
from scipy.special import comb
from random import choice, sample, seed
import numpy as np
import time
# for test environment
from gridworld import NStateMdp, GridworldEnvironment, Direction, NStateMdpHardcodedFeatures, NStateMdpGaussianFeatures,\
    NStateMdpRandomGaussianFeatures, GridworldMdpWithDistanceFeatures, GridworldMdp
from agents import ImmediateRewardAgent, DirectionalAgent, ValueIterationLikeAgent
from inference_class import Inference
import csv
import os
import datetime
from planner import Model
import tensorflow as tf

def random_combination(iterable, r):
    """Random selection from itertools.combinations(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(xrange(n), r))
    return tuple(pool[i] for i in indices)

def time_function(function, input):
    "Calls function and returns time it took"
    start = time.clock()
    function(input)
    deltat = time.clock() - start
    return deltat


class Query_Chooser(object):
    def __init__(self):
        pass

class Query_Chooser_Subclass(Query_Chooser):
    def __init__(self, inference, reward_space_proxy, num_queries_max, args, prior=None, cost_of_asking=0):
        super(Query_Chooser_Subclass, self).__init__()
        self.inference = inference
        self.reward_space_proxy = reward_space_proxy
        # self.prior = prior
        self.cost_of_asking = cost_of_asking
        self.num_queries_max = num_queries_max
        self.args = args    # all args

    # def cache_feature_expectations(self):
    #     """Calculates feature expectations for all proxies and stores them in the dictionary
    #     inference.feature_expectations_dict. This function is only needed if you want to front-load these computations."""
    #     for proxy in self.reward_space_proxy:
    #         feature_expectations = self.inference.get_feature_expectations(proxy)

    def generate_set_of_queries(self, query_size=4):
        num_queries = comb(len(self.reward_space_proxy), query_size)
        if num_queries > self.num_queries_max:
            set_of_queries = [list(random_combination(self.reward_space_proxy, query_size)) for _ in range(self.num_queries_max)]
        else: set_of_queries = combinations(self.reward_space_proxy, query_size)
        return list(set_of_queries)

    # @profile
    def find_query(self, query_size, chooser, true_reward):
        """Calls query chooser specified by chooser (string)."""
        if chooser == 'maxmin':
            return self.find_query_feature_diff(query_size)
        elif chooser == 'exhaustive':
            return self.find_regret_minimizing_query(query_size)
        elif chooser == 'greedy_regret':
            return self.find_best_query_greedy(query_size)
        elif chooser == 'random':
            return self.find_random_query(query_size)
        elif chooser == 'no_query':
            return [], self.get_exp_regret_from_query([]), None
        elif chooser == 'greedy_exp_reward':
            return self.find_best_query_greedy(query_size, total_reward=True, true_reward=true_reward)
        elif chooser == 'greedy_entropy':
            return self.find_best_query_greedy(query_size, entropy=True)
        elif chooser == 'feature_entropy':
            return self.find_feature_query_greedy(query_size, 'entropy', true_reward)
        elif chooser == 'feature_variance':
            return self.find_feature_query_greedy(query_size, 'variance', true_reward)
        else:
            raise NotImplementedError('Calling unimplemented query chooser')

    # @profile
    def find_regret_minimizing_query(self, query_size, test_before_query=True):
        """Calculates the expected posterior regret after asking for each query of size query_size. Returns the query
        that minimizes this quantity plus the cost for asking (which is constant for fixed-size queries).
        :param query_size: number of reward functions in each query considered
        :return: best query
        """
        set_of_queries = self.generate_set_of_queries(query_size)
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

    # @profile
    def find_best_query_greedy(self, query_size, total_reward=False, entropy=False, true_reward=None):
        """Finds query of size query_size that minimizes expected regret by starting with a random proxy and greedily
        adding more proxies to the query.
        Not implemented: Query size could be chosen adaptively based on cost vs gain of bigger queries.
        :param query_size: int
        :return: best_query, corresponding regret and regret plus cost of asking (which grows with query size).
        """
        cost_of_asking = self.cost_of_asking    # could use this to decide query length
        best_query = [choice(self.reward_space_proxy)]  # Initialize randomly
        while len(best_query) < query_size:
            found_new = False
            best_objective = float("inf")
            best_objective_plus_cost = best_objective
            # TODO: Use a function for this
            for proxy in self.reward_space_proxy:    #TODO: vectorize - worth the time?
                query = best_query+[proxy]
                if entropy:
                    _, objective, _ = self.inference.calc_posterior(query, get_entropy=True)
                else:
                    objective = self.get_exp_exp_post_regret(query, total_reward)
                query_cost = self.cost_of_asking * len(query)
                objective_plus_cost = objective + query_cost
                if objective_plus_cost <= best_objective_plus_cost + 1e-15:
                    best_objective_plus_cost = objective_plus_cost
                    best_objective = objective
                    best_query_new = query
                    found_new = True
            best_query = best_query_new
            try: assert found_new # If no better query was found the while loop will go forever. Use <= instead.
            except:
                assert found_new

        posteriors, post_cond_entropy, evidence = self.inference.calc_posterior(query, get_entropy=True)
        proxy_choice = best_query[int(np.argwhere(np.random.multinomial(1, evidence) == 1))]
        # proxy_choice = self.inference.get_proxy_from_query(query, true_reward)

        self.inference.update_prior(query, proxy_choice)  # TODO: Still uses calc_and_save_posterior

        return best_query, best_objective, best_objective_plus_cost

    # def find_query_feature_diff(self, query_size=4):
    #     cost_of_asking = self.cost_of_asking    # could use this to decide query length
    #     prior_avg = np.array(sum([self.inference.get_prior(true_reward) * true_reward
    #                      for true_reward in self.inference.reward_space_true]))
    #     # Find query with minimal regret
    #     best_query = [choice(self.reward_space_proxy)]  # Initialize randomly
    #     while len(best_query) < query_size:
    #         max_min_diff = -float('inf')
    #         for proxy in self.reward_space_proxy:
    #             feature_exp_new = self.inference.get_feature_expectations(proxy)
    #             feature_exp_query = np.array([self.inference.get_feature_expectations(proxy2) for proxy2 in best_query])
    #             feature_exp_diffs = feature_exp_new - feature_exp_query
    #             feature_exp_diffs = feature_exp_diffs * prior_avg    # weighted diffs by prior. Doesn't make sense, try prior variance.
    #             feature_exp_diffs_norm = np.linalg.norm(feature_exp_diffs, 1, axis=1)
    #             min_diff = feature_exp_diffs_norm.min()
    #             if min_diff > max_min_diff:
    #                 max_min_diff = min_diff
    #                 best_new_proxy = proxy
    #         best_query = best_query + [best_new_proxy]
    #     return best_query, max_min_diff, None

    # @profile
    def find_feature_query_greedy(self, query_size, measure, true_reward):
        """Returns feature query of size query_size that minimizes the objective (e.g. posterior entropy)."""
        cost_of_asking = self.cost_of_asking    # could use this to decide query length
        best_feature_list = []
        feature_dim = self.args.feature_dim
        desired_outputs = [measure, 'optimal_weights']
        # desired_outputs = [measure]
        best_optimal_weights = None
        while len(best_feature_list) < query_size:
            found_new = False
            best_objective = float("inf")
            best_objective_plus_cost = best_objective
            for feature in range(feature_dim):
                feature_list = best_feature_list+[feature]

                (objective, optimal_weights) = self.calc_objective(feature_list, desired_outputs, init=best_optimal_weights)
                query_cost = self.cost_of_asking * len(feature_list)
                objective_plus_cost = objective + query_cost
                print('Model outputs calculated')
                if objective_plus_cost <= best_objective_plus_cost + 1e-14:
                    best_objective = objective
                    best_objective_plus_cost = objective_plus_cost
                    best_optimal_weights_new = optimal_weights
                    best_feature_list_new = feature_list
                    found_new = True
            try:
                assert found_new  # If no better query was found the while loop will go forever.
            except:
                assert found_new
            best_feature_list = best_feature_list_new
            best_optimal_weights = best_optimal_weights_new
            print 'Query length increased to {s}'.format(s=len(best_feature_list))

        print('query found')
        # For the chosen query, get posterior from human answer. If using human input, replace with feature exps or trajectories.
        # Add: Get all measures for data recording?
        desired_outputs = [measure,'true_posterior']
        objective, true_posterior = self.calc_objective(best_feature_list, desired_outputs,
                                                                  true_reward=true_reward, high_iters=True)
        return best_feature_list, objective, true_posterior

    # @profile
    def calc_objective(self, feature_list, desired_outputs, true_reward=None, init=None, high_iters=False):
        """
        Returns the desired model outputs after minimizing the desired measure over settings of fixed features.

        :param feature_list: List of integers. Specifies free features in previous query.
        :return: model_outputs: dictionary of model outputs, indexed by desired_outputs
        """
        mdp = self.inference.agent.mdp
        try: height, width = mdp.height, mdp.width
        except: height, width = None, None

        if high_iters:
            num_iters = 50
        else:
            num_iters = self.args.num_iters_optim

        proxy_reward_space = [[-1],[0],[1]]

        model = Model(self.args.feature_dim, height, width, self.args.gamma, num_iters, feature_list,
                      proxy_reward_space ,self.inference.true_reward_matrix, true_reward, self.args.beta, 'entropy',
                      planner=mdp.type)

        with tf.Session() as sess:
            feature_exp_true = self.inference.feature_exp_matrix   # For testing purposes (wrong dimension)
            assert feature_exp_true is not None

            desired_outputs = ['entropy', 'optimal_weights', 'features', 'weights_unsorted', 'state_probs',
                               'reward_per_state', 'feature_exps']
            model_outputs = model.compute(desired_outputs, sess, mdp, self.inference.prior, init, feature_exp_true)
            print desired_outputs
            print desired_outputs[0], model_outputs[0]
            print desired_outputs[1], model_outputs[1]
            print desired_outputs[2], model_outputs[2]
            print desired_outputs[3], model_outputs[3]
            print desired_outputs[4], model_outputs[4]
            print desired_outputs[5], model_outputs[5]
            print desired_outputs[6], model_outputs[6]

            # print model_outputs[1]
            # print model_outputs[2]
            # print model_outputs[3]
            # print model_outputs[4]
            # print model_outputs[5]
            # print model_outputs[6]

            # if 'answer' in desired_outputs:
            #     answer = model.sample_human_answer()
            #     return model_outputs, answer
        return model_outputs


    def find_random_query(self, query_size):
        query = [choice(self.reward_space_proxy) for _ in range(query_size)]
        exp_regret = self.get_exp_regret_from_query([])
        return query, exp_regret, None
    
    # @profile
    def get_exp_exp_post_regret(self, query, total_reward=False):
        """Returns the expected regret after getting query answered. This measure should be minimized over queries.
        The calculation is done by calculating the probability of each answer and then the regret conditioned on it."""

        if len(query) == 0:
            return self.get_exp_regret(no_query=True)

        posterior, post_averages, probs_proxy_choice = self.inference.calc_posterior(query)
        avg_reward_matrix = self.inference.get_avg_reward_for_post_averages(post_averages)
        if total_reward:    # Optimizes total reward instead of regret
            regrets = -avg_reward_matrix
        else:
            optimal_rewards = self.inference.true_reward_avg_reward_vec
            regrets = optimal_rewards.reshape(1,-1) - avg_reward_matrix

        exp_regrets = np.dot(regrets * posterior, np.ones(posterior.shape[1]))  # Make sure there's no broadcasting
        exp_exp_regret = np.dot(probs_proxy_choice, exp_regrets)

        # # Old approach
        # exp_exp_regret = 0
        # for proxy in query:
        #     self.inference.calc_and_save_posterior(proxy, query)
        #     p_proxy_chosen = self.inference.get_prob_proxy_choice(proxy, query)
        #     exp_regret = self.get_exp_regret()
        #     exp_exp_regret += p_proxy_chosen * exp_regret
        return exp_exp_regret

    def get_conditional_entropy(self, query):
        posteriors, conditional_entropy, probs_proxy_choice = self.inference.calc_posterior(query, get_entropy=True)
        return conditional_entropy

    # # @profile
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

    # # @profile
    def get_regret(self, proxy, true_reward):
        """Gets difference of reward under true_reward-function for optimizing for true_reward vs proxy."""
        optimal_reward = self.inference.get_avg_reward(true_reward, true_reward)  # Cache to save 9%
        proxy_reward = self.inference.get_avg_reward(proxy, true_reward)
        regret = optimal_reward - proxy_reward
        return regret

    # # @profile
    def get_exp_regret_from_query(self, query):
        """Calculates the actual regret from a query by looping through (and weighting) true rewards."""
        if len(query) == 0:
            prior_avg = np.array(sum([self.inference.get_prior(true_reward) * true_reward
                                      for true_reward in self.inference.reward_space_true]))
            regret_vec = np.array([self.get_regret(prior_avg, true_reward) for true_reward in self.inference.reward_space_true])
            exp_regret = np.dot(regret_vec, self.inference.prior)
            return exp_regret
        else:
            raise NotImplementedError

    # # @profile
    def get_regret_from_query_and_true_reward(self, query, true_reward):
        if len(query) == 0:
            prior_avg = self.inference.get_prior_avg()
            regret = self.get_regret(prior_avg, true_reward)
            return regret
        else:
            raise NotImplementedError



class Experiment(object):
    """"""
    def __init__(self, inference, reward_space_proxy, query_size, num_queries_max, args, choosers, SEED, exp_params, exp_name):
        self.inference = inference  # TODO: Possibly create inference here and maybe input params as a dict
        self.reward_space_proxy = reward_space_proxy
        self.query_size_discrete = query_size
        self.query_size_feature = args.query_size_feature
        self.num_queries_max = num_queries_max
        self.choosers = choosers
        self.seed = SEED
        self.query_chooser = Query_Chooser_Subclass(inference, reward_space_proxy, num_queries_max, args)
        self.results = {}
        # Add variance
        self.measures = ['post_exp_regret','test_regret','norm post_avg-true','post_regret','perf_measure']
        self.exp_params = exp_params
        self.exp_name = exp_name
        self.time = str(datetime.datetime.now())[:-6]

    # @profile
    def get_experiment_stats(self, num_iter, num_experiments):
        self.results = {}
        self.t_0 = time.clock()
        post_exp_regret_measurements = []; post_regret_measurements = []
        for exp_num in range(num_experiments):
            post_exp_regrets, post_regrets = self.run_experiment(num_iter, exp_num, num_experiments)
            post_exp_regret_measurements.append(post_exp_regrets)
            post_regret_measurements.append(post_regrets)
            self.write_experiment_results_to_csv(exp_num, num_iter)

        self.write_mean_and_median_results_to_csv(num_experiments, num_iter)

        avg_post_exp_regret_per_chooser = np.array(post_exp_regret_measurements).mean(axis=0)
        std_post_exp_regret_per_chooser = np.array(post_exp_regret_measurements).std(axis=0)
        avg_post_regret_per_chooser = np.array(post_regret_measurements).mean(axis=0)
        std_post_regret_per_chooser = np.array(post_regret_measurements).std(axis=0)

        return avg_post_exp_regret_per_chooser, avg_post_regret_per_chooser, \
               std_post_exp_regret_per_chooser, std_post_regret_per_chooser, \
               self.results


    # @profile
    def run_experiment(self, num_iter, exp_num, num_experiments):
        print "======================Experiment {n}/{N}=======================".format(n=exp_num + 1, N=num_experiments)
        # Initialize variables
        self.inference.reset(reset_mdp=True)
        seed(self.seed)
        self.seed += 1
        true_reward = choice(self.inference.reward_space_true)
        self.results['true_reward', exp_num] = true_reward
        post_exp_regret_per_chooser = []
        post_regret_per_chooser = []

        # Cache feature exp and lhoods
        # function = self.inference.calc_and_save_feature_expectations
        # input = self.reward_space_proxy
        print('caching feature exp for proxies...')
        # self.inference.calc_and_save_feature_expectations(self.reward_space_proxy)
        print('caching feature exp for true rewards...')
        # self.inference.calc_and_save_feature_expectations(self.inference.reward_space_true)
        print 'NOT CACHING FEATURES FOR TRUE REWARDS!'
        print('caching likelihoods...')
        self.inference.cache_lhoods()
        print('done caching')

        perf_measure = float('inf')
        post_exp_regret = float('inf')
        post_regret = float('inf')
        post_entropy = float('inf')

        # Run experiment for each query chooser
        for chooser in self.choosers:
            print "=========Experiment {n}/{N} for {chooser}=========".format(chooser=chooser,n=exp_num+1,N=num_experiments)
            self.inference.reset_prior()

            for i in range(num_iter):
                # print "Iteration: {i}/{m}. Total time: {t}".format(i=i+1,m=num_iter,t=time.clock()-self.t_0)
                # Do iteration for feature-based choosers:
                if chooser in ['feature_entropy']:
                    query, objective, true_posterior = self.query_chooser.find_query(self.query_size_feature, chooser, true_reward)
                    self.inference.update_prior(None, None, true_posterior)
                else:
                    query, perf_measure, _ \
                        = self.query_chooser.find_query(self.query_size_discrete, chooser, true_reward)

                    _, post_cond_entropy, _ = self.inference.calc_posterior(query, get_entropy=True)
                    proxy_choice = self.inference.get_proxy_from_query(query, true_reward)
                    self.inference.update_prior(query, proxy_choice)    # TODO: Still uses calc_and_save_posterior

                # Outcome measures
                post_exp_regret = self.query_chooser.get_exp_regret_from_query(query=[])
                post_regret = self.query_chooser.get_regret_from_query_and_true_reward([], true_reward) # TODO: Still uses old get_regret, get_avg_reward, get_feature_exp
                post_avg = self.inference.get_prior_avg()
                test_regret = float('inf')  # self.test_post_avg(post_avg, true_reward)

                # Save results
                # self.results[chooser, 'post_entropy', i, exp_num], \
                self.results[chooser, 'query', i, exp_num], self.results[chooser,'perf_measure', i, exp_num], \
                self.results[chooser, 'post_exp_regret', i, exp_num], self.results[chooser, 'post_regret', i, exp_num], \
                self.results[chooser, 'norm post_avg-true', i, exp_num],   \
                self.results[chooser, 'test_regret', i, exp_num] \
                    = query, perf_measure, post_exp_regret, post_regret, np.linalg.norm(post_avg-true_reward,1), test_regret


            print('post_exp_regret: {p}'.format(p=post_exp_regret))
            post_exp_regret_per_chooser.append(post_exp_regret)
            post_regret_per_chooser.append(post_regret)

        return post_exp_regret_per_chooser, post_regret_per_chooser

    # @profile
    def test_post_avg(self, post_avg, true_reward):
        # dist_scale = 0.5
        # gamma = 0.8
        # goals = [(1, 1), (2, 6), (3, 3), (3, 4), (4, 5), (6, 4), (6, 6)]
        goals = [(1, 5), (1, 2), (3, 6), (2, 3), (6, 1), (3, 5), (4, 2)]
        num_traject = 1
        # beta = 2.
        reps = 4
        post_reward_avg = 0
        post_regret_avg = 0


        # print true_reward
        # print post_avg
        # print true_reward - post_avg

        # TODO: Randomize goal positions for repetitions
        # TODO: Why not pass on full posterior?

        for _ in range(reps):
            # Set environment and agent
            grid = GridworldMdp.generate_random(8, 8, 0.1, 0.2, goals, living_reward=-0.01)
            mdp = GridworldMdpWithDistanceFeatures(grid, dist_scale, living_reward=-0.01, noise=0, rewards=post_avg)
            agent = ValueIterationLikeAgent(gamma, num_iters=50)
            super(ValueIterationLikeAgent, agent).set_mdp(mdp)

            # Set up inference
            env = GridworldEnvironment(mdp)
            inference = Inference(agent, env, beta, self.inference.reward_space_true, self.inference.reward_space_proxy,
                                  num_traject=num_traject, prior=None)

            post_reward = inference.get_avg_reward(post_avg, true_reward)
            optimal_reward = inference.get_avg_reward(true_reward, true_reward)
            regret = optimal_reward - post_reward
            post_reward_avg += 1/float(reps) * post_reward
            post_regret_avg += 1/float(reps) * regret

        return post_regret_avg


    def write_experiment_results_to_csv(self, exp_num, num_iter):
        """Writes a CSV for every chooser for every experiment. The CSV's columns are 'iteration' and all measures in
        self.measures."""
        folder_name = self.time + ' ' + self.exp_name + '--'  + ' ' + self.exp_params
        if not os.path.exists('data/'+folder_name):
            os.mkdir('data/'+folder_name)
        else:
            Warning('Existing experiment stats overwritten')
        for chooser in self.choosers:
            f = open('data/'+folder_name+'/'+chooser+str(exp_num)+'.csv','w')  # Open CSV in folder with name exp_params
            writer = csv.DictWriter(f, fieldnames=['iteration']+self.measures)
            writer.writeheader()
            rows = []
            for i in range(num_iter):
                csvdict = {}
                csvdict['iteration'] = i
                for measure in self.measures:
                    entry = self.results[chooser, measure, i, exp_num]
                    csvdict[measure] = entry
                rows.append(csvdict)
            writer.writerows(rows)


    def write_mean_and_median_results_to_csv(self, num_experiments, num_iter):
        """Writes a CSV for every chooser averaged (and median-ed) across experiments. Saves in the same folder as
        CSVs per experiment. Columns are 'iteration' and all measures in self.measures."""
        time = str(datetime.datetime.now())[:-13]
        folder_name = self.time + ' ' + self.exp_name + '--'  + ' ' + self.exp_params
        if not os.path.exists('data/'+folder_name):
            os.mkdir('data/'+folder_name)
        else:
            Warning('Existing experiment stats overwritten')

        f_mean_all = open('data/'+folder_name+'/'+'all choosers'+'-means-'+'.csv','w')
        writer_mean_all_choosers = csv.DictWriter(f_mean_all, fieldnames=['iteration']+self.measures)

        for chooser in self.choosers:
            f_mean = open('data/'+folder_name+'/'+chooser+'-means-'+'.csv','w')
            f_median = open('data/'+folder_name+'/'+chooser+'-medians-'+'.csv','w')
            writer_mean = csv.DictWriter(f_mean, fieldnames=['iteration']+self.measures)
            writer_median = csv.DictWriter(f_median, fieldnames=['iteration']+self.measures)
            writer_mean.writeheader()
            writer_median.writeheader()
            rows_mean = []
            rows_median = []
            for i in range(num_iter):
                csvdict_mean = {}
                csvdict_median = {}
                csvdict_mean['iteration'] = i
                csvdict_median['iteration'] = i
                for measure in self.measures:
                    entries = np.zeros(num_experiments)
                    for exp_num in range(num_experiments):
                        entry = self.results[chooser, measure, i, exp_num]
                        entries[exp_num] = entry
                    csvdict_mean[measure] = np.mean(entries)
                    csvdict_median[measure] = np.median(entries)
                rows_mean.append(csvdict_mean)
                rows_median.append(csvdict_mean)
            writer_mean.writerows(rows_mean)
            writer_median.writerows(rows_median)

            # Also append statistics for this chooser to CSV with all choosers
            writer_mean_all_choosers.writerow({measure: chooser for measure in self.measures})
            writer_mean_all_choosers.writeheader()
            writer_mean_all_choosers.writerows(rows_mean)

        'Also compare choosers: '
        'Also get std'
