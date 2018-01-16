import time

print('importing')

start = time.clock()
import numpy as np
import environment
import agent_class
from inference_class import Inference, test_inference
import itertools
from gridworld import NStateMdp, GridworldEnvironment, Direction, NStateMdpHardcodedFeatures, NStateMdpGaussianFeatures, NStateMdpRandomGaussianFeatures
from agents import ImmediateRewardAgent, DirectionalAgent
from query_chooser_class import Regret_Minimizing_Query_Chooser
# from interface_discrete import Interface
from random import choice
from scipy.special import comb
import copy


print('importing done')
print 'Time to import: {deltat}'.format(deltat=time.clock() - start)

"""Time sinks:
-import agent runner in inference
-from utils import Distribution in agents.py (ca 2s)
-from agent_runner import run_agent in interface_discrete.py (0.3s)
"""

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


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

    query_chooser = Regret_Minimizing_Query_Chooser(inference, reward_space_proxy, cost_of_asking=0.)
    set_of_queries = list(query_chooser.generate_set_of_queries(query_size))
    print(len(set_of_queries))
    best_query, best_regret, _ = query_chooser.find_regret_minimizing_query(set_of_queries)

def experiment(inference_sim, reward_space_proxy, iterations_random=10, iterations_optimized=20):
    exp_regret_diff = []
    exp_regret_gain = []
    regret_compare = []
    inference_sim = copy.deepcopy(inference_sim)
    inference_eval = copy.deepcopy(inference_sim)
    for i in range(iterations_optimized):
        print('Experiment number:{i}/{iter}'.format(i=i,iter=iterations_optimized))
        inference_sim.agent.mdp.populate_features()
        inference_sim.feature_expectations_dict = {}    # Replace these lines with inference.reset()
        query_chooser = Regret_Minimizing_Query_Chooser(inference_sim, reward_space_proxy, cost_of_asking=0.)
        set_of_queries = list(query_chooser.generate_set_of_queries(query_size))
        # for _ in range(0):
        #     set_of_queries += set_of_queries
        # print('duplicates query set times 2^3')
        random_query = choice(set_of_queries)
        best_query, best_regret, _     = query_chooser.find_regret_minimizing_query(set_of_queries)
        _, random_regret, _ = query_chooser.find_regret_minimizing_query([random_query]) # Sometimes finds lower regret for empty query
        _, prior_regret, _             = query_chooser.find_regret_minimizing_query([])
        exp_regret_diff.append(random_regret - best_regret) # This should match the actual regret diff on average
        exp_regret_gain.append(prior_regret - best_regret)

        # Do inference on chosen query, compare regret to that of random query and prior regret
        true_reward = choice(reward_space_true) # Replace with sample from
        regret_optimized = get_regret_from_query(inference_eval, best_query, true_reward)
        regret_random_query = get_regret_from_query(inference_eval, random_query, true_reward)
        regret_compare.append((regret_optimized, regret_random_query))

    regret_diff_actual = np.array([x-y for x,y in regret_compare])
    mean_std_regret_diff_actual = (regret_diff_actual.mean(), regret_diff_actual.std())
    return np.array(exp_regret_diff).mean(), np.array(exp_regret_diff).std(), sum(sum([np.array(exp_regret_diff) < 0])), \
           np.array(exp_regret_gain).mean(), mean_std_regret_diff_actual, regret_compare


def get_regret_from_query(inference_eval, best_query, true_reward):
    lhoods = []
    for i, proxy in enumerate(best_query):
        lhood = inference_eval.get_likelihood(true_reward, proxy, best_query)
        lhoods.append(lhood)
    if len(lhoods) == 0:
        pass
    try: chosen_proxy_number = np.array(lhoods).argmax()  # Replace argmax with sampling
    except:
        return 100
    chosen_proxy = best_query[chosen_proxy_number]
    inference_eval.calc_and_save_posterior(chosen_proxy, best_query)
    post_avg = inference_eval.get_posterior_avg()
    # TODO: Make a query_chooser / inference.function from query to regret or so
    optimal_reward = inference_eval.get_avg_reward(true_reward, true_reward)
    post_reward = inference_eval.get_avg_reward(post_avg, true_reward)
    regret = optimal_reward - post_reward
    return regret


def test_planning_speed(inference, reward_space_proxy):
    print('testing planning speed')
    for i, proxy in enumerate(reward_space_proxy):
        inference.get_feature_expectations(proxy)


if __name__=='__main__':
    # Define environment and agent
    SEED = 1
    beta = 4.
    num_states = 20; print('num states = 100')
    feature_dim = 20; print('feature dim = 5')
    # print('planning trivialized')
    query_size = 2
    size_reward_space_true = 50
    size_reward_space_proxy = 10
    proxy_given = np.zeros(feature_dim)
    # mdp = NStateMdpRandomGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],
    #                                 feature_dim=feature_dim, num_states_reachable=num_states, SEED=SEED)
    mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],
                                    feature_dim=feature_dim, num_states_reachable=num_states, SEED=SEED)
    env = GridworldEnvironment(mdp)
    agent = ImmediateRewardAgent()
    agent.set_mdp(mdp)
    # print(run_agent(agent, env, episode_length=6))


    # Set up inference
    # true_reward_given = np.array([0,0,1,0])
    # reward_space_true = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
    from itertools import product
    reward_space_true = list(product([0,1], repeat=feature_dim))
    # reward_space_true.remove((0,0,0,0))
    reward_space_true = [np.array(reward) for reward in reward_space_true]
    reward_space_true = [choice(reward_space_true) for _ in range(size_reward_space_true)]
    # reward_space_true = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0])]
    reward_space_proxy = [choice(reward_space_true) for _ in range(size_reward_space_proxy)]
    # reward_space_proxy = reward_space_true
    # len_reward_space = len(reward_space_true)
    # reward_space = [np.array([1,0]),np.array([0,1]), np.array([1,1])]
    inference = Inference(agent, env, beta=beta, reward_space_true=reward_space_true,
                          num_traject=1, prior=None)

    'Print derived parameters'
    print('Size of reward_space_true:{size}'.format(size=size_reward_space_true))
    print('Size of reward_space_proxy:{size}'.format(size=len(reward_space_proxy)))
    print('Query size:{size}'.format(size=query_size))
    num_queries = comb(len(reward_space_proxy), query_size)
    print('Number of queries:{size}'.format(size=num_queries))
    num_planning_problems = len(reward_space_proxy) + num_queries * query_size
    print('Number of proxies to plan with:{size}'.format(size=num_planning_problems))
    print('======================================================================================================')



    'Set up test environment (not used)'
    # print 'starting posterior calculation'
    # inference.calc_and_save_posterior(proxy_given, reward_space_proxy)
    # prior = dict([(tuple(true_reward), inference.get_posterior(true_reward))
    #               for true_reward in reward_space_true])
    # print('new prior: {prior}'.format(prior=prior))
    # mdp_test = NStateMdpGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],   # proxy_given should have no effect
    #                                      feature_dim=feature_dim, num_states_reachable=num_states, SEED=SEED)
    # mdp_test.add_feature_map(mdp.features)
    # env_test = GridworldEnvironment(mdp_test)
    # agent.set_mdp(mdp_test)
    # inference_test = Inference(agent, env_test, beta=1., reward_space_true=reward_space_true, num_traject=1, prior=prior)



    'Calculate proposal set'
    # # set_of_proposal_sets = powerset(reward_space_proxy)
    # # uniform_prior = {tuple(reward): np.true_divide(1,len_reward_space) for reward in reward_space_true}
    # # best_r_set, best_r_set_regret, best_regret_plus_cost, best_posterior, best_post_avg = choose_regret_minimizing_proposal(set_of_proposal_sets,
    # #                                                                 reward_space_true, prior=uniform_prior, inference=inference)
    # query_chooser = Regret_Minimizing_Query_Chooser(inference, reward_space_proxy, cost_of_asking=0.)
    # set_of_queries = list(query_chooser.generate_set_of_queries(query_size))
    # print(len(set_of_queries))
    # best_query, best_regret, _ = query_chooser.find_regret_minimizing_query(set_of_queries)


    'Experiment'
    # test_planning_speed(inference, reward_space_proxy); print('tested planning speed')
    mean, std, failures, gain, mean_std_actual, regret_compare = experiment(inference, reward_space_proxy)
    print mean, std, failures, gain, mean_std_actual
    print 'Actual regret diff optimized vs random:{r}'.format(r=regret_compare)

    'Print results'
    # print 'mdp_features:'
    # print np.array([np.concatenate([[state], mdp.features[state]]) for state in range(num_states)])
    # print np.array([np.concatenate([[state], mdp.get_features(state)]) for state in range(num_states)])
    # # # print('mdp features: {features}'.format(features=mdp_test.features))
    # # print(avg_weighted_reward)
    # # print(exp_reward_after_asking)
    # print(best_regret)
    # print(best_query)
    # # print('Best post_avg:{post_avg}').format(post_avg=best_post_avg)
    # # print('Best posterior:{posterior}').format(posterior=best_posterior)
    print 'Total time:{deltat}'.format(deltat=time.clock() - start)


    'Create interface'
    # omega = best_r_set # replace with chosen omega
    # interface = Interface(omega, agent, env_test, num_states=num_states)
    # interface.plot()

    """Todo:

    # TODO: Third task: Make sure to only do planning for each proxy once. Maybe make this function get the feature expectations instead and cache it.
        - Increase state space and feature_dim
        - Compare minimizer against random query
            -Randomize: Random features MDP; re-draw Mdp between experiments

            -Test random MDP (why so slow?); test posterior=1.
            -Why qubsequent experiments faster?
                -Try with static features: 5x faster
                -Why done planning for >32 proxies?: bc post_avg
                -Try duplicating the query space
                -See effect of trivializing planning
                -Use debugger
            -Run it!

                -Draw a true reward each time!
        - Try a problem where I know the answer?
    # TODO: Calculate joint of w, tilde(w)
        -P(w') = E_w P(w' | w)P(w)
        -Calculation of P(w' | w):
            -Cache feature expectations with every w' in Q.
            -For every w:
                -Get likelihood P(w' | w). Done.
            -Or just P(w') = sum_w prior(w) * get_likelihood(proxy, true)
    # TODO: Fixed size queries (ca 4)
        -Test
        -Increase feature_dim
        -Test: Compare binary comparisons
        -Scale to sets of 4
    # TODO: Compare greedy lookahead against random query. Test by sampling a real reward from the prior and seeing how we do on average.
    # TODO: line-by-line profiler: https://github.com/rkern/line_profiler or https://plugins.jetbrains.com/plugin/8525-python-profiler-experimental
        # Github has a python file to run my files
        # python

    -Implement regret minimizer
        -Test inference first
        -Why stochasticity in choice and regret?
        -Test: Does the omega choice make sense?
    -Non-static features
    -Posterior not adding up - fix?
       -Test if new inference adds up
    -Interface:
        (-Visualize feature dist)
    -Try not sampling actions - Change back!
    -Other approaches/features for tomorrow
            + outline code (and section?)
    -Implement Race car / Dorsa domain with between-track generalization
    """
