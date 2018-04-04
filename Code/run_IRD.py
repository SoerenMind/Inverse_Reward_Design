import time

print('importing')

start = time.clock()
import datetime
print 'time: '+str(datetime.datetime.now())
import numpy as np
from inference_class import InferenceDiscrete
from gridworld import GridworldEnvironment, Direction, NStateMdpHardcodedFeatures, NStateMdpGaussianFeatures,\
    NStateMdpRandomGaussianFeatures, GridworldMdpWithDistanceFeatures, GridworldMdp
from agents import ImmediateRewardAgent, DirectionalAgent, OptimalAgent
from query_chooser_class import Query_Chooser_Subclass, Experiment
# from interface_discrete import Interface
from random import choice, seed
# from scipy.special import comb
import copy
from utils import Distribution
# from scipy.misc import logsumexp
import sys
import argparse
import tensorflow as tf


print 'Time to import: {deltat}'.format(deltat=time.clock() - start)




def pprint(y):
    print(y)
    return y



# ==================================================================================================== #
# ==================================================================================================== #
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('choosers',type=list,default='[greedy_entropy]')
    parser.add_argument('-c','--c', action='append', required=True) # c for choosers
    parser.add_argument('--query_size',type=int,default=3)
    parser.add_argument('--num_experiments',type=int,default=2) # 3-5
    parser.add_argument('--num_iter',type=int,default=20)    # number of queries asked
    parser.add_argument('--gamma',type=float,default=1.) # otherwise 0.98. Values <1 might make test regret inaccurate.
    parser.add_argument('--size_true_space',type=int,default=1000000)
    parser.add_argument('--size_proxy_space',type=int,default=100)  # Sample subspace for exhaustive
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--beta',type=float,default=0.2)
    parser.add_argument('--beta_planner',type=float,default=0.5) # 1 for small version of results
    parser.add_argument('--num_states',type=int,default=100)  # 10 options if env changes over time, 100 otherwise
    parser.add_argument('--dist_scale',type=float,default=0.5) # test briefly to get ent down
    parser.add_argument('--num_traject',type=int,default=1)
    parser.add_argument('--num_queries_max',type=int,default=2000)
    parser.add_argument('--height',type=int,default=12)
    parser.add_argument('--width',type=int,default=12)
    parser.add_argument('--lr',type=float,default=20)  # Learning rate
    parser.add_argument('--num_iters_optim',type=int,default=10)
    parser.add_argument('--value_iters',type=int,default=15)    # max_reward / (1-gamma) or height+width
    parser.add_argument('--mdp_type',type=str,default='gridworld')
    parser.add_argument('--feature_dim',type=int,default=20)    # 10 if positions fixed, 100 otherwise
    parser.add_argument('--discretization_size',type=int,default=5)
    parser.add_argument('--discretization_size_human',type=int,default=5)
    parser.add_argument('--num_test_envs',type=int,default=100)    # 10 if positions fixed, 100 otherwise
    parser.add_argument('--well_spec',type=int,default=1)    # default is well-specified
    parser.add_argument('--subsampling',type=int,default=1)
    parser.add_argument('--num_subsamples',type=int,default=10000)
    parser.add_argument('--weighting',type=int,default=1)
    parser.add_argument('--linear_features',type=int,default=1)
    parser.add_argument('--objective',type=str,default='entropy')
    parser.add_argument('--rational_test_planner',type=int,default=1)
    parser.add_argument('-weights_dist_init',type=str,default='normal2')
    parser.add_argument('-weights_dist_search',type=str,default='normal2')
    parser.add_argument('-square_probs',type=int,default=0)
    parser.add_argument('--only_optim_biggest',type=int,default=1)




    args = parser.parse_args()
    print args
    assert args.discretization_size % 2 == 1

    # Experiment description
    adapted_description = False
    # print "Adapted description: ", adapted_description

    # Set parameters
    feature_dim = args.feature_dim
    dummy_rewards = np.zeros(feature_dim)
    # Set parameters
    choosers = args.c
    SEED = args.seed
    seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    beta = args.beta
    num_states = args.num_states
    size_reward_space_true = args.size_true_space
    size_reward_space_proxy = args.size_proxy_space
    num_queries_max = args.num_queries_max
    num_traject = args.num_traject
    num_experiments = args.num_experiments
    num_iter_per_experiment = args.num_iter #; print('num iter = {i}'.format(i=num_iter_per_experiment))
    # Params for Gridworld
    gamma = args.gamma
    query_size = args.query_size
    dist_scale = args.dist_scale
    height = args.height
    width = args.width
    num_iters_optim = args.num_iters_optim
    # choosers = ['greedy', 'greedy_exp_reward']
    # choosers = ['no_query','greedy_entropy', 'greedy', 'greedy_exp_reward', 'random']
    # choosers = ['greedy_entropy', 'random', 'no_query']

    # These will be in the folder name of the log
    exp_params = {
        # 'rational_test_planner': args.rational_test_planner,
        'qsize': query_size,
        'mdp': args.mdp_type,
        'dim': feature_dim,
        'dsize': args.discretization_size,
        'size_true': size_reward_space_true,
        'size_proxy': size_reward_space_proxy,
        'seed': SEED,
        'beta': beta,
        # 'num_states': num_states,
        # 'dist_scale': dist_scale,
        'n_q_max': num_queries_max,
        # 'num_iters_optim': num_iters_optim,
        'well_spec': args.well_spec,
        # 'subsamp': args.subsampling,
        # 'num_subsamp': args.num_subsamples,
        # 'weighting': args.weighting,
        # 'viters': args.value_iters,
        # 'linfeat': args.linear_features,
        # 'objective': args.objective,
        'w_dist_i': args.weights_dist_init,
        'w_dist_s': args.weights_dist_search,
        'optim_big': args.only_optim_biggest,
        'rational_test': args.rational_test_planner
    }

    'Sample true rewards and reward spaces'
    reward_space_true = np.array(np.random.randint(-9, 10, size=[size_reward_space_true, args.feature_dim]), dtype=np.int16)
    # reward_space_proxy = np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim])
    if not args.well_spec:
        true_rewards = [np.random.randint(-9, 10, size=[args.feature_dim]) for _ in range(num_experiments)]
    else:
        true_rewards = [choice(reward_space_true) for _ in range(num_experiments)]
    prior_avg = -0.5 * np.ones(args.feature_dim) + 1e-4 * np.random.exponential(1,args.feature_dim) # post_avg for uniform prior + noise

    # Set up env and agent for NStateMdp
    if args.mdp_type == 'bandits':

        'Create train and test MDPs'
        test_mdps = []
        for i in range(args.num_test_envs):
            # mdp = NStateMdpRandomGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim),
            #                                       start_state=0, preterminal_states=[],
            #                                       feature_dim=args.feature_dim, num_states_reachable=num_states,
            #                                       SEED=SEED)
            mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                            feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED+i*50+100)
            test_mdps.append(mdp)

        train_mdps = []
        for i in range(num_experiments):
            # mdp = NStateMdpRandomGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim),
            #                                       start_state=0, preterminal_states=[],
            #                                       feature_dim=args.feature_dim, num_states_reachable=num_states,
            #                                       SEED=SEED)
            mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                            feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED+i*50)
            train_mdps.append(mdp)

        'Create train and test inferences'
        test_inferences = []
        for i in range(args.num_test_envs):
            mdp = test_mdps[i]
            env = GridworldEnvironment(mdp)
            agent = ImmediateRewardAgent()  # Not Boltzmann unlike train agent
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy=[], num_traject=1, prior=None)

            test_inferences.append(inference)

        train_inferences = []
        for i in range(num_experiments):
            mdp = train_mdps[i]
            env = GridworldEnvironment(mdp)
            agent = ImmediateRewardAgent()  # Not Boltzmann unlike train agent
            reward_space_proxy = np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim])
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy, num_traject=1, prior=None)

            train_inferences.append(inference)


    # Set up env and agent for gridworld
    elif args.mdp_type == 'gridworld':
        'Create train and test MDPs'
        test_inferences = []
        for i in range(args.num_test_envs):
            test_grid = GridworldMdp.generate_random(height,width,0.35,feature_dim,None,living_reward=-0.01, print_grid=False)
            mdp = GridworldMdpWithDistanceFeatures(test_grid, args.linear_features, dist_scale, living_reward=-0.01, noise=0, rewards=dummy_rewards)
            env = GridworldEnvironment(mdp)
            agent = OptimalAgent(gamma, num_iters=args.value_iters)
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy=[], num_traject=num_traject, prior=None)

            test_inferences.append(inference)


        train_inferences = []
        for j in range(num_experiments):
            grid = GridworldMdp.generate_random(height,width,0.35,feature_dim,None,living_reward=-0.01, print_grid=False)
            mdp = GridworldMdpWithDistanceFeatures(grid, args.linear_features, dist_scale, living_reward=-0.01, noise=0, rewards=dummy_rewards)
            env = GridworldEnvironment(mdp)
            agent = OptimalAgent(gamma, num_iters=args.value_iters)
            reward_space_proxy = np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim])
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy,
                num_traject=num_traject, prior=None)

            train_inferences.append(inference)


    else:
        raise ValueError('Unknown MDP type: ' + str(args.mdp_type))



    # # Set up inference
    # env = GridworldEnvironment(mdp)
    # inference = InferenceDiscrete(
    #     agent, mdp, env, beta, reward_space_true, reward_space_proxy,
    #     num_traject=num_traject, prior=None)

    'Print derived parameters'
    # print('Size of reward_space_true:{size}'.format(size=size_reward_space_true))
    # print('Size of reward_space_proxy:{size}'.format(size=len(reward_space_proxy)))
    # print('Query size:{size}'.format(size=query_size))
    # print('Choosers: {c}').format(c=choosers)
    # if greedy == False:
    #     num_queries = min([comb(len(reward_space_proxy), query_size),    num_queries_max])
    #     # num_queries = comb(len(reward_space_proxy), query_size)
    #     num_post_avg_plans = num_queries * query_size
    # else:
    #     num_queries = (query_size-1) * len(reward_space_proxy)
    #     avg_query_size = (query_size+2)/2.
    #     num_post_avg_plans = num_queries * avg_query_size
    # print('Number of queries: min({size},{max})'.format(size=num_queries,max=num_queries_max))
    # num_planning_problems = len(reward_space_proxy) + num_post_avg_plans + size_reward_space_true
    # print('Number of rewards to plan with:{size}'.format(size=num_planning_problems))
    # print('Greedy: {g}').format(g=greedy)
    print('======================================================================================================')



    'Set up test environment (not used)'
    # print 'starting posterior calculation'
    # inference.get_full_posterior(reward_space_proxy, proxy_given)
    # prior = dict([(tuple(true_reward), inference.get_posterior(true_reward, reward_space_proxy, proxy_given))
    #               for true_reward in reward_space_true])
    # print('new prior: {prior}'.format(prior=prior))
    # mdp_test = NStateMdpGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],   # proxy_given should have no effect
    #                                      feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED)
    # mdp_test.add_feature_map(mdp.features)
    # env_test = GridworldEnvironment(mdp_test)
    # inference_test = InferenceDiscrete(agent, mdp_test, env_test, beta=1., reward_space_true=reward_space_true, num_traject=1, prior=prior)




    'Experiment'
    def run_experiment(query_size, train_inferences, test_inferences, true_rewards, prior_avg):
        experiment = Experiment(true_rewards, reward_space_proxy, query_size, num_queries_max,
                                args, choosers, SEED, exp_params, train_inferences, test_inferences, prior_avg)
        results = experiment.get_experiment_stats(num_iter_per_experiment, num_experiments)


        print('__________________________Finished experiment__________________________')

    run_experiment(query_size, train_inferences, test_inferences, true_rewards, prior_avg)


    'Create interface'
    # omega = [choice(reward_space_true) for _ in range(4)] # replace with chosen omega
    # interface = Interface(omega, agent, env, num_states=num_states)
    # interface.plot()
