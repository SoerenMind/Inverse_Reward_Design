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
    parser.add_argument('--num_iter',type=int,default=15)    # number of queries asked
    parser.add_argument('--gamma',type=float,default=1.) # otherwise 0.98. Values <1 might make test regret inaccurate.
    parser.add_argument('--size_true_space',type=int,default=1000)
    parser.add_argument('--size_proxy_space',type=int,default=100)  # Sample subspace for exhaustive
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--beta',type=float,default=0.1)
    parser.add_argument('--beta_planner',type=float,default=10.) # 1 for small version of results
    parser.add_argument('--num_states',type=int,default=100)  # 10 options if env changes over time, 100 otherwise
    parser.add_argument('--dist_scale',type=float,default=0.5) # test briefly to get ent down
    parser.add_argument('--num_traject',type=int,default=1)
    parser.add_argument('--num_queries_max',type=int,default=500)   # x10 the number tried for greedy
    parser.add_argument('--height',type=int,default=12)
    parser.add_argument('--width',type=int,default=12)
    parser.add_argument('--lr',type=float,default=0.1)  # Learning rate
    parser.add_argument('--num_iters_optim',type=int,default=10)
    parser.add_argument('--value_iters',type=int,default=25)    # max_reward / (1-gamma) or height+width
    parser.add_argument('--mdp_type',type=str,default='gridworld')
    parser.add_argument('--feature_dim',type=int,default=10)    # 10 if positions fixed, 100 otherwise
    parser.add_argument('--num_test_envs',type=int,default=20)    # 10 if positions fixed, 100 otherwise
    parser.add_argument('--well_spec',type=int,default=1)    # default is well-specified
    parser.add_argument('--subsampling',type=int,default=1)
    parser.add_argument('--num_subsamples',type=int,default=10000)


    args = parser.parse_args()
    print args


    # Experiment description
    adapted_description = False
    print "Adapted description: ", adapted_description
    exp_description = pprint("Comparing to entropy with many states and few true rewards. {nexp} experiments.")

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
        'qsize': query_size,
        'mdp': args.mdp_type,
        'dim': feature_dim,
        'size_true': size_reward_space_true,
        'size_proxy': size_reward_space_proxy,
        # 'true_rw_random' : args.true_rw_random,
        'seed': SEED,
        'beta': beta,
        'num_states': num_states,
        'dist_scale': dist_scale,
        # 'num_q_max': num_queries_max,
        # 'num_iters_optim': num_iters_optim,
        'well_spec': args.well_spec,
        'subsampling': args.subsampling,
        'num_subsamples': args.num_subsamples,
        'viters: ': args.value_iters
    }

    # # Set up env and agent for NStateMdp
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

        'Sample true rewards and reward spaces'

        reward_space_true = np.array(np.random.randint(-9, 9, size=[size_reward_space_true, args.feature_dim]), dtype=np.int16)
        reward_space_proxy = np.random.randint(-9, 9, size=[size_reward_space_proxy, args.feature_dim])

        if not args.well_spec:
            true_rewards = [np.random.randint(-9, 9, size=[args.feature_dim]) for _ in range(num_experiments)]
        else:
            true_rewards = [choice(reward_space_true) for _ in range(num_experiments)]
        prior_avg = -0.5 * np.ones(args.feature_dim) + 1e-4 * np.random.exponential(1,args.feature_dim) # post_avg for uniform prior + noise

        'Create train and test inferences'
        test_inferences = []
        for i in range(args.num_test_envs):
            mdp = test_mdps[i]
            env = GridworldEnvironment(mdp)
            agent = ImmediateRewardAgent()  # Not Boltzmann unlike train agent
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy, num_traject=1, prior=None)

            test_inferences.append(inference)

        train_inferences = []
        for i in range(num_experiments):
            mdp = train_mdps[i]
            env = GridworldEnvironment(mdp)
            agent = ImmediateRewardAgent()  # Not Boltzmann unlike train agent
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy, num_traject=1, prior=None)

            train_inferences.append(inference)


    # Set up env and agent for gridworld
    elif args.mdp_type == 'gridworld':

        'Sample true rewards and reward spaces'
        # reward_space_true = [np.random.multinomial(18, np.ones(args.feature_dim)/18) for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.multinomial(18, np.ones(args.feature_dim)) for _ in xrange(size_reward_space_proxy)]

        reward_space_true = np.array(np.random.randint(-9, 9, size=[size_reward_space_true, args.feature_dim]), dtype=np.int16)
        reward_space_proxy = np.random.randint(-9, 9, size=[size_reward_space_proxy, args.feature_dim])

        if not args.well_spec:
            true_rewards = [np.random.randint(-9, 9, size=[args.feature_dim]) for _ in range(num_experiments)]
        else:
            true_rewards = [choice(reward_space_true) for _ in range(num_experiments)]
        prior_avg = -0.5 * np.ones(args.feature_dim) + 1e-4 * np.random.exponential(1,args.feature_dim) # post_avg for uniform prior + noise

        # reward_space_true = [np.random.randint(-9, 9, size=[args.feature_dim])   for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.randint(-9, 9, size=[args.feature_dim]) for _ in xrange(size_reward_space_proxy)]

        # reward_space_true = [np.random.dirichlet(np.ones(args.feature_dim)) * args.feature_dim - 1 for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.dirichlet(np.ones(args.feature_dim)) * args.feature_dim - 1 for _ in xrange(size_reward_space_proxy)]


        'Create train and test MDPs'
        test_inferences = []
        for i in range(args.num_test_envs):
            test_grid = GridworldMdp.generate_random(height,width,0.35,feature_dim,None,living_reward=-0.01, print_grid=False)
            mdp = GridworldMdpWithDistanceFeatures(test_grid, dist_scale, living_reward=-0.01, noise=0, rewards=dummy_rewards)
            env = GridworldEnvironment(mdp)
            agent = OptimalAgent(gamma, num_iters=args.value_iters)
            inference = InferenceDiscrete(
                agent, mdp, env, beta, reward_space_true, reward_space_proxy, num_traject=num_traject, prior=None)

            test_inferences.append(inference)


        train_inferences = []
        for j in range(num_experiments):
            grid = GridworldMdp.generate_random(height,width,0.35,feature_dim,None,living_reward=-0.01, print_grid=False)
            mdp = GridworldMdpWithDistanceFeatures(grid, dist_scale, living_reward=-0.01, noise=0, rewards=dummy_rewards)
            env = GridworldEnvironment(mdp)
            agent = OptimalAgent(gamma, num_iters=args.value_iters)
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
    def experiment(query_size, train_inferences, test_inferences, true_rewards, prior_avg):
        # test_planning_speed(inference, reward_space_proxy); print('tested planning speed')
        # mean, std, failures, gain, mean_std_actual, regret_compare, regret_exp_vs_actual \
        #     = experiment(inference, reward_space_proxy, query_size, iterations_optimized=num_experiments, greedy=greedy,
        #                  num_queries_max=num_queries_max)
        # print mean, std, failures, gain, mean_std_actual
        # print 'Expected regret improvement over random query (mean): {r}'.format(r=mean)
        # print 'Expected regret improvement over no query (mean): {r}'.format(r=gain)
        # print 'Mean actual -reduction and std(mean) over random query: {r}'.format(r=mean_std_actual)
        # print 'Actual regret diff optimized vs random:{r}'.format(r=regret_compare)
        # print 'Expected vs actual regret: {vs}'.format(vs=regret_exp_vs_actual)
        experiment = Experiment(true_rewards, reward_space_proxy, query_size, num_queries_max,
                                args, choosers, SEED, exp_params, train_inferences, test_inferences, prior_avg)
        results = experiment.get_experiment_stats(num_iter_per_experiment, num_experiments)


        # 'Print results'
        # print "Choosers:                        {c}".format(c=choosers)
        # print "Avg post exp regret per chooser: {x}".format(x=avg_post_exp_regrets)
        # print "Avg post regret per chooser: {x}".format(x=avg_post_regrets)
        # print "Std post exp regret per chooser: {x}".format(x=std_post_exp_regrets)
        # print "Std post regret per chooser: {x}".format(x=std_post_regrets)
        # # print [-results['greedy_exp_reward','perf_measure',4, n] for n in range(num_experiments)]
        # # print [results['greedy_exp_reward','post_exp_regret',4, n] for n in range(num_experiments)]
        # print [results['greedy_entropy','post_exp_regret',num_iter_per_experiment-1, n] for n in range(num_experiments)]
        # print [results['greedy_entropy','perf_measure',num_iter_per_experiment-1, n] for n in range(num_experiments)]
        # # print [results['greedy','perf_measure',4, n] for n in range(num_experiments)]
        #
        # print "Entropy per iteration for greedy_entropy:"
        # print [np.array([results['greedy_entropy','perf_measure',i, n] for n in range(num_experiments)]).mean() for i in range(num_iter_per_experiment)]
        # # print "Entropy per iteration for greedy:"
        # # print [[results['greedy','post_entropy',i, n] for n in range(num_experiments)] for i in range(num_iter_per_experiment)]
        #
        # # print("Test environment regret for greedy:")
        # # print [results['greedy','test_regret',num_iter_per_experiment-1,n] for n in range(num_experiments)]
        # print("Test environment regret for greedy_entropy:")
        # print [results['greedy_entropy','test_regret',num_iter_per_experiment-1,n] for n in range(num_experiments)]
        #
        # # print "Exp regret per iteration for greedy:"
        # # print [[results['greedy','post_exp_regret',i, n] for n in range(num_experiments)] for i in range(num_iter_per_experiment)]
        # print "Exp regret per iteration for greedy_entropy:"
        # print [np.array([results['greedy_entropy','post_exp_regret',i, n] for n in range(num_experiments)]).mean() for i in range(num_iter_per_experiment)]
        #
        #
        # # print 'mdp_features:'
        # # print np.array([np.concatenate([[state], mdp.features[state]]) for state in range(num_states)])
        # # print np.array([np.concatenate([[state], mdp.get_features(state)]) for state in range(num_states)])
        # # # # print('mdp features: {features}'.format(features=mdp_test.features))
        # # print(best_query)
        # # # print('Best post_avg:{post_avg}').format(post_avg=best_post_avg)
        # # # print('Best posterior:{posterior}').format(posterior=best_posterior)
        # print 'Total time:{deltat}'.format(deltat=time.clock() - start)
        # print 'Finished experiment: ', exp_description.format(nexp=num_experiments) + 'Updated description: '+str(adapted_description)

    # for q_size in range(2,50):
    #     if q_size % 4 == 0:
    #         experiment(q_size)
    experiment(query_size, train_inferences, test_inferences, true_rewards, prior_avg)


    'Create interface'
    # omega = [choice(reward_space_true) for _ in range(4)] # replace with chosen omega
    # interface = Interface(omega, agent, env, num_states=num_states)
    # interface.plot()
