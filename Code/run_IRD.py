import time

print('importing')

start = time.clock()
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
    parser.add_argument('q_size',type=int)
    # parser.add_argument('choosers',type=list,default='[greedy_entropy]')
    parser.add_argument('-c','--c', action='append', required=True) # c for choosers
    parser.add_argument('--query_size_feature',type=int,default=3)
    parser.add_argument('--num_experiments',type=int,default=1) # 3-5
    parser.add_argument('--num_iter',type=int,default=10)    # number of queries asked
    # TODO: Values are computed as if trajectories are infinite. Problem?
    parser.add_argument('--gamma',type=float,default=1.) # otherwise 0.98
    parser.add_argument('--size_true_space',type=int,default=1000)
    parser.add_argument('--size_proxy_space',type=int,default=100)  # Sample subspace for exhaustive
    parser.add_argument('--num_trajectories',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--beta',type=float,default=.1)
    parser.add_argument('--beta_planner',type=float,default=50.) # 1 for small version of results
    parser.add_argument('--num_states',type=int,default=6)  # 10 options if env changes over time, 100 otherwise
    parser.add_argument('--dist_scale',type=float,default=0.5) # test briefly to get ent down
    parser.add_argument('--num_traject',type=int,default=1)
    parser.add_argument('--num_queries_max',type=int,default=500)   # x10 the number tried for greedy
    parser.add_argument('--is_greedy',type=bool,default=True)
    parser.add_argument('--height',type=int,default=12)
    parser.add_argument('--width',type=int,default=12)
    parser.add_argument('--lr',type=float,default=0.1)  # Learning rate
    parser.add_argument('--num_iters_optim',type=int,default=10)
    parser.add_argument('--value_iters',type=int,default=40)    # max_reward / (1-gamma) or height+width
    # parser.add_argument('--value_iters_discrete',type=int,default=50)
    parser.add_argument('--mdp_type',type=str,default='gridworld')
    parser.add_argument('--feature_dim',type=int,default=25)    # 10 if positions fixed, 100 otherwise


    args = parser.parse_args()


    # Experiment description
    adapted_description = False
    print "Adapted description: ", adapted_description
    exp_description = pprint("Comparing to entropy with many states and few true rewards. {nexp} experiments.")

    # Set parameters
    dummy_rewards = np.zeros(3)
    feature_dim = args.feature_dim
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
    query_size = args.q_size
    dist_scale = args.dist_scale
    height = args.height
    width = args.width
    num_iters_optim = args.num_iters_optim
    # choosers = ['greedy', 'greedy_exp_reward']
    # choosers = ['no_query','greedy_entropy', 'greedy', 'greedy_exp_reward', 'random']
    # choosers = ['greedy_entropy', 'random', 'no_query']

    exp_params = {
        'qsize': query_size,
        'num_experiments': num_experiments,
        'mdp': args.mdp_type,
        'dim': feature_dim,
        'num_iter': args.num_iter,
        'gamma': gamma,
        'size_true': size_reward_space_true,
        'size_proxy': size_reward_space_proxy,
        'seed': SEED,
        'beta': beta,
        'num_states': num_states,
        'dist_scale': dist_scale,
        'num_traject': num_traject,
        'num_queries_max': num_queries_max,
        'height': height,
        'width': width,
        'num_iters_optim': num_iters_optim,
        'value_iters': args.value_iters
    }

    # # Set up env and agent for NStateMdp
    if args.mdp_type == 'bandits':
        # mdp = NStateMdpRandomGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
        #                                 feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED)
        mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                        feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED)
        agent = ImmediateRewardAgent()

        # Reward spaces
        reward_space_true = [np.random.randint(-9, 9, size=[args.feature_dim]) for _ in xrange(size_reward_space_true)]
        reward_space_proxy = [np.random.randint(-9, 9, size=[args.feature_dim]) for _ in xrange(size_reward_space_proxy)]


        # from itertools import product
        # reward_space_true = list(product([0,1], repeat=args.feature_dim))
        # # reward_space_true.remove((0,0,0,0))
        # # TODO(rohinmshah): These reward spaces have many copies of the same reward function
        # reward_space_true = [np.array(reward) for reward in reward_space_true]
        # reward_space_true = [choice(reward_space_true) for _ in range(size_reward_space_true)]
        # # reward_space_true = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0])]
        # reward_space_proxy = [choice(reward_space_true) for _ in range(size_reward_space_proxy)]
        # # reward_space_proxy = reward_space_true
        # # len_reward_space = len(reward_space_true)
        # # reward_space = [np.array([1,0]),np.array([0,1]), np.array([1,1])]

    # Set up env and agent for gridworld
    elif args.mdp_type == 'gridworld':
        for i in range(10):
            test_grid = GridworldMdp.generate_random(height,width,0.3,feature_dim,None,living_reward=-0.01, print_grid=False)

        grid = GridworldMdp.generate_random(height,width,0.3,feature_dim,None,living_reward=-0.01, print_grid=True)
        mdp = GridworldMdpWithDistanceFeatures(grid, dist_scale, living_reward=-0.01, noise=0, rewards=dummy_rewards)
        agent = OptimalAgent(gamma, num_iters=args.value_iters)



        # Create reward spaces for gridworld
        # reward_space_true = [np.random.multinomial(18, np.ones(args.feature_dim)/18) for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.multinomial(18, np.ones(args.feature_dim)) for _ in xrange(size_reward_space_proxy)]
        reward_space_true = [np.random.randint(-9, 9, size=[args.feature_dim])   for _ in xrange(size_reward_space_true)]
        reward_space_proxy = [np.random.randint(-9, 9, size=[args.feature_dim]) for _ in xrange(size_reward_space_proxy)]
        # reward_space_true = [np.random.dirichlet(np.ones(args.feature_dim)) * args.feature_dim - 1 for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.dirichlet(np.ones(args.feature_dim)) * args.feature_dim - 1 for _ in xrange(size_reward_space_proxy)]

    else:
        raise ValueError('Unknown MDP type: ' + str(args.mdp_type))



    # Set up inference
    env = GridworldEnvironment(mdp)
    inference = InferenceDiscrete(
        agent, mdp, env, beta, reward_space_true, reward_space_proxy,
        num_traject=num_traject, prior=None)

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
    def experiment(query_size):
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
        experiment = Experiment(inference, reward_space_proxy, query_size, num_queries_max, args, choosers, SEED, exp_params)
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
    experiment(query_size)


    'Create interface'
    # omega = [choice(reward_space_true) for _ in range(4)] # replace with chosen omega
    # interface = Interface(omega, agent, env, num_states=num_states)
    # interface.plot()
