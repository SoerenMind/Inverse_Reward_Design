import time

print('importing')

start = time.clock()
import datetime
print('time: '+str(datetime.datetime.now()))
import numpy as np
from inference_class import Inference
from gridworld import GridworldEnvironment, NStateMdpHardcodedFeatures, NStateMdpGaussianFeatures,\
    NStateMdpRandomGaussianFeatures, GridworldMdpWithDistanceFeatures, GridworldMdp
from query_chooser_class import Experiment
from random import choice, seed
import copy
from utils import Distribution
import sys
import argparse
import tensorflow as tf


print('Time to import: {deltat}'.format(deltat=time.clock() - start))




def pprint(y):
    print(y)
    return y



# ==================================================================================================== #
# ==================================================================================================== #
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('choosers',type=list,default='[greedy_entropy]')
    parser.add_argument('-c','--c', action='append', required=True) # c for choosers
    parser.add_argument('--exp_name',type=str,default='no_exp_name')
    parser.add_argument('--query_size',type=int,default=3)
    parser.add_argument('--num_experiments',type=int,default=2)
    parser.add_argument('--num_iter',type=int,default=20)    # number of queries asked
    parser.add_argument('--gamma',type=float,default=1.) # discount
    parser.add_argument('--size_true_space',type=int,default=10000)
    parser.add_argument('--size_proxy_space',type=int,default=100)  # Sample subspace for exhaustive
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--beta',type=float,default=0.2)
    parser.add_argument('--num_states',type=int,default=100)  # 10 options if env changes over time, 100 otherwise
    parser.add_argument('--dist_scale',type=float,default=0.2) # test briefly to get ent down
    parser.add_argument('--height',type=int,default=12)
    parser.add_argument('--width',type=int,default=12)
    parser.add_argument('--value_iters',type=int,default=15)    # max_reward / (1-gamma) or height+width
    parser.add_argument('--mdp_type',type=str,default='gridworld')
    parser.add_argument('--feature_dim',type=int,default=20)
    parser.add_argument('--nonlinear_true_space',type=int,default=1)   # if 1, the true reward space is nonlinear functions
    parser.add_argument('--nonlinear_proxy_space',type=int,default=0)   # if 1, the proxy reward space is nonlinear functions
    parser.add_argument('--test_misspec_linear_space',type=int,default=0)   # if 1, the true reward is nonlinear even though the true reward space is linear
    parser.add_argument('--num_test_envs',type=int,default=20)    # 10 if positions fixed, 100 otherwise. Can be reduced though because it's expensive.
    parser.add_argument('--well_spec',type=int,default=1)    # default is well-specified
    parser.add_argument('--subsampling',type=int,default=1)
    parser.add_argument('--num_subsamples',type=int,default=10000)
    parser.add_argument('--weighting',type=int,default=1)
    parser.add_argument('--euclid_features',type=int,default=1) # if 0, gridworld features are RBF (not euclidean) distances to object
    parser.add_argument('--objective',type=str,default='entropy')
    parser.add_argument('--log_objective',type=int,default=1)
    parser.add_argument('--rational_test_planner',type=int,default=1)
    # args for experiment with correlated features
    parser.add_argument('--repeated_obj',type=int,default=0)  # Creates gridworld with k object types, k features, and num_objects >= k objects
    parser.add_argument('--num_obj_if_repeated',type=int,default=50)  # Usually feature_dim_proxy is # of objects except for correlated features experiment. Must be > feature_dim_proxy
    parser.add_argument('--decorrelate_test_feat',type=int,default=1)

    # args for optimization
    parser.add_argument('-weights_dist_init',type=str,default='normal2')
    parser.add_argument('-weights_dist_search',type=str,default='normal2')
    parser.add_argument('--lr',type=float,default=20)  # Learning rate
    parser.add_argument('--only_optim_biggest',type=int,default=1)
    parser.add_argument('--num_iters_optim',type=int,default=10)
    parser.add_argument('--beta_planner',type=float,default=0.5) # 1 for small version of results
    parser.add_argument('--num_queries_max',type=int,default=2000)
    parser.add_argument('--discretization_size',type=int,default=5) # for continuous query selection
    parser.add_argument('--discretization_size_human',type=int,default=5)   # for continuous query actually posed

    # args for testing full IRD
    parser.add_argument('--proxy_space_is_true_space', type=int, default=0)
    parser.add_argument('--full_IRD_subsample_belief', type=str, default='no')  # other options: yes, uniform



    args = parser.parse_args()
    #print(args)
    # assert args.discretization_size % 2 == 1

    # Experiment description
    adapted_description = False
    # #print("Adapted description: ", adapted_description)

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
    num_experiments = args.num_experiments
    num_iter_per_experiment = args.num_iter #; print('num iter = {i}'.format(i=num_iter_per_experiment))
    # Params for Gridworld
    gamma = args.gamma
    query_size = args.query_size
    dist_scale = args.dist_scale
    height = args.height
    width = args.width
    num_iters_optim = args.num_iters_optim
    p_wall = 0.35 if args.height < 20 else 0.1
    # If either space is nonlinear, then planning has to be done with nonlinear features
    args.nonlinear_planner = args.nonlinear_true_space or args.nonlinear_proxy_space
    # Even if both spaces are linear, if we're using a nonlinear true reward (i.e. the true reward space is misspecified), we'll need nonlinear features
    args.nonlinear_features = args.nonlinear_planner or args.test_misspec_linear_space
    nonlinear_feature_dim = (args.feature_dim + 1) ** 2
    args.feature_dim_planner = nonlinear_feature_dim if args.nonlinear_planner else args.feature_dim
    args.feature_dim_true = nonlinear_feature_dim if args.nonlinear_true_space else args.feature_dim
    args.feature_dim_proxy = nonlinear_feature_dim if args.nonlinear_proxy_space else args.feature_dim
    if args.test_misspec_linear_space:
        assert not args.nonlinear_true_space

    # These will be in the folder name of the log
    exp_params = {
        # 'rational_test_planner': args.rational_test_planner,
        'qsize': query_size,
        'mdp': args.mdp_type,
        'dim': args.feature_dim,
        'misspec_linear': args.test_misspec_linear_space,
        'dsize': args.discretization_size,
        'size_true': size_reward_space_true,
        'size_proxy': size_reward_space_proxy,
        'seed': SEED,
        'beta': beta,
        'exp_name': args.exp_name,
        # 'num_states': num_states,
        'dist_scale': dist_scale,
        # 'n_q_max': num_queries_max,
        # 'num_iters_optim': num_iters_optim,
        # 'well_spec': args.well_spec,
        # 'subsamp': args.subsampling,
        'num_subsamp': args.num_subsamples,
        # 'weighting': args.weighting,
        # 'viters': args.value_iters,
        # 'euclfeat': args.euclid_features,
        'nonlin_true': args.nonlinear_true_space,
        'nonlin_proxy': args.nonlinear_proxy_space,
        'objective': args.objective,
        # 'w_dist_i': args.weights_dist_init,
        # 'w_dist_s': args.weights_dist_search,
        # 'optim_big': args.only_optim_biggest,
        # 'rational_test': args.rational_test_planner
        'proxy_is_true': args.proxy_space_is_true_space,
        'full_IRD_subs': args.full_IRD_subsample_belief,
        # 'corr_feat': args.repeated_obj,
        # 'num_obj_if_corr': args.num_obj_if_repeated
    }

    'Sample true rewards and reward spaces'
    reward_space_true = np.array(np.random.randint(-9, 10, size=[size_reward_space_true, args.feature_dim_true]), dtype=np.int16)
    if args.test_misspec_linear_space:
        true_rewards = [np.random.randint(-9, 10, size=[nonlinear_feature_dim]) for _ in range(num_experiments)]
    elif not args.well_spec:
        true_rewards = [np.random.randint(-9, 10, size=[args.feature_dim_true]) for _ in range(num_experiments)]
    else:
        true_rewards = [choice(reward_space_true) for _ in range(num_experiments)]
        if args.repeated_obj:
            # Set values of proxy and goal
            for i, reward in enumerate(true_rewards):
                for j in range(args.feature_dim):
                    if reward[j] > 7: reward[j] = np.random.randint(-9, 6)
                reward[-1] = 9
                reward[-2] = -2
                true_rewards[i] = reward
                reward_space_true[i,:] = reward
    prior_avg = -0.5 * np.ones(args.feature_dim_true) + 1e-4 * np.random.exponential(1,args.feature_dim_true) # post_avg for uniform prior + noise

    # Set up env and agent for NStateMdp
    if args.mdp_type == 'bandits':

        'Create train and test MDPs'
        test_mdps = []
        for i in range(args.num_test_envs):
            mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                            feature_dim=args.feature_dim,
                                            num_states_reachable=num_states, SEED=SEED+i*50)
            test_mdps.append(mdp)

        train_mdps = []
        for i in range(num_experiments):
            mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                            feature_dim=args.feature_dim,
                                            num_states_reachable=num_states, SEED=SEED+i*50+1)
            train_mdps.append(mdp)

        'Create train and test inferences'
        test_inferences = []
        for i in range(args.num_test_envs):
            mdp = test_mdps[i]
            env = GridworldEnvironment(mdp)
            inference = Inference(
                mdp, env, beta, reward_space_true, reward_space_proxy=[])
            test_inferences.append(inference)

        train_inferences = []
        for i in range(num_experiments):
            mdp = train_mdps[i]
            env = GridworldEnvironment(mdp)
            reward_space_proxy = reward_space_true if args.proxy_space_is_true_space \
                else np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim_proxy])

            # give weight zero to nonlinear features if necessary
            if args.nonlinear_true_space and not args.nonlinear_proxy_space:
                reward_space_proxy = np.concatenate([reward_space_proxy, np.zeros((size_reward_space_proxy, args.feature_dim_true - args.feature_dim_proxy))], axis=-1)

            inference = Inference(
                mdp, env, beta, reward_space_true, reward_space_proxy)
            train_inferences.append(inference)


    # Set up env and agent for gridworld
    elif args.mdp_type == 'gridworld':
        'Create train and test MDPs'
        test_inferences = []
        for i in range(args.num_test_envs):
            test_grid, test_goals = GridworldMdp.generate_random(args,height,width,0.25,args.feature_dim,None,
                                        living_reward=-0.01, print_grid=False, decorrelate=args.decorrelate_test_feat)
            mdp = GridworldMdpWithDistanceFeatures(test_grid, test_goals, args, dist_scale, living_reward=-0.01, noise=0)
            env = GridworldEnvironment(mdp)
            inference = Inference(
                mdp, env, beta, reward_space_true, reward_space_proxy=[])

            test_inferences.append(inference)


        train_inferences = []
        for j in range(num_experiments):
            grid, goals = GridworldMdp.generate_random(args,height,width,0.25,args.feature_dim,None,living_reward=-0.01, print_grid=False)
            mdp = GridworldMdpWithDistanceFeatures(grid, goals, args, dist_scale, living_reward=-0.01, noise=0)
            env = GridworldEnvironment(mdp)
            reward_space_proxy = reward_space_true if args.proxy_space_is_true_space \
                else np.random.randint(-9, 10, size=[size_reward_space_proxy, args.feature_dim_proxy])

            # give weight zero to nonlinear features if necessary
            if args.nonlinear_true_space and not args.nonlinear_proxy_space:
                reward_space_proxy = np.concatenate([reward_space_proxy, np.zeros((size_reward_space_proxy, args.feature_dim_true - args.feature_dim_proxy))], axis=-1)

            inference = Inference(
                mdp, env, beta, reward_space_true, reward_space_proxy)
            train_inferences.append(inference)


    else:
        raise ValueError('Unknown MDP type: ' + str(args.mdp_type))



    # Run experiment
    def run_experiment(query_size, train_inferences, test_inferences, true_rewards, prior_avg):
        experiment = Experiment(true_rewards, query_size, num_queries_max,
                                args, choosers, SEED, exp_params, train_inferences, test_inferences, prior_avg)
        results = experiment.get_experiment_stats(num_iter_per_experiment, num_experiments)


        print('__________________________Finished experiment__________________________')

    run_experiment(query_size, train_inferences, test_inferences, true_rewards, prior_avg)
