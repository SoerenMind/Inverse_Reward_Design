from subprocess import call

# Discrete experiments

NUM_EXPERIMENTS = '50'  # Modify this to change the sample size

discr_query_sizes = ['5', '2']
choosers_continuous = ['feature_entropy_search_then_optim', 'feature_random', 'feature_entropy_random_init_none'] #'feature_entropy_init_none', 'feature_entropy_search']
choosers_discrete = ['greedy_discrete','random']
mdp_types = ['gridworld','bandits']
num_iter = {'gridworld': '20', 'bandits': '20'}
num_subsamples_full = '5000'; num_subsamples_not_full = '5000'
beta_both_mdps = '0.5'
num_q_max = '10000'
rsize = '1000000'
proxy_space_is_true_space = '0'
exp_name = '21Feb_nonlinear_continuous'


def run(seed, chooser, qsize, mdp_type, num_iter, objective='entropy', discretization_size='5', discretization_size_human='5',
        viter='15', rsize=rsize, subsampling='1', proxy_space_is_true_space='0',
        subs_full=num_subsamples_full, full_IRD_subsample_belief='no', log_objective='1',
        repeated_obj='0', num_obj_if_repeated='50', decorrelate_test_feat='1',
        dist_scale='0.2', euclid_features='1', height='12', width='12',
        num_test_envs='30',beta=beta_both_mdps,
        nonlinear_true_space='0', nonlinear_proxy_space='0', test_misspec_linear_space='0'):
    if mdp_type == 'bandits':
        # Values range from -5 to 5 approximately, so setting beta to 1 makes
        # the worst Q-value e^10 times less likely than the best one
        beta_planner = '0.5'
        dim = '20'
        # TODO: Set the following to the right values
        lr = '20.'
        num_iters_optim = '20'
    else:
        # Values range from 50-100 when using 25 value iterations.
        beta_planner = '1'
        dim = '20'
        # TODO: Set the following to the right values
        lr = '20'
        num_iters_optim = '20'

    command = ['python', 'run_IRD.py',
               '-c', chooser,
               '--query_size', qsize,
               '--num_experiments', NUM_EXPERIMENTS,
               '--num_iter', num_iter[mdp_type],
               '--gamma', '1.',
               '--size_true_space', rsize,
               '--size_proxy_space', '100',
               '--seed', seed,
               '--beta', beta,
               '--beta_planner', beta_planner,
               '--num_states', '100',  # Only applies for bandits
               '--dist_scale', dist_scale,
               '--num_queries_max', num_q_max,
               '--height', height,  # Only applies for gridworld
               '--width', width,    # Only applies for gridworld
               '--lr', lr,
               '--num_iters_optim', num_iters_optim,
               '--value_iters', viter,
               '--mdp_type', mdp_type,
               '--feature_dim', dim,
               '--nonlinear_true_space', nonlinear_true_space,
               '--nonlinear_proxy_space', nonlinear_proxy_space,
               '--discretization_size', discretization_size,
               '--test_misspec_linear_space', test_misspec_linear_space,
               '--discretization_size_human', discretization_size_human,
               '--num_test_envs', num_test_envs,
               '--subsampling', subsampling,
               '--num_subsamples', subs_full if chooser == 'full' else num_subsamples_not_full,
               '--weighting', '1',
               '--well_spec', '1',
               '--euclid_features', euclid_features,
               '--objective', objective,
               '--log_objective', log_objective,
               '-weights_dist_init', 'normal2',
               '-weights_dist_search', 'normal2',
               '--only_optim_biggest', '1',
               '--proxy_space_is_true_space', proxy_space_is_true_space,
               '--full_IRD_subsample_belief', full_IRD_subsample_belief,
               '--exp_name', exp_name,
               '--repeated_obj', repeated_obj,
               '--num_obj_if_repeated', num_obj_if_repeated,
               '--decorrelate_test_feat', decorrelate_test_feat
               ]
    print('Running command', ' '.join(command))
    call(command)
    return seed + 1

def run_nonlinear_discrete():
    seed = 1111
    for mdp_type in mdp_types:
        for test_misspec_linear_space in ['0', '1']:
            seed = run(seed, 'full', '2', mdp_type, num_iter=num_iter, test_misspec_linear_space=test_misspec_linear_space)
            for chooser in choosers_discrete:
                for qsize in discr_query_sizes:
                    seed = run(seed, chooser, qsize, mdp_type, num_iter=num_iter, test_misspec_linear_space=test_misspec_linear_space, nonlinear_true_space='1')



# Run as usual
def run_discrete():
    seed = 2222
    for mdp_type in mdp_types:

        seed = run(seed, 'full', '2', mdp_type, num_iter=num_iter, proxy_space_is_true_space=proxy_space_is_true_space)

        for chooser in choosers_discrete:
            for qsize in discr_query_sizes:
                seed = run(seed, chooser, qsize, mdp_type, num_iter=num_iter)


def run_reward_hacking():
    seed = 3333
    mdp_type = 'gridworld'
    repeated_obj = '1'
    num_obj_if_repeated = '100'
    qsizes = ['2','5']
    height, width = '52', '52'
    viter = str(int(int(height)*1.5))
    beta = str(7.5 / float(viter))  # Decrease beta for higher viter. Make prop to num objects too?
    num_test_envs = '25'


    for dist_scale in ['0.1', '0.3', '1']:
        for decorrelate_test_feat in ['1','0']:

            seed = run(seed, 'full', '2', mdp_type, num_iter=num_iter,
                repeated_obj=repeated_obj, num_obj_if_repeated=num_obj_if_repeated, dist_scale=dist_scale,
                height=height, width=width, num_test_envs=num_test_envs, viter=viter, beta=beta, decorrelate_test_feat=decorrelate_test_feat)

            for chooser in ['greedy_discrete', 'random']:
                for qsize in qsizes:
                    seed = run(seed, chooser, qsize, mdp_type, num_iter=num_iter,
                        repeated_obj=repeated_obj, num_obj_if_repeated=num_obj_if_repeated, dist_scale=dist_scale,
                        height=height, width=width, num_test_envs=num_test_envs, viter=viter, beta=beta, decorrelate_test_feat=decorrelate_test_feat)


def run_full():
    seed = 4444
    for mdp_type in mdp_types:
        for num_subsamples_full in ['1000', '500','100','50','10','5','2','10000']:
            for full_IRD_subsample_belief in ['yes','uniform','no']:
                if full_IRD_subsample_belief == 'no':
                    seed = run(seed, 'full', '2', mdp_type, num_iter=num_iter, proxy_space_is_true_space=proxy_space_is_true_space,
                        subs_full=num_subsamples_full,full_IRD_subsample_belief=full_IRD_subsample_belief,size_proxy_space=num_subsamples_full)
                seed = run(seed, 'full', '2', mdp_type, num_iter=num_iter, proxy_space_is_true_space=proxy_space_is_true_space,
                    subs_full=num_subsamples_full,full_IRD_subsample_belief=full_IRD_subsample_belief)
        # Interesting question: How high can 'uniform' go before it gets worse? (could be pretty high)
        # Test with smaller r_size if the turning point turns out >100.
        # Hypothesis: 'yes' gets monotonely better with larger sizes bc only top samples matter (but flattens out quite quickly)


def run_objectives():
    seed = 5555

    # Continuous
    # qsize = '3'
    # discretization_size = '3'
    # discretization_size_human = '5'
    # chooser = 'feature_entropy_search_then_optim'

    # Discrete
    chooser = 'greedy_discrete'
    for mdp_type in mdp_types:
        # for log_objective in ['1' , '0']:
        for qsize in discr_query_sizes:
            for objective in ['query_neg_entropy','entropy']:
                seed = run(seed, chooser, qsize, mdp_type,
                    # discretization_size=discretization_size, discretization_size_human=discretization_size_human, , log_objective=log_objective
                    num_iter=num_iter, objective=objective)

# # Run with different rsize and subsampling values
# def run_subsampling():
#     seed = 6666
#     for mdp_type in mdp_types:
#         for rsize in true_reward_space_sizes:
#             if rsize == '10000':
#                 subsampling = '0'
#             else: subsampling = '1'
#
#
#             for objective in objectives:
#                 for chooser in choosers_discrete:
#                         for qsize in discr_query_sizes:
#                             seed = run(seed, chooser, qsize, mdp_type, objective, rsize=rsize, subsampling=subsampling)
#                 run('full', '2', mdp_type, objective, rsize=rsize, subsampling=subsampling, num_iter=num_iter)


def run_discrete_optimization():
    seed = 7777
    for mdp_type in mdp_types:
        for qsize in discr_query_sizes:
            for chooser in ['incremental_optimize', 'joint_optimize']:
                seed = run(seed, chooser, qsize, mdp_type)


def run_continuous():
    seed = 8888
    for mdp_type in mdp_types:
        for qsize, discretization_size, discretization_size_human in [('3', '3', '5'), ('2', '5', '9'), ('1', '9', '18')]:
            for chooser in choosers_continuous:
                seed = run(seed, chooser, qsize, mdp_type,
                    discretization_size=discretization_size,
                    discretization_size_human=discretization_size_human,
                    num_iter=num_iter)

def run_nonlinear_continuous():
    seed = 9999
    for test_misspec_linear_space, nonlin_true, nonlin_proxy in [('0', '1', '0'), ('1', '0', '0'), ('0', '1', '1')]:
        for mdp_type in mdp_types:
            for qsize, discretization_size, discretization_size_human in [('3', '3', '5'), ('2', '5', '9'), ('1', '9', '18')]:
                seed = run(seed, 'feature_entropy_search_then_optim', qsize,
                           mdp_type, num_iter=num_iter,
                           test_misspec_linear_space=test_misspec_linear_space,
                           nonlinear_true_space=nonlin_true,
                           nonlinear_proxy_space=nonlin_proxy)

if __name__ == '__main__':
    # run_objectives()
    run_nonlinear_continuous()
    # run_reward_hacking()
    # run_continuous()
    # run_discrete()
