from subprocess import call

# Discrete experiments

NUM_EXPERIMENTS = '100'  # Modify this to change the sample size

# choosers = ['greedy_discrete', 'random']
discr_query_sizes = ['2','3','5','10']
# choosers = ['feature_random', 'feature_entropy_search_then_optim', 'feature_entropy_init_none', 'feature_entropy_search', 'feature_entropy_random_init_none']
choosers = ['greedy', 'random', 'exhaustive']
mdp_types = ['gridworld','bandits']
num_iter = {'gridworld': '20', 'bandits': '20'}
beta_both_mdps = '0.5'
num_q_max = '10000'
rsize = '10000'
full_IRD_full_proxy_space = '1'
exp_name = 'no_exp_name'


def run(chooser, qsize, mdp_type, objective='entropy', discretization_size='5', discretization_size_human='5',
        viter='15', rsize=rsize, subsampling='1', num_iter='20', proxy_space_is_true_space=False):
    if mdp_type == 'bandits':
        # Values range from -5 to 5 approximately, so setting beta to 1 makes
        # the worst Q-value e^10 times less likely than the best one
        beta = beta_both_mdps
        beta_planner = '0.5'
        dim = '20'
        # TODO: Set the following to the right values
        lr = '20.'
        num_iters_optim = '10'
    else:
        # Values range from 50-100 when using 25 value iterations.
        beta = beta_both_mdps
        beta_planner = '1'
        dim = '20'
        # TODO: Set the following to the right values
        lr = '20'
        num_iters_optim = '10'

    command = ['python', 'run_IRD.py',
               '-c', chooser,
               '--query_size', qsize,
               '--num_experiments', NUM_EXPERIMENTS,
               '--num_iter', num_iter[mdp_type],
               '--gamma', '1.',
               '--size_true_space', rsize,
               '--size_proxy_space', '100',
               '--seed', '1',
               '--beta', beta,
               '--beta_planner', beta_planner,
               '--num_states', '100',  # Only applies for bandits
               '--dist_scale', '0.5',
               '--num_traject', '1',
               '--num_queries_max', num_q_max,
               '--height', '12',  # Only applies for gridworld
               '--width', '12',  # Only applies for gridworld
               '--lr', lr,  # Doesn't matter, only applies in continuous case
               '--num_iters_optim', num_iters_optim,
               '--value_iters', viter,  # Consider decreasing viters to 10-15 to make the path more important as opposed to ending up at the right goal
               '--mdp_type', mdp_type,
               '--feature_dim', dim,
               '--discretization_size', discretization_size,
               '--discretization_size_human', discretization_size_human,
               '--num_test_envs', '100',
               '--subsampling', subsampling,
               '--num_subsamples','10000',
               '--weighting', '1',
               '--well_spec', '1',
               '--linear_features', '1',
               '--objective', objective,
               '-weights_dist_init', 'normal2',
               '-weights_dist_search', 'normal2',
               '--only_optim_biggest', '1',
               '--proxy_space_is_true_space', proxy_space_is_true_space,
               '--exp_name', exp_name
               ]
    print 'Running command', ' '.join(command)
    call(command)


# Run as usual
def run_discrete():
    for mdp_type in mdp_types:

        run('full', '2', mdp_type, num_iter=num_iter, proxy_space_is_true_space=full_IRD_full_proxy_space)

        for chooser in choosers:
            for qsize in discr_query_sizes:
                run(chooser, qsize, mdp_type, num_iter=num_iter)



# # Run with different rsize and subsampling values
# def run_subsampling():
#     for mdp_type in mdp_types:
#         for rsize in true_reward_space_sizes:
#             if rsize == '10000':
#                 subsampling = '0'
#             else: subsampling = '1'
#
#
#             for objective in objectives:
#                 for chooser in choosers:
#                         for qsize in discr_query_sizes:
#                             run(chooser, qsize, mdp_type, objective, rsize=rsize, subsampling=subsampling)
#                 run('full', '2', mdp_type, objective, rsize=rsize, subsampling=subsampling, num_iter=num_iter)


def run_discrete_optimization():
    for mdp_type in mdp_types:
        for qsize in discr_query_sizes:
            for chooser in ['incremental_optimize', 'joint_optimize']:
                run(chooser, qsize, mdp_type)


def run_continuous():
    for mdp_type in mdp_types:
        for qsize, discretization_size, discretization_size_human in [('3', '3', '3'), ('2', '5', '5'), ('1', '9', '9')]:
            for chooser in choosers:
                run(chooser, qsize, mdp_type,
                    discretization_size=discretization_size,
                    discretization_size_human=discretization_size_human,
                    num_iter=num_iter)

if __name__ == '__main__':
    run_discrete()
